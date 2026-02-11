# DreamBooth training
# XXX dropped option: fine_tune

import argparse
import itertools
import math
import os
from multiprocessing import Value
import toml

from tqdm import tqdm

import torch
from library import deepspeed_utils, strategy_base
from library.device_utils import init_ipex, clean_memory_on_device


init_ipex()

from accelerate.utils import set_seed
from diffusers import DDPMScheduler

import library.train_util as train_util
import library.config_util as config_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    apply_snr_weight,
    get_weighted_text_embeddings,
    prepare_scheduler_for_custom_training,
    pyramid_noise_like,
    apply_noise_offset,
    scale_v_prediction_loss_like_noise_prediction,
    apply_debiased_estimation,
    apply_masked_loss,
)
from library.utils import setup_logging, add_logging_arguments
import library.strategy_sd as strategy_sd

setup_logging()
import logging

logger = logging.getLogger(__name__)

# perlin_noise,


def train(args):
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, False)
    deepspeed_utils.prepare_deepspeed_args(args)
    setup_logging(args, reset=True)

    cache_latents = args.cache_latents

    if args.seed is not None:
        set_seed(args.seed)  # 乱数系列を初期化する

    tokenize_strategy = strategy_sd.SdTokenizeStrategy(args.v2, args.max_token_length, args.tokenizer_cache_dir)
    strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)

    # prepare caching strategy: this must be set before preparing dataset. because dataset may use this strategy for initialization.
    latents_caching_strategy = strategy_sd.SdSdxlLatentsCachingStrategy(
        False, args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
    )
    strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    # データセットを準備する
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, False, args.masked_loss, True))
        if args.dataset_config is not None:
            logger.info(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "reg_data_dir"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning(
                    "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                        ", ".join(ignored)
                    )
                )
        else:
            user_config = {
                "datasets": [
                    {"subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(args.train_data_dir, args.reg_data_dir)}
                ]
            }

        blueprint = blueprint_generator.generate(user_config, args)
        train_dataset_group, val_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args)
        val_dataset_group = None

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    if args.no_token_padding:
        train_dataset_group.disable_token_padding()

    train_dataset_group.verify_bucket_reso_steps(64)

    if args.debug_dataset:
        train_util.debug_dataset(train_dataset_group)
        return

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

    # acceleratorを準備する
    logger.info("prepare accelerator")

    if args.gradient_accumulation_steps > 1:
        logger.warning(
            f"gradient_accumulation_steps is {args.gradient_accumulation_steps}. accelerate does not support gradient_accumulation_steps when training multiple models (U-Net and Text Encoder), so something might be wrong"
        )
        logger.warning(
            f"gradient_accumulation_stepsが{args.gradient_accumulation_steps}に設定されています。accelerateは複数モデル（U-NetおよびText Encoder）の学習時にgradient_accumulation_stepsをサポートしていないため結果は未知数です"
        )

    accelerator = train_util.prepare_accelerator(args)

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    # モデルを読み込む
    text_encoder, vae, unet, load_stable_diffusion_format = train_util.load_target_model(args, weight_dtype, accelerator)

    # verify load/save model formats
    if load_stable_diffusion_format:
        src_stable_diffusion_ckpt = args.pretrained_model_name_or_path
        src_diffusers_model_path = None
    else:
        src_stable_diffusion_ckpt = None
        src_diffusers_model_path = args.pretrained_model_name_or_path

    if args.save_model_as is None:
        save_stable_diffusion_format = load_stable_diffusion_format
        use_safetensors = args.use_safetensors
    else:
        save_stable_diffusion_format = args.save_model_as.lower() == "ckpt" or args.save_model_as.lower() == "safetensors"
        use_safetensors = args.use_safetensors or ("safetensors" in args.save_model_as.lower())

    # モデルに xformers とか memory efficient attention を組み込む
    train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)

    # 学習を準備する
    if cache_latents:
        vae.to(accelerator.device, dtype=vae_dtype)
        vae.requires_grad_(False)
        vae.eval()

        train_dataset_group.new_cache_latents(vae, accelerator)

        vae.to("cpu")
        clean_memory_on_device(accelerator.device)

        accelerator.wait_for_everyone()

    text_encoding_strategy = strategy_sd.SdTextEncodingStrategy(args.clip_skip)
    strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

    # 学習を準備する：モデルを適切な状態にする
    train_text_encoder = args.stop_text_encoder_training is None or args.stop_text_encoder_training >= 0
    unet.requires_grad_(True)  # 念のため追加
    text_encoder.requires_grad_(train_text_encoder)
    if not train_text_encoder:
        accelerator.print("Text Encoder is not trained.")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable()

    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=weight_dtype)

    # 学習に必要なクラスを準備する
    accelerator.print("prepare optimizer, data loader etc.")
    if train_text_encoder:
        if args.learning_rate_te is None:
            # wightout list, adamw8bit is crashed
            trainable_params = list(itertools.chain(unet.parameters(), text_encoder.parameters()))
        else:
            trainable_params = [
                {"params": list(unet.parameters()), "lr": args.learning_rate},
                {"params": list(text_encoder.parameters()), "lr": args.learning_rate_te},
            ]
    else:
        trainable_params = unet.parameters()

    _, _, optimizer = train_util.get_optimizer(args, trainable_params)

    # prepare dataloader
    # strategies are set here because they cannot be referenced in another process. Copy them with the dataset
    # some strategies can be None
    train_dataset_group.set_current_strategies()

    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    # 学習ステップ数を計算する
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        accelerator.print(
            f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}"
        )

    # データセット側にも学習ステップを送信
    train_dataset_group.set_max_train_steps(args.max_train_steps)

    if args.stop_text_encoder_training is None:
        args.stop_text_encoder_training = args.max_train_steps + 1  # do not stop until end

    # lr schedulerを用意する TODO gradient_accumulation_stepsの扱いが何かおかしいかもしれない。後で確認する
    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # 実験的機能：勾配も含めたfp16学習を行う　モデル全体をfp16にする
    if args.full_fp16:
        assert (
            args.mixed_precision == "fp16"
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        accelerator.print("enable full fp16 training.")
        unet.to(weight_dtype)
        text_encoder.to(weight_dtype)

    # acceleratorがなんかよろしくやってくれるらしい
    if args.deepspeed:
        if args.train_text_encoder:
            ds_model = deepspeed_utils.prepare_deepspeed_model(args, unet=unet, text_encoder=text_encoder)
        else:
            ds_model = deepspeed_utils.prepare_deepspeed_model(args, unet=unet)
        ds_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            ds_model, optimizer, train_dataloader, lr_scheduler
        )
        training_models = [ds_model]

    else:
        if train_text_encoder:
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
            training_models = [unet, text_encoder]
        else:
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)
            training_models = [unet]

    if not train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)  # to avoid 'cpu' vs 'cuda' error

    # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
    if args.full_fp16:
        train_util.patch_accelerator_for_fp16_training(accelerator)

    # resumeする
    train_util.resume_from_local_or_hf_if_specified(accelerator, args)

    # epoch数を計算する
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

    # 学習する
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    accelerator.print("running training / 学習開始")
    accelerator.print(f"  num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset_group.num_train_images}")
    accelerator.print(f"  num reg images / 正則化画像の数: {train_dataset_group.num_reg_images}")
    accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
    accelerator.print(f"  batch size per device / バッチサイズ: {args.train_batch_size}")
    accelerator.print(
        f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}"
    )
    accelerator.print(f"  gradient ccumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )
    prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
    if args.zero_terminal_snr:
        custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            "dreambooth" if args.log_tracker_name is None else args.log_tracker_name,
            config=train_util.get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )

    # For --sample_at_first
    train_util.sample_images(
        accelerator, args, 0, global_step, accelerator.device, vae, tokenize_strategy.tokenizer, text_encoder, unet
    )
    if len(accelerator.trackers) > 0:
        # log empty object to commit the sample images to wandb
        accelerator.log({}, step=0)

    loss_recorder = train_util.LossRecorder()
    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        # 指定したステップ数までText Encoderを学習する：epoch最初の状態
        unet.train()
        # train==True is required to enable gradient_checkpointing
        if args.gradient_checkpointing or global_step < args.stop_text_encoder_training:
            text_encoder.train()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step
            # 指定したステップ数でText Encoderの学習を止める
            if global_step == args.stop_text_encoder_training:
                accelerator.print(f"stop text encoder training at step {global_step}")
                if not args.gradient_checkpointing:
                    text_encoder.train(False)
                text_encoder.requires_grad_(False)
                if len(training_models) == 2:
                    training_models = training_models[0]  # remove text_encoder from training_models

            with accelerator.accumulate(*training_models):
                with torch.no_grad():
                    # latentに変換
                    if cache_latents:
                        latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
                    else:
                        latents = vae.encode(batch["images"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * 0.18215
                b_size = latents.shape[0]

                # Get the text embedding for conditioning
                with torch.set_grad_enabled(global_step < args.stop_text_encoder_training):
                    if args.weighted_captions:
                        input_ids_list, weights_list = tokenize_strategy.tokenize_with_weights(batch["captions"])
                        encoder_hidden_states = text_encoding_strategy.encode_tokens_with_weights(
                            tokenize_strategy, [text_encoder], input_ids_list, weights_list
                        )[0]
                    else:
                        input_ids = batch["input_ids_list"][0].to(accelerator.device)
                        encoder_hidden_states = text_encoding_strategy.encode_tokens(
                            tokenize_strategy, [text_encoder], [input_ids]
                        )[0]
                    if args.full_fp16:
                        encoder_hidden_states = encoder_hidden_states.to(weight_dtype)

                # Sample noise, sample a random timestep for each image, and add noise to the latents,
                # with noise offset and/or multires noise if specified
                noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents)

                # Predict the noise residual
                with accelerator.autocast():
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if args.v_parameterization:
                    # v-parameterization training
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)
                loss = train_util.conditional_loss(noise_pred.float(), target.float(), args.loss_type, "none", huber_c)
                if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                    loss = apply_masked_loss(loss, batch)
                loss = loss.mean([1, 2, 3])

                loss_weights = batch["loss_weights"]  # 各sampleごとのweight
                loss = loss * loss_weights

                if args.min_snr_gamma:
                    loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma, args.v_parameterization)
                if args.scale_v_pred_loss_like_noise_pred:
                    loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                if args.debiased_estimation_loss:
                    loss = apply_debiased_estimation(loss, timesteps, noise_scheduler, args.v_parameterization)

                loss = loss.mean()  # 平均なのでbatch_sizeで割る必要なし

                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                    if train_text_encoder:
                        params_to_clip = itertools.chain(unet.parameters(), text_encoder.parameters())
                    else:
                        params_to_clip = unet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                train_util.sample_images(
                    accelerator, args, None, global_step, accelerator.device, vae, tokenize_strategy.tokenizer, text_encoder, unet
                )

                # 指定ステップごとにモデルを保存
                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                        train_util.save_sd_model_on_epoch_end_or_stepwise(
                            args,
                            False,
                            accelerator,
                            src_path,
                            save_stable_diffusion_format,
                            use_safetensors,
                            save_dtype,
                            epoch,
                            num_train_epochs,
                            global_step,
                            accelerator.unwrap_model(text_encoder),
                            accelerator.unwrap_model(unet),
                            vae,
                        )

            current_loss = loss.detach().item()
            if len(accelerator.trackers) > 0:
                logs = {"loss": current_loss}
                train_util.append_lr_to_logs(logs, lr_scheduler, args.optimizer_type, including_unet=True)
                accelerator.log(logs, step=global_step)

            loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
            avr_loss: float = loss_recorder.moving_average
            logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if len(accelerator.trackers) > 0:
            logs = {"loss/epoch": loss_recorder.moving_average}
            accelerator.log(logs, step=epoch + 1)

        accelerator.wait_for_everyone()

        if args.save_every_n_epochs is not None:
            if accelerator.is_main_process:
                # checking for saving is in util
                src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                train_util.save_sd_model_on_epoch_end_or_stepwise(
                    args,
                    True,
                    accelerator,
                    src_path,
                    save_stable_diffusion_format,
                    use_safetensors,
                    save_dtype,
                    epoch,
                    num_train_epochs,
                    global_step,
                    accelerator.unwrap_model(text_encoder),
                    accelerator.unwrap_model(unet),
                    vae,
                )

        train_util.sample_images(
            accelerator, args, epoch + 1, global_step, accelerator.device, vae, tokenize_strategy.tokenizer, text_encoder, unet
        )

    is_main_process = accelerator.is_main_process
    if is_main_process:
        unet = accelerator.unwrap_model(unet)
        text_encoder = accelerator.unwrap_model(text_encoder)

    accelerator.end_training()

    if is_main_process and (args.save_state or args.save_state_on_train_end):
        train_util.save_state_on_train_end(args, accelerator)

    del accelerator  # この後メモリを使うのでこれは消す

    if is_main_process:
        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
        train_util.save_sd_model_on_train_end(
            args, src_path, save_stable_diffusion_format, use_safetensors, save_dtype, epoch, global_step, text_encoder, unet, vae
        )
        logger.info("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, False, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)

    parser.add_argument(
        "--learning_rate_te",
        type=float,
        default=None,
        help="learning rate for text encoder, default is same as unet / Text Encoderの学習率、デフォルトはunetと同じ",
    )
    parser.add_argument(
        "--no_token_padding",
        action="store_true",
        help="disable token padding (same as Diffuser's DreamBooth) / トークンのpaddingを無効にする（Diffusers版DreamBoothと同じ動作）",
    )
    parser.add_argument(
        "--stop_text_encoder_training",
        type=int,
        default=None,
        help="steps to stop text encoder training, -1 for no training / Text Encoderの学習を止めるステップ数、-1で最初から学習しない",
    )
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    train(args)
