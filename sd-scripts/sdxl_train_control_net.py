import argparse
import math
import os
import random
from multiprocessing import Value
import toml

from tqdm import tqdm

import torch
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from accelerate.utils import set_seed
from accelerate import init_empty_weights
from diffusers import DDPMScheduler
from diffusers.utils.torch_utils import is_compiled_module
from safetensors.torch import load_file
from library import (
    deepspeed_utils,
    sai_model_spec,
    sdxl_model_util,
    sdxl_train_util,
    strategy_base,
    strategy_sd,
    strategy_sdxl,
    sai_model_spec
)

import library.train_util as train_util
import library.config_util as config_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.huggingface_util as huggingface_util
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    add_v_prediction_like_loss,
    apply_snr_weight,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    apply_debiased_estimation,
)
from library.sdxl_original_control_net import SdxlControlNet, SdxlControlledUNet
from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)


# TODO 他のスクリプトと共通化する
def generate_step_logs(args: argparse.Namespace, current_loss, avr_loss, lr_scheduler):
    logs = {
        "loss/current": current_loss,
        "loss/average": avr_loss,
        "lr": lr_scheduler.get_last_lr()[0],
    }

    if args.optimizer_type.lower().startswith("DAdapt".lower()):
        logs["lr/d*lr"] = lr_scheduler.optimizers[-1].param_groups[0]["d"] * lr_scheduler.optimizers[-1].param_groups[0]["lr"]

    return logs


def train(args):
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    sdxl_train_util.verify_sdxl_training_args(args)
    setup_logging(args, reset=True)

    cache_latents = args.cache_latents
    use_user_config = args.dataset_config is not None

    if args.seed is None:
        args.seed = random.randint(0, 2**32)
    set_seed(args.seed)

    tokenize_strategy = strategy_sdxl.SdxlTokenizeStrategy(args.max_token_length, args.tokenizer_cache_dir)
    strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
    tokenizer1, tokenizer2 = tokenize_strategy.tokenizer1, tokenize_strategy.tokenizer2  # this is used for sampling images

    # prepare caching strategy: this must be set before preparing dataset. because dataset may use this strategy for initialization.
    latents_caching_strategy = strategy_sd.SdSdxlLatentsCachingStrategy(
        False, args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
    )
    strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    # データセットを準備する
    blueprint_generator = BlueprintGenerator(ConfigSanitizer(False, False, True, True))
    if use_user_config:
        logger.info(f"Load dataset config from {args.dataset_config}")
        user_config = config_util.load_user_config(args.dataset_config)
        ignored = ["train_data_dir", "conditioning_data_dir"]
        if any(getattr(args, attr) is not None for attr in ignored):
            logger.warning(
                "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                    ", ".join(ignored)
                )
            )
    else:
        user_config = {
            "datasets": [
                {
                    "subsets": config_util.generate_controlnet_subsets_config_by_subdirs(
                        args.train_data_dir,
                        args.conditioning_data_dir,
                        args.caption_extension,
                    )
                }
            ]
        }

    blueprint = blueprint_generator.generate(user_config, args)
    train_dataset_group, val_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(32)

    if args.debug_dataset:
        train_dataset_group.set_current_strategies()  # dasaset needs to know the strategies explicitly
        train_util.debug_dataset(train_dataset_group)
        return
    if len(train_dataset_group) == 0:
        logger.error(
            "No data found. Please verify arguments (train_data_dir must be the parent of folders with images) / 画像がありません。引数指定を確認してください（train_data_dirには画像があるフォルダではなく、画像があるフォルダの親フォルダを指定する必要があります）"
        )
        return

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"
    else:
        logger.warning(
            "WARNING: random_crop is not supported yet for ControlNet training / ControlNetの学習ではrandom_cropはまだサポートされていません"
        )

    if args.cache_text_encoder_outputs:
        assert (
            train_dataset_group.is_text_encoder_output_cacheable()
        ), "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / Text Encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

    # acceleratorを準備する
    logger.info("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)
    is_main_process = accelerator.is_main_process

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    # モデルを読み込む
    (
        load_stable_diffusion_format,
        text_encoder1,
        text_encoder2,
        vae,
        unet,
        logit_scale,
        ckpt_info,
    ) = sdxl_train_util.load_target_model(args, accelerator, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, weight_dtype)

    unet.to(accelerator.device)  # reduce main memory usage

    # convert U-Net to Controlled U-Net
    logger.info("convert U-Net to Controlled U-Net")
    unet_sd = unet.state_dict()
    with init_empty_weights():
        unet = SdxlControlledUNet()
    unet.load_state_dict(unet_sd, strict=True, assign=True)
    del unet_sd

    # make control net
    logger.info("make ControlNet")
    if args.controlnet_model_name_or_path:
        with init_empty_weights():
            control_net = SdxlControlNet()

        logger.info(f"load ControlNet from {args.controlnet_model_name_or_path}")
        filename = args.controlnet_model_name_or_path
        if os.path.splitext(filename)[1] == ".safetensors":
            state_dict = load_file(filename)
        else:
            state_dict = torch.load(filename)
        info = control_net.load_state_dict(state_dict, strict=True, assign=True)
        logger.info(f"ControlNet loaded from {filename}: {info}")
    else:
        control_net = SdxlControlNet()

        logger.info("initialize ControlNet from U-Net")
        info = control_net.init_from_unet(unet)
        logger.info(f"ControlNet initialized from U-Net: {info}")

    # 学習を準備する
    if cache_latents:
        vae.to(accelerator.device, dtype=vae_dtype)
        vae.requires_grad_(False)
        vae.eval()

        train_dataset_group.new_cache_latents(vae, accelerator)

        vae.to("cpu")
        clean_memory_on_device(accelerator.device)

        accelerator.wait_for_everyone()

    text_encoding_strategy = strategy_sdxl.SdxlTextEncodingStrategy()
    strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

    # TextEncoderの出力をキャッシュする
    if args.cache_text_encoder_outputs:
        # Text Encodes are eval and no grad
        text_encoder_output_caching_strategy = strategy_sdxl.SdxlTextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk, None, False
        )
        strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_output_caching_strategy)

        text_encoder1.to(accelerator.device)
        text_encoder2.to(accelerator.device)
        with accelerator.autocast():
            train_dataset_group.new_cache_text_encoder_outputs([text_encoder1, text_encoder2], accelerator)

        accelerator.wait_for_everyone()

    # モデルに xformers とか memory efficient attention を組み込む
    # train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
    if args.xformers:
        unet.set_use_memory_efficient_attention(True, False)
        control_net.set_use_memory_efficient_attention(True, False)
    elif args.sdpa:
        unet.set_use_sdpa(True)
        control_net.set_use_sdpa(True)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        control_net.enable_gradient_checkpointing()

    # 学習に必要なクラスを準備する
    accelerator.print("prepare optimizer, data loader etc.")

    trainable_params = []
    ctrlnet_params = []
    unet_params = []
    for name, param in control_net.named_parameters():
        if name.startswith("controlnet_"):
            ctrlnet_params.append(param)
        else:
            unet_params.append(param)
    trainable_params.append({"params": ctrlnet_params, "lr": args.control_net_lr})
    trainable_params.append({"params": unet_params, "lr": args.learning_rate})
    all_params = ctrlnet_params + unet_params

    logger.info(f"trainable params count: {len(all_params)}")
    logger.info(f"number of trainable parameters: {sum(p.numel() for p in all_params)}")

    _, _, optimizer = train_util.get_optimizer(args, trainable_params)

    # prepare dataloader
    # strategies are set here because they cannot be referenced in another process. Copy them with the dataset
    # some strategies can be None
    train_dataset_group.set_current_strategies()

    # DataLoaderのプロセス数：0 は persistent_workers が使えないので注意
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

    # lr schedulerを用意する
    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
    if args.full_fp16:
        assert (
            args.mixed_precision == "fp16"
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        accelerator.print("enable full fp16 training.")
        control_net.to(weight_dtype)
    elif args.full_bf16:
        assert (
            args.mixed_precision == "bf16"
        ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        accelerator.print("enable full bf16 training.")
        control_net.to(weight_dtype)

    # acceleratorがなんかよろしくやってくれるらしい
    control_net, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        control_net, optimizer, train_dataloader, lr_scheduler
    )

    if args.fused_backward_pass:
        # use fused optimizer for backward pass: other optimizers will be supported in the future
        import library.adafactor_fused

        library.adafactor_fused.patch_adafactor_fused(optimizer)
        for param_group in optimizer.param_groups:
            for parameter in param_group["params"]:
                if parameter.requires_grad:

                    def __grad_hook(tensor: torch.Tensor, param_group=param_group):
                        if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                            accelerator.clip_grad_norm_(tensor, args.max_grad_norm)
                        optimizer.step_param(tensor, param_group)
                        tensor.grad = None

                    parameter.register_post_accumulate_grad_hook(__grad_hook)

    unet.requires_grad_(False)
    text_encoder1.requires_grad_(False)
    text_encoder2.requires_grad_(False)
    unet.to(accelerator.device, dtype=weight_dtype)

    unet.eval()
    control_net.train()

    # TextEncoderの出力をキャッシュするときにはCPUへ移動する
    if args.cache_text_encoder_outputs:
        # move Text Encoders for sampling images. Text Encoder doesn't work on CPU with fp16
        text_encoder1.to("cpu", dtype=torch.float32)
        text_encoder2.to("cpu", dtype=torch.float32)
        clean_memory_on_device(accelerator.device)
    else:
        # make sure Text Encoders are on GPU
        text_encoder1.to(accelerator.device)
        text_encoder2.to(accelerator.device)

    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=vae_dtype)

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
    # TODO: find a way to handle total batch size when there are multiple datasets
    accelerator.print("running training / 学習開始")
    accelerator.print(f"  num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset_group.num_train_images}")
    accelerator.print(f"  num reg images / 正則化画像の数: {train_dataset_group.num_reg_images}")
    accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
    accelerator.print(
        f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
    )
    # logger.info(f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}")
    accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
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
            ("sdxl_control_net_train" if args.log_tracker_name is None else args.log_tracker_name),
            config=train_util.get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )

    loss_recorder = train_util.LossRecorder()
    del train_dataset_group

    # function for saving/removing
    def save_model(ckpt_name, model, force_sync_upload=False):
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)

        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        sai_metadata = train_util.get_sai_model_spec(None, args, True, True, False)
        sai_metadata["modelspec.architecture"] = sai_model_spec.ARCH_SD_XL_V1_BASE + "/controlnet"
        state_dict = model.state_dict()

        if save_dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(save_dtype)
                state_dict[key] = v

        if os.path.splitext(ckpt_file)[1] == ".safetensors":
            from safetensors.torch import save_file

            save_file(state_dict, ckpt_file, sai_metadata)
        else:
            torch.save(state_dict, ckpt_file)

        if args.huggingface_repo_id is not None:
            huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

    def remove_model(old_ckpt_name):
        old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
        if os.path.exists(old_ckpt_file):
            accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
            os.remove(old_ckpt_file)

    # For --sample_at_first
    sdxl_train_util.sample_images(
        accelerator,
        args,
        0,
        global_step,
        accelerator.device,
        vae,
        [tokenizer1, tokenizer2],
        [text_encoder1, text_encoder2, unwrap_model(text_encoder2)],
        unet,
        controlnet=control_net,
    )

    # training loop
    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        control_net.train()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step
            with accelerator.accumulate(control_net):
                with torch.no_grad():
                    if "latents" in batch and batch["latents"] is not None:
                        latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
                    else:
                        # latentに変換
                        latents = vae.encode(batch["images"].to(dtype=vae_dtype)).latent_dist.sample().to(dtype=weight_dtype)

                        # NaNが含まれていれば警告を表示し0に置き換える
                        if torch.any(torch.isnan(latents)):
                            accelerator.print("NaN found in latents, replacing with zeros")
                            latents = torch.nan_to_num(latents, 0, out=latents)
                    latents = latents * sdxl_model_util.VAE_SCALE_FACTOR

                text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
                if text_encoder_outputs_list is not None:
                    # Text Encoder outputs are cached
                    encoder_hidden_states1, encoder_hidden_states2, pool2 = text_encoder_outputs_list
                    encoder_hidden_states1 = encoder_hidden_states1.to(accelerator.device, dtype=weight_dtype)
                    encoder_hidden_states2 = encoder_hidden_states2.to(accelerator.device, dtype=weight_dtype)
                    pool2 = pool2.to(accelerator.device, dtype=weight_dtype)
                else:
                    input_ids1, input_ids2 = batch["input_ids_list"]
                    with torch.no_grad():
                        input_ids1 = input_ids1.to(accelerator.device)
                        input_ids2 = input_ids2.to(accelerator.device)
                        encoder_hidden_states1, encoder_hidden_states2, pool2 = text_encoding_strategy.encode_tokens(
                            tokenize_strategy, [text_encoder1, text_encoder2, unwrap_model(text_encoder2)], [input_ids1, input_ids2]
                        )
                        if args.full_fp16:
                            encoder_hidden_states1 = encoder_hidden_states1.to(weight_dtype)
                            encoder_hidden_states2 = encoder_hidden_states2.to(weight_dtype)
                            pool2 = pool2.to(weight_dtype)

                # get size embeddings
                orig_size = batch["original_sizes_hw"]
                crop_size = batch["crop_top_lefts"]
                target_size = batch["target_sizes_hw"]
                embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

                # concat embeddings
                vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
                text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

                # Sample noise, sample a random timestep for each image, and add noise to the latents,
                # with noise offset and/or multires noise if specified
                noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents)

                controlnet_image = batch["conditioning_images"].to(dtype=weight_dtype)

                # '-1 to +1' to '0 to 1'
                controlnet_image = (controlnet_image + 1) / 2

                with accelerator.autocast():
                    input_resi_add, mid_add = control_net(
                        noisy_latents, timesteps, text_embedding, vector_embedding, controlnet_image
                    )
                    noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding, input_resi_add, mid_add)

                if args.v_parameterization:
                    # v-parameterization training
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)
                loss = train_util.conditional_loss(noise_pred.float(), target.float(), args.loss_type, "none", huber_c)
                loss = loss.mean([1, 2, 3])

                loss_weights = batch["loss_weights"]  # 各sampleごとのweight
                loss = loss * loss_weights

                if args.min_snr_gamma:
                    loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma, args.v_parameterization)
                if args.scale_v_pred_loss_like_noise_pred:
                    loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                if args.v_pred_like_loss:
                    loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)
                if args.debiased_estimation_loss:
                    loss = apply_debiased_estimation(loss, timesteps, noise_scheduler)

                loss = loss.mean()  # 平均なのでbatch_sizeで割る必要なし

                accelerator.backward(loss)
                if not args.fused_backward_pass:
                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        params_to_clip = control_net.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    # optimizer.step() and optimizer.zero_grad() are called in the optimizer hook
                    lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                sdxl_train_util.sample_images(
                    accelerator,
                    args,
                    None,
                    global_step,
                    accelerator.device,
                    vae,
                    [tokenizer1, tokenizer2],
                    [text_encoder1, text_encoder2, unwrap_model(text_encoder2)],
                    unet,
                    controlnet=control_net,
                )

                # 指定ステップごとにモデルを保存
                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, global_step)
                        save_model(ckpt_name, unwrap_model(control_net))

                        if args.save_state:
                            train_util.save_and_remove_state_stepwise(args, accelerator, global_step)

                        remove_step_no = train_util.get_remove_step_no(args, global_step)
                        if remove_step_no is not None:
                            remove_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no)
                            remove_model(remove_ckpt_name)

            current_loss = loss.detach().item()
            loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
            avr_loss: float = loss_recorder.moving_average
            logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if len(accelerator.trackers) > 0:
                logs = generate_step_logs(args, current_loss, avr_loss, lr_scheduler)
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if len(accelerator.trackers) > 0:
            logs = {"loss/epoch": loss_recorder.moving_average}
            accelerator.log(logs, step=epoch + 1)

        accelerator.wait_for_everyone()

        # 指定エポックごとにモデルを保存
        if args.save_every_n_epochs is not None:
            saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
            if is_main_process and saving:
                ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
                save_model(ckpt_name, unwrap_model(control_net))

                remove_epoch_no = train_util.get_remove_epoch_no(args, epoch + 1)
                if remove_epoch_no is not None:
                    remove_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no)
                    remove_model(remove_ckpt_name)

                if args.save_state:
                    train_util.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)

        sdxl_train_util.sample_images(
            accelerator,
            args,
            epoch + 1,
            global_step,
            accelerator.device,
            vae,
            [tokenizer1, tokenizer2],
            [text_encoder1, text_encoder2, unwrap_model(text_encoder2)],
            unet,
            controlnet=control_net,
        )

        # end of epoch

    if is_main_process:
        control_net = unwrap_model(control_net)

    accelerator.end_training()

    if is_main_process and (args.save_state or args.save_state_on_train_end):
        train_util.save_state_on_train_end(args, accelerator)

    if is_main_process:
        ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as)
        save_model(ckpt_name, control_net, force_sync_upload=True)

        logger.info("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    sai_model_spec.add_model_spec_arguments(parser)
    train_util.add_dataset_arguments(parser, False, True, True)
    train_util.add_training_arguments(parser, False)
    # train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    # train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)
    sdxl_train_util.add_sdxl_training_arguments(parser)

    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="controlnet model name or path / controlnetのモデル名またはパス",
    )
    parser.add_argument(
        "--conditioning_data_dir",
        type=str,
        default=None,
        help="conditioning data directory / 条件付けデータのディレクトリ",
    )
    parser.add_argument(
        "--save_model_as",
        type=str,
        default="safetensors",
        choices=[None, "ckpt", "pt", "safetensors"],
        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）",
    )
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    parser.add_argument(
        "--control_net_lr",
        type=float,
        default=1e-4,
        help="learning rate for controlnet modules / controlnetモジュールの学習率",
    )
    return parser


if __name__ == "__main__":
    # sdxl_original_unet.USE_REENTRANT = False

    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    train(args)
