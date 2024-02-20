# training with captions

import argparse
import math
import os
from multiprocessing import Value
from typing import List
import toml

from tqdm import tqdm

import torch
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from accelerate.utils import set_seed
from diffusers import DDPMScheduler

import library.train_util as train_util
from library.sdxl_train_util import add_sdxl_training_arguments
import library.stable_cascade_utils as sc_utils
import library.stable_cascade as sc

from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)

import library.config_util as config_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)


def train(args):
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    setup_logging(args, reset=True)

    # assert (
    #     not args.weighted_captions
    # ), "weighted_captions is not supported currently / weighted_captionsは現在サポートされていません"

    # TODO add assertions for other unsupported options

    cache_latents = args.cache_latents
    use_dreambooth_method = args.in_json is None

    if args.seed is not None:
        set_seed(args.seed)  # 乱数系列を初期化する

    tokenizer = sc_utils.load_tokenizer(args)

    # データセットを準備する
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, False, True))
        if args.dataset_config is not None:
            logger.info(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning(
                    "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                        ", ".join(ignored)
                    )
                )
        else:
            if use_dreambooth_method:
                logger.info("Using DreamBooth method.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                args.train_data_dir, args.reg_data_dir
                            )
                        }
                    ]
                }
            else:
                logger.info("Training with captions.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": [
                                {
                                    "image_dir": args.train_data_dir,
                                    "metadata_file": args.in_json,
                                }
                            ]
                        }
                    ]
                }

        blueprint = blueprint_generator.generate(user_config, args, tokenizer=[tokenizer])
        train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args, [tokenizer])

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(32)

    if args.debug_dataset:
        train_util.debug_dataset(train_dataset_group, True)
        return
    if len(train_dataset_group) == 0:
        logger.error(
            "No data found. Please verify the metadata file and train_data_dir option. / 画像がありません。メタデータおよびtrain_data_dirオプションを確認してください。"
        )
        return

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

    if args.cache_text_encoder_outputs:
        assert (
            train_dataset_group.is_text_encoder_output_cacheable()
        ), "when caching text encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / text encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

    # acceleratorを準備する
    logger.info("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    effnet_dtype = torch.float32 if args.no_half_vae else weight_dtype

    # モデルを読み込む
    loading_device = accelerator.device if args.lowram else "cpu"
    effnet = sc_utils.load_effnet(args.effnet_checkpoint_path, loading_device)
    stage_c = sc_utils.load_stage_c_model(args.stage_c_checkpoint_path, dtype=weight_dtype, device=loading_device)
    text_encoder1 = sc_utils.load_clip_text_model(args.text_model_checkpoint_path, dtype=weight_dtype, device=loading_device)

    if args.sample_at_first or args.sample_every_n_steps is not None or args.sample_every_n_epochs is not None:
        # Previewer is small enough to be loaded on CPU
        previewer = sc_utils.load_previewer_model(args.previewer_checkpoint_path, dtype=torch.float32, device="cpu")
        previewer.eval()
    else:
        previewer = None

    # 学習を準備する
    if cache_latents:
        effnet.to(accelerator.device, dtype=effnet_dtype)
        effnet.requires_grad_(False)
        effnet.eval()
        with torch.no_grad():
            train_dataset_group.cache_latents(
                effnet,
                args.vae_batch_size,
                args.cache_latents_to_disk,
                accelerator.is_main_process,
                train_util.STABLE_CASCADE_LATENTS_CACHE_SUFFIX,
                32,
            )
        effnet.to("cpu")
        clean_memory_on_device(accelerator.device)

        accelerator.wait_for_everyone()

    # 学習を準備する：モデルを適切な状態にする
    if args.gradient_checkpointing:
        accelerator.print("enable gradient checkpointing")
        stage_c.set_gradient_checkpointing(True)

    train_stage_c = args.learning_rate > 0
    train_text_encoder1 = False

    if args.train_text_encoder:
        accelerator.print("enable text encoder training")
        if args.gradient_checkpointing:
            text_encoder1.gradient_checkpointing_enable()
        lr_te1 = args.learning_rate_te1 if args.learning_rate_te1 is not None else args.learning_rate  # 0 means not train
        train_text_encoder1 = lr_te1 > 0
        assert (
            train_text_encoder1
        ), "text_encoder1 learning rate is 0. Please set a positive value / text_encoder1の学習率が0です。正の値を設定してください。"

        if not train_text_encoder1:
            text_encoder1.to(weight_dtype)
        text_encoder1.requires_grad_(train_text_encoder1)
        text_encoder1.train(train_text_encoder1)
    else:
        text_encoder1.to(weight_dtype)
        text_encoder1.requires_grad_(False)
        text_encoder1.eval()

    # TextEncoderの出力をキャッシュする
    if args.cache_text_encoder_outputs:
        # Text Encodes are eval and no grad
        with torch.no_grad(), accelerator.autocast():
            train_dataset_group.cache_text_encoder_outputs(
                (tokenizer,),
                (text_encoder1,),
                accelerator.device,
                None,
                args.cache_text_encoder_outputs_to_disk,
                accelerator.is_main_process,
                sc_utils.TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX,
            )
        accelerator.wait_for_everyone()

    if not cache_latents:
        effnet.requires_grad_(False)
        effnet.eval()
        effnet.to(accelerator.device, dtype=effnet_dtype)

    stage_c.requires_grad_(True)
    if not train_stage_c:
        stage_c.to(accelerator.device, dtype=weight_dtype)  # because of stage_c will not be prepared

    training_models = []
    params_to_optimize = []
    if train_stage_c:
        training_models.append(stage_c)
        params_to_optimize.append({"params": list(stage_c.parameters()), "lr": args.learning_rate})

    if train_text_encoder1:
        training_models.append(text_encoder1)
        params_to_optimize.append({"params": list(text_encoder1.parameters()), "lr": args.learning_rate_te1 or args.learning_rate})

    # calculate number of trainable parameters
    n_params = 0
    for params in params_to_optimize:
        for p in params["params"]:
            n_params += p.numel()

    accelerator.print(f"train stage-C: {train_stage_c}, text_encoder1: {train_text_encoder1}")
    accelerator.print(f"number of models: {len(training_models)}")
    accelerator.print(f"number of trainable parameters: {n_params}")

    # 学習に必要なクラスを準備する
    accelerator.print("prepare optimizer, data loader etc.")
    _, _, optimizer = train_util.get_optimizer(args, trainable_params=params_to_optimize)

    # dataloaderを準備する
    # DataLoaderのプロセス数：0はメインプロセスになる
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count() - 1)  # cpu_count-1 ただし最大で指定された数まで
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
        stage_c.to(weight_dtype)
        text_encoder1.to(weight_dtype)
    elif args.full_bf16:
        assert (
            args.mixed_precision == "bf16"
        ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        accelerator.print("enable full bf16 training.")
        stage_c.to(weight_dtype)
        text_encoder1.to(weight_dtype)

    # acceleratorがなんかよろしくやってくれるらしい
    if train_stage_c:
        stage_c = accelerator.prepare(stage_c)
    if train_text_encoder1:
        text_encoder1 = accelerator.prepare(text_encoder1)

    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

    # TextEncoderの出力をキャッシュするときにはCPUへ移動する
    if args.cache_text_encoder_outputs:
        # move Text Encoders for sampling images. Text Encoder doesn't work on CPU with fp16
        text_encoder1.to("cpu", dtype=torch.float32)
        clean_memory_on_device(accelerator.device)
    else:
        # make sure Text Encoders are on GPU
        text_encoder1.to(accelerator.device)

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
    # total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    accelerator.print("running training / 学習開始")
    accelerator.print(f"  num examples / サンプル数: {train_dataset_group.num_train_images}")
    accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
    accelerator.print(
        f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
    )
    # accelerator.print(
    #     f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}"
    # )
    accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0

    # 謎のクラス GDF
    gdf = sc.GDF(
        schedule=sc.CosineSchedule(clamp_range=[0.0001, 0.9999]),
        input_scaler=sc.VPScaler(),
        target=sc.EpsilonTarget(),
        noise_cond=sc.CosineTNoiseCond(),
        loss_weight=sc.AdaptiveLossWeight() if args.adaptive_loss_weight else sc.P2LossWeight(),
    )

    # 以下2つの変数は、どうもデフォルトのままっぽい
    # gdf.loss_weight.bucket_ranges = torch.tensor(self.info.adaptive_loss['bucket_ranges'])
    # gdf.loss_weight.bucket_losses = torch.tensor(self.info.adaptive_loss['bucket_losses'])

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers("finetuning" if args.log_tracker_name is None else args.log_tracker_name, init_kwargs=init_kwargs)

    # For --sample_at_first
    sc_utils.sample_images(accelerator, args, 0, global_step, previewer, tokenizer, text_encoder1, stage_c, gdf)

    loss_recorder = train_util.LossRecorder()
    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        for m in training_models:
            m.train()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step
            with accelerator.accumulate(*training_models):
                if "latents" in batch and batch["latents"] is not None:
                    latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
                else:
                    with torch.no_grad():
                        # latentに変換
                        # XXX Effnet preprocessing is included in encode method
                        latents = effnet.encode(batch["images"].to(effnet_dtype)).latent_dist.sample().to(weight_dtype)

                        # NaNが含まれていれば警告を表示し0に置き換える
                        if torch.any(torch.isnan(latents)):
                            accelerator.print("NaN found in latents, replacing with zeros")
                            latents = torch.nan_to_num(latents, 0, out=latents)

                # # debug: decode latent with previewer and save it
                # import time
                # import numpy as np
                # from PIL import Image
                # ts = time.time()
                # images = previewer(latents.to(previewer.device, dtype=previewer.dtype))
                # for i, img in enumerate(images):
                #     img = img.detach().cpu().numpy().transpose(1, 2, 0)
                #     img = np.clip(img, 0, 1)
                #     img = (img * 255).astype(np.uint8)
                #     img = Image.fromarray(img)
                #     img.save(f"logs/previewer_{i}_{ts}.png")

                if "text_encoder_outputs1_list" not in batch or batch["text_encoder_outputs1_list"] is None:
                    input_ids1 = batch["input_ids"]
                    with torch.set_grad_enabled(args.train_text_encoder):
                        # Get the text embedding for conditioning
                        # TODO support weighted captions
                        input_ids1 = input_ids1.to(accelerator.device)
                        # unwrap_model is fine for models not wrapped by accelerator
                        encoder_hidden_states, pool = train_util.get_hidden_states_stable_cascade(
                            args.max_token_length,
                            input_ids1,
                            tokenizer,
                            text_encoder1,
                            None if not args.full_fp16 else weight_dtype,
                            accelerator,
                        )
                else:
                    encoder_hidden_states = batch["text_encoder_outputs1_list"].to(accelerator.device).to(weight_dtype)
                    pool = batch["text_encoder_pool2_list"].to(accelerator.device).to(weight_dtype)

                pool = pool.unsqueeze(1)  # add extra dimension b,1280 -> b,1,1280

                # FORWARD PASS
                with torch.no_grad():
                    noised, noise, target, logSNR, noise_cond, loss_weight = gdf.diffuse(latents, shift=1, loss_shift=1)

                zero_img_emb = torch.zeros(noised.shape[0], 768, device=accelerator.device)
                with accelerator.autocast():
                    pred = stage_c(
                        noised, noise_cond, clip_text=encoder_hidden_states, clip_text_pooled=pool, clip_img=zero_img_emb
                    )
                    loss = torch.nn.functional.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
                    loss_adjusted = (loss * loss_weight).mean()

                if args.adaptive_loss_weight:
                    gdf.loss_weight.update_buckets(logSNR, loss)  # use loss instead of loss_adjusted

                accelerator.backward(loss_adjusted)
                if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                    params_to_clip = []
                    for m in training_models:
                        params_to_clip.extend(m.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                sc_utils.sample_images(accelerator, args, None, global_step, previewer, tokenizer, text_encoder1, stage_c, gdf)

                # 指定ステップごとにモデルを保存
                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        sc_utils.save_stage_c_model_on_epoch_end_or_stepwise(
                            args,
                            False,
                            accelerator,
                            save_dtype,
                            epoch,
                            num_train_epochs,
                            global_step,
                            accelerator.unwrap_model(stage_c),
                            accelerator.unwrap_model(text_encoder1) if train_text_encoder1 else None,
                        )

            current_loss = loss_adjusted.detach().item()  # 平均なのでbatch sizeは関係ないはず
            if args.logging_dir is not None:
                logs = {"loss": current_loss}
                train_util.append_lr_to_logs(logs, lr_scheduler, args.optimizer_type, including_unet=True)

                accelerator.log(logs, step=global_step)

            loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
            avr_loss: float = loss_recorder.moving_average
            logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if args.logging_dir is not None:
            logs = {"loss/epoch": loss_recorder.moving_average}
            accelerator.log(logs, step=epoch + 1)

        accelerator.wait_for_everyone()

        if args.save_every_n_epochs is not None:
            if accelerator.is_main_process:
                sc_utils.save_stage_c_model_on_epoch_end_or_stepwise(
                    args,
                    True,
                    accelerator,
                    save_dtype,
                    epoch,
                    num_train_epochs,
                    global_step,
                    accelerator.unwrap_model(stage_c),
                    accelerator.unwrap_model(text_encoder1) if train_text_encoder1 else None,
                )

        sc_utils.sample_images(accelerator, args, epoch + 1, global_step, previewer, tokenizer, text_encoder1, stage_c, gdf)

    is_main_process = accelerator.is_main_process
    # if is_main_process:
    stage_c = accelerator.unwrap_model(stage_c)
    text_encoder1 = accelerator.unwrap_model(text_encoder1)

    accelerator.end_training()

    if args.save_state:  # and is_main_process:
        train_util.save_state_on_train_end(args, accelerator)

    del accelerator  # この後メモリを使うのでこれは消す

    if is_main_process:
        sc_utils.save_stage_c_model_on_end(
            args, save_dtype, epoch, global_step, stage_c, text_encoder1 if train_text_encoder1 else None
        )
        logger.info("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    sc_utils.add_effnet_arguments(parser)
    sc_utils.add_stage_c_arguments(parser)
    sc_utils.add_text_model_arguments(parser)
    sc_utils.add_previewer_arguments(parser)
    sc_utils.add_training_arguments(parser)
    train_util.add_tokenizer_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, False)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    add_sdxl_training_arguments(parser)  # cache text encoder outputs

    parser.add_argument("--train_text_encoder", action="store_true", help="train text encoder / text encoderも学習する")
    parser.add_argument(
        "--learning_rate_te1",
        type=float,
        default=None,
        help="learning rate for text encoder / text encoderの学習率",
    )
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 Effnet in mixed precision (use float Effnet) / mixed precisionでも fp16/bf16 Effnetを使わずfloat Effnetを使う",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)

    train(args)
