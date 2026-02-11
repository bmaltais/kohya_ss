# training with captions

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import math
import os
from multiprocessing import Value
from typing import List
import toml

from tqdm import tqdm

import torch
from library import utils
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from library import deepspeed_utils, sd3_models, sd3_train_utils, sd3_utils, strategy_base, strategy_sd3
from library.sdxl_train_util import match_mixed_precision

# , sdxl_model_util

import library.train_util as train_util

from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)

import library.config_util as config_util

# import library.sdxl_train_util as sdxl_train_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
from library.custom_train_functions import apply_masked_loss, add_custom_train_arguments

# from library.custom_train_functions import (
#     apply_snr_weight,
#     prepare_scheduler_for_custom_training,
#     scale_v_prediction_loss_like_noise_prediction,
#     add_v_prediction_like_loss,
#     apply_debiased_estimation,
#     apply_masked_loss,
# )


def train(args):
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    # sdxl_train_util.verify_sdxl_training_args(args)
    deepspeed_utils.prepare_deepspeed_args(args)
    setup_logging(args, reset=True)

    # temporary: backward compatibility for deprecated options. remove in the future
    if not args.skip_cache_check:
        args.skip_cache_check = args.skip_latents_validity_check

    # assert (
    #     not args.weighted_captions
    # ), "weighted_captions is not supported currently / weighted_captionsは現在サポートされていません"
    # assert (
    #     not args.train_text_encoder or not args.cache_text_encoder_outputs
    # ), "cache_text_encoder_outputs is not supported when training text encoder / text encoderを学習するときはcache_text_encoder_outputsはサポートされていません"
    if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
        logger.warning(
            "cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled / cache_text_encoder_outputs_to_diskが有効になっているため、cache_text_encoder_outputsも有効になります"
        )
        args.cache_text_encoder_outputs = True

    assert not args.train_text_encoder or (args.use_t5xxl_cache_only or not args.cache_text_encoder_outputs), (
        "when training text encoder, text encoder outputs must not be cached (except for T5XXL)"
        + " / text encoderの学習時はtext encoderの出力はキャッシュできません（t5xxlのみキャッシュすることは可能です）"
    )

    if args.use_t5xxl_cache_only and not args.cache_text_encoder_outputs:
        logger.warning(
            "use_t5xxl_cache_only is enabled, so cache_text_encoder_outputs is automatically enabled."
            + " / use_t5xxl_cache_onlyが有効なため、cache_text_encoder_outputsも自動的に有効になります"
        )
        args.cache_text_encoder_outputs = True

    if args.train_t5xxl:
        assert (
            args.train_text_encoder
        ), "when training T5XXL, text encoder (CLIP-L/G) must be trained / T5XXLを学習するときはtext encoder (CLIP-L/G)も学習する必要があります"
        assert (
            not args.cache_text_encoder_outputs
        ), "when training T5XXL, t5xxl output must not be cached / T5XXLを学習するときはt5xxlの出力をキャッシュできません"

    cache_latents = args.cache_latents
    use_dreambooth_method = args.in_json is None

    if args.seed is not None:
        set_seed(args.seed)  # 乱数系列を初期化する

    # prepare caching strategy: this must be set before preparing dataset. because dataset may use this strategy for initialization.
    if args.cache_latents:
        latents_caching_strategy = strategy_sd3.Sd3LatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    # データセットを準備する
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
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

        blueprint = blueprint_generator.generate(user_config, args)
        train_dataset_group, val_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args)
        val_dataset_group = None

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(8)  # TODO これでいいか確認

    if args.debug_dataset:
        if args.cache_text_encoder_outputs:
            strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(
                strategy_sd3.Sd3TextEncoderOutputsCachingStrategy(
                    args.cache_text_encoder_outputs_to_disk,
                    args.text_encoder_batch_size,
                    False,
                    False,
                    False,
                    False,
                )
            )
        train_dataset_group.set_current_strategies()
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

    # モデルを読み込む

    # t5xxl_dtype = weight_dtype
    model_dtype = match_mixed_precision(args, weight_dtype)  # None (default) or fp16/bf16 (full_xxxx)
    if args.clip_l is None:
        sd3_state_dict = utils.load_safetensors(
            args.pretrained_model_name_or_path, "cpu", args.disable_mmap_load_safetensors, model_dtype
        )
    else:
        sd3_state_dict = None

    # load tokenizer and prepare tokenize strategy
    sd3_tokenize_strategy = strategy_sd3.Sd3TokenizeStrategy(args.t5xxl_max_token_length)
    strategy_base.TokenizeStrategy.set_strategy(sd3_tokenize_strategy)

    # load clip_l, clip_g, t5xxl for caching text encoder outputs
    # clip_l = sd3_train_utils.load_target_model("clip_l", args, sd3_state_dict, accelerator, attn_mode, clip_dtype, device_to_load)
    # clip_g = sd3_train_utils.load_target_model("clip_g", args, sd3_state_dict, accelerator, attn_mode, clip_dtype, device_to_load)
    clip_l = sd3_utils.load_clip_l(args.clip_l, weight_dtype, "cpu", args.disable_mmap_load_safetensors, state_dict=sd3_state_dict)
    clip_g = sd3_utils.load_clip_g(args.clip_g, weight_dtype, "cpu", args.disable_mmap_load_safetensors, state_dict=sd3_state_dict)
    t5xxl = sd3_utils.load_t5xxl(args.t5xxl, weight_dtype, "cpu", args.disable_mmap_load_safetensors, state_dict=sd3_state_dict)
    assert clip_l is not None and clip_g is not None and t5xxl is not None, "clip_l, clip_g, t5xxl must be specified"

    # prepare text encoding strategy
    text_encoding_strategy = strategy_sd3.Sd3TextEncodingStrategy(
        args.apply_lg_attn_mask, args.apply_t5_attn_mask, args.clip_l_dropout_rate, args.clip_g_dropout_rate, args.t5_dropout_rate
    )
    strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

    # 学習を準備する：モデルを適切な状態にする
    train_clip = False
    train_t5xxl = False

    if args.train_text_encoder:
        accelerator.print("enable text encoder training")
        if args.gradient_checkpointing:
            clip_l.gradient_checkpointing_enable()
            clip_g.gradient_checkpointing_enable()
            if args.train_t5xxl:
                t5xxl.gradient_checkpointing_enable()

        lr_te1 = args.learning_rate_te1 if args.learning_rate_te1 is not None else args.learning_rate  # 0 means not train
        lr_te2 = args.learning_rate_te2 if args.learning_rate_te2 is not None else args.learning_rate  # 0 means not train
        lr_t5xxl = args.learning_rate_te3 if args.learning_rate_te3 is not None else args.learning_rate  # 0 means not train
        train_clip = lr_te1 != 0 or lr_te2 != 0
        train_t5xxl = lr_t5xxl != 0 and args.train_t5xxl

        clip_l.to(weight_dtype)
        clip_g.to(weight_dtype)
        t5xxl.to(weight_dtype)
        clip_l.requires_grad_(train_clip)
        clip_g.requires_grad_(train_clip)
        t5xxl.requires_grad_(train_t5xxl)
    else:
        print("disable text encoder training")
        clip_l.to(weight_dtype)
        clip_g.to(weight_dtype)
        t5xxl.to(weight_dtype)
        clip_l.requires_grad_(False)
        clip_g.requires_grad_(False)
        t5xxl.requires_grad_(False)
        lr_te1 = 0
        lr_te2 = 0
        lr_t5xxl = 0

    # cache text encoder outputs
    sample_prompts_te_outputs = None
    if args.cache_text_encoder_outputs:
        clip_l.to(accelerator.device)
        clip_g.to(accelerator.device)
        t5xxl.to(accelerator.device)
        clip_l.eval()
        clip_g.eval()
        t5xxl.eval()

        text_encoder_caching_strategy = strategy_sd3.Sd3TextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk,
            args.text_encoder_batch_size,
            args.skip_cache_check,
            train_clip or args.use_t5xxl_cache_only,  # if clip is trained or t5xxl is cached, caching is partial
            args.apply_lg_attn_mask,
            args.apply_t5_attn_mask,
        )
        strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_caching_strategy)

        with accelerator.autocast():
            train_dataset_group.new_cache_text_encoder_outputs([clip_l, clip_g, t5xxl], accelerator)

        # cache sample prompt's embeddings to free text encoder's memory
        if args.sample_prompts is not None:
            logger.info(f"cache Text Encoder outputs for sample prompt: {args.sample_prompts}")
            prompts = train_util.load_prompts(args.sample_prompts)
            sample_prompts_te_outputs = {}  # key: prompt, value: text encoder outputs
            with accelerator.autocast(), torch.no_grad():
                for prompt_dict in prompts:
                    for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                        if p not in sample_prompts_te_outputs:
                            logger.info(f"cache Text Encoder outputs for prompt: {p}")
                            tokens_and_masks = sd3_tokenize_strategy.tokenize(p)
                            sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                                sd3_tokenize_strategy,
                                [clip_l, clip_g, t5xxl],
                                tokens_and_masks,
                                args.apply_lg_attn_mask,
                                args.apply_t5_attn_mask,
                                enable_dropout=False,
                            )

        accelerator.wait_for_everyone()

        # now we can delete Text Encoders to free memory
        if not args.use_t5xxl_cache_only:
            clip_l = None
            clip_g = None
        t5xxl = None

        clean_memory_on_device(accelerator.device)

    # load VAE for caching latents
    if sd3_state_dict is None:
        logger.info(f"load state dict for MMDiT and VAE from {args.pretrained_model_name_or_path}")
        sd3_state_dict = utils.load_safetensors(
            args.pretrained_model_name_or_path, "cpu", args.disable_mmap_load_safetensors, model_dtype
        )

    vae = sd3_utils.load_vae(args.vae, weight_dtype, "cpu", args.disable_mmap_load_safetensors, state_dict=sd3_state_dict)
    if cache_latents:
        # vae = sd3_train_utils.load_target_model("vae", args, sd3_state_dict, accelerator, attn_mode, vae_dtype, device_to_load)
        vae.to(accelerator.device, dtype=weight_dtype)
        vae.requires_grad_(False)
        vae.eval()

        train_dataset_group.new_cache_latents(vae, accelerator)

        vae.to("cpu")  # if no sampling, vae can be deleted
        clean_memory_on_device(accelerator.device)

        accelerator.wait_for_everyone()

    # load MMDIT
    mmdit = sd3_utils.load_mmdit(sd3_state_dict, model_dtype, "cpu")

    # attn_mode = "xformers" if args.xformers else "torch"
    # assert (
    #     attn_mode == "torch"
    # ), f"attn_mode {attn_mode} is not supported yet. Please use `--sdpa` instead of `--xformers`. / attn_mode {attn_mode} はサポートされていません。`--xformers`の代わりに`--sdpa`を使ってください。"

    mmdit.set_pos_emb_random_crop_rate(args.pos_emb_random_crop_rate)

    # set resolutions for positional embeddings
    if args.enable_scaled_pos_embed:
        resolutions = train_dataset_group.get_resolutions()
        latent_sizes = [round(math.sqrt(res[0] * res[1])) // 8 for res in resolutions]  # 8 is stride for latent
        latent_sizes = list(set(latent_sizes))  # remove duplicates
        logger.info(f"Prepare scaled positional embeddings for resolutions: {resolutions}, sizes: {latent_sizes}")
        mmdit.enable_scaled_pos_embed(True, latent_sizes)

    if args.gradient_checkpointing:
        mmdit.enable_gradient_checkpointing()

    train_mmdit = args.learning_rate != 0
    mmdit.requires_grad_(train_mmdit)
    if not train_mmdit:
        mmdit.to(accelerator.device, dtype=weight_dtype)  # because of mmdit will not be prepared

    # block swap
    is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
    if is_swapping_blocks:
        # Swap blocks between CPU and GPU to reduce memory usage, in forward and backward passes.
        # This idea is based on 2kpr's great work. Thank you!
        logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
        mmdit.enable_block_swap(args.blocks_to_swap, accelerator.device)

    if not cache_latents:
        # move to accelerator device
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=weight_dtype)

    mmdit.requires_grad_(train_mmdit)
    if not train_mmdit:
        mmdit.to(accelerator.device, dtype=weight_dtype)  # because of unet is not prepared

    if args.num_last_block_to_freeze:
        # freeze last n blocks of MM-DIT
        block_name = "x_block"
        filtered_blocks = [(name, param) for name, param in mmdit.named_parameters() if block_name in name]
        accelerator.print(f"filtered_blocks: {len(filtered_blocks)}")

        num_blocks_to_freeze = min(len(filtered_blocks), args.num_last_block_to_freeze)

        accelerator.print(f"freeze_blocks: {num_blocks_to_freeze}")

        start_freezing_from = max(0, len(filtered_blocks) - num_blocks_to_freeze)

        for i in range(start_freezing_from, len(filtered_blocks)):
            _, param = filtered_blocks[i]
            param.requires_grad = False

    training_models = []
    params_to_optimize = []
    param_names = []
    training_models.append(mmdit)
    params_to_optimize.append({"params": list(filter(lambda p: p.requires_grad, mmdit.parameters())), "lr": args.learning_rate})
    param_names.append([n for n, _ in mmdit.named_parameters()])

    if train_clip:
        if lr_te1 > 0:
            training_models.append(clip_l)
            params_to_optimize.append({"params": list(clip_l.parameters()), "lr": args.learning_rate_te1 or args.learning_rate})
            param_names.append([n for n, _ in clip_l.named_parameters()])
        if lr_te2 > 0:
            training_models.append(clip_g)
            params_to_optimize.append({"params": list(clip_g.parameters()), "lr": args.learning_rate_te2 or args.learning_rate})
            param_names.append([n for n, _ in clip_g.named_parameters()])
    if train_t5xxl:
        training_models.append(t5xxl)
        params_to_optimize.append({"params": list(t5xxl.parameters()), "lr": args.learning_rate_te3 or args.learning_rate})
        param_names.append([n for n, _ in t5xxl.named_parameters()])

    # calculate number of trainable parameters
    n_params = 0
    for group in params_to_optimize:
        for p in group["params"]:
            n_params += p.numel()

    accelerator.print(f"train mmdit: {train_mmdit} , clip:{train_clip}, t5xxl:{train_t5xxl}")
    accelerator.print(f"number of models: {len(training_models)}")
    accelerator.print(f"number of trainable parameters: {n_params}")

    # 学習に必要なクラスを準備する
    accelerator.print("prepare optimizer, data loader etc.")

    if args.blockwise_fused_optimizers:
        # fused backward pass: https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html
        # Instead of creating an optimizer for all parameters as in the tutorial, we create an optimizer for each block of parameters.
        # This balances memory usage and management complexity.

        # split params into groups for mmdit. clip_l, clip_g, t5xxl are in each group
        grouped_params = []
        param_group = {}
        group = params_to_optimize[0]
        named_parameters = list(mmdit.named_parameters())
        assert len(named_parameters) == len(group["params"]), "number of parameters does not match"
        for p, np in zip(group["params"], named_parameters):
            # determine target layer and block index for each parameter
            block_type = "other"  # joint or other
            if np[0].startswith("joint_blocks"):
                block_idx = int(np[0].split(".")[1])
                block_type = "joint"
            else:
                block_idx = -1

            param_group_key = (block_type, block_idx)
            if param_group_key not in param_group:
                param_group[param_group_key] = []
            param_group[param_group_key].append(p)

        block_types_and_indices = []
        for param_group_key, param_group in param_group.items():
            block_types_and_indices.append(param_group_key)
            grouped_params.append({"params": param_group, "lr": args.learning_rate})

            num_params = 0
            for p in param_group:
                num_params += p.numel()
            accelerator.print(f"block {param_group_key}: {num_params} parameters")

        grouped_params.extend(params_to_optimize[1:])  # add clip_l, clip_g, t5xxl if they are trained

        # prepare optimizers for each group
        optimizers = []
        for group in grouped_params:
            _, _, optimizer = train_util.get_optimizer(args, trainable_params=[group])
            optimizers.append(optimizer)
        optimizer = optimizers[0]  # avoid error in the following code

        logger.info(f"using {len(optimizers)} optimizers for blockwise fused optimizers")

        if train_util.is_schedulefree_optimizer(optimizers[0], args):
            raise ValueError("Schedule-free optimizer is not supported with blockwise fused optimizers")
        optimizer_train_fn = lambda: None  # dummy function
        optimizer_eval_fn = lambda: None  # dummy function
    else:
        _, _, optimizer = train_util.get_optimizer(args, trainable_params=params_to_optimize)
        optimizer_train_fn, optimizer_eval_fn = train_util.get_optimizer_train_eval_fn(optimizer, args)

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
    if args.blockwise_fused_optimizers:
        # prepare lr schedulers for each optimizer
        lr_schedulers = [train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes) for optimizer in optimizers]
        lr_scheduler = lr_schedulers[0]  # avoid error in the following code
    else:
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
    if args.full_fp16:
        assert (
            args.mixed_precision == "fp16"
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        accelerator.print("enable full fp16 training.")
        mmdit.to(weight_dtype)
        if clip_l is not None:
            clip_l.to(weight_dtype)
        if clip_g is not None:
            clip_g.to(weight_dtype)
        if t5xxl is not None:
            t5xxl.to(weight_dtype)
    elif args.full_bf16:
        assert (
            args.mixed_precision == "bf16"
        ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        accelerator.print("enable full bf16 training.")
        mmdit.to(weight_dtype)
        if clip_l is not None:
            clip_l.to(weight_dtype)
        if clip_g is not None:
            clip_g.to(weight_dtype)
        if t5xxl is not None:
            t5xxl.to(weight_dtype)

    # TODO check if this is necessary. SD3 uses pool for clip_l and clip_g
    # # freeze last layer and final_layer_norm in te1 since we use the output of the penultimate layer
    # if train_clip_l:
    #     clip_l.text_model.encoder.layers[-1].requires_grad_(False)
    #     clip_l.text_model.final_layer_norm.requires_grad_(False)

    # move Text Encoders to GPU if not caching outputs
    if not args.cache_text_encoder_outputs:
        # make sure Text Encoders are on GPU
        # TODO support CPU for text encoders
        clip_l.to(accelerator.device)
        clip_g.to(accelerator.device)
        if t5xxl is not None:
            t5xxl.to(accelerator.device)

    clean_memory_on_device(accelerator.device)

    if args.deepspeed:
        ds_model = deepspeed_utils.prepare_deepspeed_model(
            args, mmdit=mmdit, clip_l=clip_l if train_clip else None, clip_g=clip_g if train_clip else None
        )
        # most of ZeRO stage uses optimizer partitioning, so we have to prepare optimizer and ds_model at the same time. # pull/1139#issuecomment-1986790007
        ds_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            ds_model, optimizer, train_dataloader, lr_scheduler
        )
        training_models = [ds_model]

    else:
        # acceleratorがなんかよろしくやってくれるらしい
        if train_mmdit:
            mmdit = accelerator.prepare(mmdit, device_placement=[not is_swapping_blocks])
            if is_swapping_blocks:
                accelerator.unwrap_model(mmdit).move_to_device_except_swap_blocks(accelerator.device)  # reduce peak memory usage
        if train_clip:
            clip_l = accelerator.prepare(clip_l)
            clip_g = accelerator.prepare(clip_g)
        if train_t5xxl:
            t5xxl = accelerator.prepare(t5xxl)
        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

    # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
    if args.full_fp16:
        # During deepseed training, accelerate not handles fp16/bf16|mixed precision directly via scaler. Let deepspeed engine do.
        # -> But we think it's ok to patch accelerator even if deepspeed is enabled.
        train_util.patch_accelerator_for_fp16_training(accelerator)

    # resumeする
    train_util.resume_from_local_or_hf_if_specified(accelerator, args)

    if args.fused_backward_pass:
        # use fused optimizer for backward pass: other optimizers will be supported in the future
        import library.adafactor_fused

        library.adafactor_fused.patch_adafactor_fused(optimizer)

        for param_group, param_name_group in zip(optimizer.param_groups, param_names):
            for parameter, param_name in zip(param_group["params"], param_name_group):
                if parameter.requires_grad:

                    def create_grad_hook(p_name, p_group):
                        def grad_hook(tensor: torch.Tensor):
                            if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                                accelerator.clip_grad_norm_(tensor, args.max_grad_norm)
                            optimizer.step_param(tensor, p_group)
                            tensor.grad = None

                        return grad_hook

                    parameter.register_post_accumulate_grad_hook(create_grad_hook(param_name, param_group))

    elif args.blockwise_fused_optimizers:
        # prepare for additional optimizers and lr schedulers
        for i in range(1, len(optimizers)):
            optimizers[i] = accelerator.prepare(optimizers[i])
            lr_schedulers[i] = accelerator.prepare(lr_schedulers[i])

        # counters are used to determine when to step the optimizer
        global optimizer_hooked_count
        global num_parameters_per_group
        global parameter_optimizer_map

        optimizer_hooked_count = {}
        num_parameters_per_group = [0] * len(optimizers)
        parameter_optimizer_map = {}

        for opt_idx, optimizer in enumerate(optimizers):
            for param_group in optimizer.param_groups:
                for parameter in param_group["params"]:
                    if parameter.requires_grad:

                        def grad_hook(parameter: torch.Tensor):
                            if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                                accelerator.clip_grad_norm_(parameter, args.max_grad_norm)

                            i = parameter_optimizer_map[parameter]
                            optimizer_hooked_count[i] += 1
                            if optimizer_hooked_count[i] == num_parameters_per_group[i]:
                                optimizers[i].step()
                                optimizers[i].zero_grad(set_to_none=True)

                        parameter.register_post_accumulate_grad_hook(grad_hook)
                        parameter_optimizer_map[parameter] = opt_idx
                        num_parameters_per_group[opt_idx] += 1

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

    # only used to get timesteps, etc. TODO manage timesteps etc. separately
    dummy_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            "finetuning" if args.log_tracker_name is None else args.log_tracker_name,
            config=train_util.get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )

    if is_swapping_blocks:
        accelerator.unwrap_model(mmdit).prepare_block_swap_before_forward()

    # For --sample_at_first
    optimizer_eval_fn()
    sd3_train_utils.sample_images(accelerator, args, 0, global_step, mmdit, vae, [clip_l, clip_g, t5xxl], sample_prompts_te_outputs)
    optimizer_train_fn()
    if len(accelerator.trackers) > 0:
        # log empty object to commit the sample images to wandb
        accelerator.log({}, step=0)

    # show model device and dtype
    logger.info(
        f"mmdit device: {accelerator.unwrap_model(mmdit).device}, dtype: {accelerator.unwrap_model(mmdit).dtype}"
        if mmdit
        else "mmdit is None"
    )
    logger.info(
        f"clip_l device: {accelerator.unwrap_model(clip_l).device}, dtype: {accelerator.unwrap_model(clip_l).dtype}"
        if clip_l
        else "clip_l is None"
    )
    logger.info(
        f"clip_g device: {accelerator.unwrap_model(clip_g).device}, dtype: {accelerator.unwrap_model(clip_g).dtype}"
        if clip_g
        else "clip_g is None"
    )
    logger.info(
        f"t5xxl device: {accelerator.unwrap_model(t5xxl).device}, dtype: {accelerator.unwrap_model(t5xxl).dtype}"
        if t5xxl
        else "t5xxl is None"
    )
    logger.info(
        f"vae device: {accelerator.unwrap_model(vae).device}, dtype: {accelerator.unwrap_model(vae).dtype}"
        if vae is not None
        else "vae is None"
    )

    loss_recorder = train_util.LossRecorder()
    epoch = 0  # avoid error when max_train_steps is 0
    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        for m in training_models:
            m.train()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step

            if args.blockwise_fused_optimizers:
                optimizer_hooked_count = {i: 0 for i in range(len(optimizers))}  # reset counter for each step

            with accelerator.accumulate(*training_models):
                if "latents" in batch and batch["latents"] is not None:
                    latents = batch["latents"].to(accelerator.device, dtype=weight_dtype)
                else:
                    with torch.no_grad():
                        # encode images to latents. images are [-1, 1]
                        latents = vae.encode(batch["images"].to(vae.device, dtype=vae.dtype)).to(
                            accelerator.device, dtype=weight_dtype
                        )

                    # NaNが含まれていれば警告を表示し0に置き換える
                    if torch.any(torch.isnan(latents)):
                        accelerator.print("NaN found in latents, replacing with zeros")
                        latents = torch.nan_to_num(latents, 0, out=latents)

                # latents = latents * sdxl_model_util.VAE_SCALE_FACTOR
                latents = sd3_models.SDVAE.process_in(latents)

                text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
                if text_encoder_outputs_list is not None:
                    text_encoder_outputs_list = text_encoding_strategy.drop_cached_text_encoder_outputs(*text_encoder_outputs_list)
                    lg_out, t5_out, lg_pooled, l_attn_mask, g_attn_mask, t5_attn_mask = text_encoder_outputs_list
                    if args.use_t5xxl_cache_only:
                        lg_out = None
                        lg_pooled = None
                else:
                    lg_out = None
                    t5_out = None
                    lg_pooled = None
                    l_attn_mask = None
                    g_attn_mask = None
                    t5_attn_mask = None

                if lg_out is None:
                    # not cached or training, so get from text encoders
                    input_ids_clip_l, input_ids_clip_g, _, l_attn_mask, g_attn_mask, _ = batch["input_ids_list"]
                    with torch.set_grad_enabled(train_clip):
                        # TODO support weighted captions
                        # text models in sd3_models require "cpu" for input_ids
                        input_ids_clip_l = input_ids_clip_l.to("cpu")
                        input_ids_clip_g = input_ids_clip_g.to("cpu")
                        lg_out, _, lg_pooled, l_attn_mask, g_attn_mask, _ = text_encoding_strategy.encode_tokens(
                            sd3_tokenize_strategy,
                            [clip_l, clip_g, None],
                            [input_ids_clip_l, input_ids_clip_g, None, l_attn_mask, g_attn_mask, None],
                        )

                if t5_out is None:
                    _, _, input_ids_t5xxl, _, _, t5_attn_mask = batch["input_ids_list"]
                    with torch.set_grad_enabled(train_t5xxl):
                        input_ids_t5xxl = input_ids_t5xxl.to("cpu")
                        _, t5_out, _, _, _, t5_attn_mask = text_encoding_strategy.encode_tokens(
                            sd3_tokenize_strategy, [None, None, t5xxl], [None, None, input_ids_t5xxl, None, None, t5_attn_mask]
                        )

                context, lg_pooled = text_encoding_strategy.concat_encodings(lg_out, t5_out, lg_pooled)

                # TODO support some features for noise implemented in get_noise_noisy_latents_and_timesteps

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                # bsz = latents.shape[0]

                # get noisy model input and timesteps
                noisy_model_input, timesteps, sigmas = sd3_train_utils.get_noisy_model_input_and_timesteps(
                    args, latents, noise, accelerator.device, weight_dtype
                )

                # debug: NaN check for all inputs
                if torch.any(torch.isnan(noisy_model_input)):
                    accelerator.print("NaN found in noisy_model_input, replacing with zeros")
                    noisy_model_input = torch.nan_to_num(noisy_model_input, 0, out=noisy_model_input)
                if torch.any(torch.isnan(context)):
                    accelerator.print("NaN found in context, replacing with zeros")
                    context = torch.nan_to_num(context, 0, out=context)
                if torch.any(torch.isnan(lg_pooled)):
                    accelerator.print("NaN found in pool, replacing with zeros")
                    lg_pooled = torch.nan_to_num(lg_pooled, 0, out=lg_pooled)

                # call model
                with accelerator.autocast():
                    # TODO support attention mask
                    model_pred = mmdit(noisy_model_input, timesteps, context=context, y=lg_pooled)

                # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                # Preconditioning of the model outputs.
                model_pred = model_pred * (-sigmas) + noisy_model_input

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = sd3_train_utils.compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = latents

                # # Compute regular loss. TODO simplify this
                # loss = torch.mean(
                #     (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                #     1,
                # )
                # calculate loss
                huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, dummy_scheduler)
                loss = train_util.conditional_loss(model_pred.float(), target.float(), args.loss_type, "none", huber_c)
                if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                    loss = apply_masked_loss(loss, batch)
                loss = loss.mean([1, 2, 3])

                if weighting is not None:
                    loss = loss * weighting

                loss_weights = batch["loss_weights"]  # 各sampleごとのweight
                loss = loss * loss_weights
                loss = loss.mean()

                accelerator.backward(loss)

                if not (args.fused_backward_pass or args.blockwise_fused_optimizers):
                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        params_to_clip = []
                        for m in training_models:
                            params_to_clip.extend(m.parameters())
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    # optimizer.step() and optimizer.zero_grad() are called in the optimizer hook
                    lr_scheduler.step()
                    if args.blockwise_fused_optimizers:
                        for i in range(1, len(optimizers)):
                            lr_schedulers[i].step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                optimizer_eval_fn()
                sd3_train_utils.sample_images(
                    accelerator, args, None, global_step, mmdit, vae, [clip_l, clip_g, t5xxl], sample_prompts_te_outputs
                )

                # 指定ステップごとにモデルを保存
                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        sd3_train_utils.save_sd3_model_on_epoch_end_or_stepwise(
                            args,
                            False,
                            accelerator,
                            save_dtype,
                            epoch,
                            num_train_epochs,
                            global_step,
                            accelerator.unwrap_model(clip_l) if train_clip else None,
                            accelerator.unwrap_model(clip_g) if train_clip else None,
                            accelerator.unwrap_model(t5xxl) if train_t5xxl else None,
                            accelerator.unwrap_model(mmdit) if train_mmdit else None,
                            vae,
                        )
                optimizer_train_fn()

            current_loss = loss.detach().item()  # 平均なのでbatch sizeは関係ないはず
            if len(accelerator.trackers) > 0:
                logs = {"loss": current_loss}
                train_util.append_lr_to_logs(logs, lr_scheduler, args.optimizer_type, including_unet=train_mmdit)

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

        optimizer_eval_fn()
        if args.save_every_n_epochs is not None:
            if accelerator.is_main_process:
                sd3_train_utils.save_sd3_model_on_epoch_end_or_stepwise(
                    args,
                    True,
                    accelerator,
                    save_dtype,
                    epoch,
                    num_train_epochs,
                    global_step,
                    accelerator.unwrap_model(clip_l) if train_clip else None,
                    accelerator.unwrap_model(clip_g) if train_clip else None,
                    accelerator.unwrap_model(t5xxl) if train_t5xxl else None,
                    accelerator.unwrap_model(mmdit) if train_mmdit else None,
                    vae,
                )

        sd3_train_utils.sample_images(
            accelerator, args, epoch + 1, global_step, mmdit, vae, [clip_l, clip_g, t5xxl], sample_prompts_te_outputs
        )

    is_main_process = accelerator.is_main_process
    # if is_main_process:
    mmdit = accelerator.unwrap_model(mmdit)
    clip_l = accelerator.unwrap_model(clip_l)
    clip_g = accelerator.unwrap_model(clip_g)
    if t5xxl is not None:
        t5xxl = accelerator.unwrap_model(t5xxl)

    accelerator.end_training()
    optimizer_eval_fn()

    if args.save_state or args.save_state_on_train_end:
        train_util.save_state_on_train_end(args, accelerator)

    del accelerator  # この後メモリを使うのでこれは消す

    if is_main_process:
        sd3_train_utils.save_sd3_model_on_train_end(
            args,
            save_dtype,
            epoch,
            global_step,
            clip_l if train_clip else None,
            clip_g if train_clip else None,
            t5xxl if train_t5xxl else None,
            mmdit if train_mmdit else None,
            vae,
        )
        logger.info("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, False)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    add_custom_train_arguments(parser)
    train_util.add_dit_training_arguments(parser)
    sd3_train_utils.add_sd3_training_arguments(parser)

    parser.add_argument(
        "--train_text_encoder", action="store_true", help="train text encoder (CLIP-L and G) / text encoderも学習する"
    )
    parser.add_argument("--train_t5xxl", action="store_true", help="train T5-XXL / T5-XXLも学習する")
    parser.add_argument(
        "--use_t5xxl_cache_only", action="store_true", help="cache T5-XXL outputs only / T5-XXLの出力のみキャッシュする"
    )

    parser.add_argument(
        "--learning_rate_te1",
        type=float,
        default=None,
        help="learning rate for text encoder 1 (ViT-L) / text encoder 1 (ViT-L)の学習率",
    )
    parser.add_argument(
        "--learning_rate_te2",
        type=float,
        default=None,
        help="learning rate for text encoder 2 (BiG-G) / text encoder 2 (BiG-G)の学習率",
    )
    parser.add_argument(
        "--learning_rate_te3",
        type=float,
        default=None,
        help="learning rate for text encoder 3 (T5-XXL) / text encoder 3 (T5-XXL)の学習率",
    )

    # parser.add_argument(
    #     "--diffusers_xformers", action="store_true", help="use xformers by diffusers / Diffusersでxformersを使用する"
    # )
    # parser.add_argument(
    #     "--no_half_vae",
    #     action="store_true",
    #     help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    # )
    # parser.add_argument(
    #     "--block_lr",
    #     type=str,
    #     default=None,
    #     help=f"learning rates for each block of U-Net, comma-separated, {UNET_NUM_BLOCKS_FOR_BLOCK_LR} values / "
    #     + f"U-Netの各ブロックの学習率、カンマ区切り、{UNET_NUM_BLOCKS_FOR_BLOCK_LR}個の値",
    # )
    parser.add_argument(
        "--blockwise_fused_optimizers",
        action="store_true",
        help="enable blockwise optimizers for fused backward pass and optimizer step / fused backward passとoptimizer step のためブロック単位のoptimizerを有効にする",
    )
    parser.add_argument(
        "--fused_optimizer_groups",
        type=int,
        default=None,
        help="[DOES NOT WORK] number of optimizer groups for fused backward pass and optimizer step / fused backward passとoptimizer stepのためのoptimizerグループ数",
    )
    parser.add_argument(
        "--skip_latents_validity_check",
        action="store_true",
        help="[Deprecated] use 'skip_cache_check' instead / 代わりに 'skip_cache_check' を使用してください",
    )
    parser.add_argument(
        "--num_last_block_to_freeze",
        type=int,
        default=None,
        help="freeze last n blocks of MM-DIT / MM-DITの最後のnブロックを凍結する",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    train(args)
