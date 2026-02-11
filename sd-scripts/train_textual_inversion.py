import argparse
import math
import os
from multiprocessing import Value
from typing import Any, List, Optional, Union
import toml

from tqdm import tqdm

import torch
from library.device_utils import init_ipex, clean_memory_on_device


init_ipex()

from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from transformers import CLIPTokenizer
from library import deepspeed_utils, model_util, strategy_base, strategy_sd, sai_model_spec

import library.train_util as train_util
import library.huggingface_util as huggingface_util
import library.config_util as config_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    apply_snr_weight,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
    apply_masked_loss,
)
from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


class TextualInversionTrainer:
    def __init__(self):
        self.vae_scale_factor = 0.18215
        self.is_sdxl = False

    def assert_extra_args(self, args, train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset], val_dataset_group: Optional[train_util.DatasetGroup]):
        train_dataset_group.verify_bucket_reso_steps(64)

        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(64)

    def load_target_model(self, args, weight_dtype, accelerator):
        text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)
        return model_util.get_model_version_str_for_sd1_sd2(args.v2, args.v_parameterization), [text_encoder], vae, unet

    def get_tokenize_strategy(self, args):
        return strategy_sd.SdTokenizeStrategy(args.v2, args.max_token_length, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_sd.SdTokenizeStrategy) -> List[Any]:
        return [tokenize_strategy.tokenizer]

    def get_latents_caching_strategy(self, args):
        latents_caching_strategy = strategy_sd.SdSdxlLatentsCachingStrategy(
            True, args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        return latents_caching_strategy

    def assert_token_string(self, token_string, tokenizers: CLIPTokenizer):
        pass

    def get_text_encoding_strategy(self, args):
        return strategy_sd.SdTextEncodingStrategy(args.clip_skip)

    def get_models_for_text_encoding(self, args, accelerator, text_encoders) -> List[Any]:
        return text_encoders

    def call_unet(self, args, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype):
        noise_pred = unet(noisy_latents, timesteps, text_conds[0]).sample
        return noise_pred

    def sample_images(
        self, accelerator, args, epoch, global_step, device, vae, tokenizers, text_encoders, unet, prompt_replacement
    ):
        train_util.sample_images(
            accelerator, args, epoch, global_step, device, vae, tokenizers[0], text_encoders[0], unet, prompt_replacement
        )

    def save_weights(self, file, updated_embs, save_dtype, metadata):
        state_dict = {"emb_params": updated_embs[0]}

        if save_dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(save_dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)  # can be loaded in Web UI

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            data = load_file(file)
        else:
            # compatible to Web UI's file format
            data = torch.load(file, map_location="cpu")
            if type(data) != dict:
                raise ValueError(f"weight file is not dict / 重みファイルがdict形式ではありません: {file}")

            if "string_to_param" in data:  # textual inversion embeddings
                data = data["string_to_param"]
                if hasattr(data, "_parameters"):  # support old PyTorch?
                    data = getattr(data, "_parameters")

        emb = next(iter(data.values()))
        if type(emb) != torch.Tensor:
            raise ValueError(f"weight file does not contains Tensor / 重みファイルのデータがTensorではありません: {file}")

        if len(emb.size()) == 1:
            emb = emb.unsqueeze(0)

        return [emb]

    def train(self, args):
        if args.output_name is None:
            args.output_name = args.token_string
        use_template = args.use_object_template or args.use_style_template

        train_util.verify_training_args(args)
        train_util.prepare_dataset_args(args, True)
        setup_logging(args, reset=True)

        cache_latents = args.cache_latents

        if args.seed is not None:
            set_seed(args.seed)

        tokenize_strategy = self.get_tokenize_strategy(args)
        strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
        tokenizers = self.get_tokenizers(tokenize_strategy)  # will be removed after sample_image is refactored

        # prepare caching strategy: this must be set before preparing dataset. because dataset may use this strategy for initialization.
        latents_caching_strategy = self.get_latents_caching_strategy(args)
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

        # acceleratorを準備する
        logger.info("prepare accelerator")
        accelerator = train_util.prepare_accelerator(args)

        # mixed precisionに対応した型を用意しておき適宜castする
        weight_dtype, save_dtype = train_util.prepare_dtype(args)
        vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

        # モデルを読み込む
        model_version, text_encoders, vae, unet = self.load_target_model(args, weight_dtype, accelerator)

        # Convert the init_word to token_id
        init_token_ids_list = []
        if args.init_word is not None:
            for i, tokenizer in enumerate(tokenizers):
                init_token_ids = tokenizer.encode(args.init_word, add_special_tokens=False)
                if len(init_token_ids) > 1 and len(init_token_ids) != args.num_vectors_per_token:
                    accelerator.print(
                        f"token length for init words is not same to num_vectors_per_token, init words is repeated or truncated / "
                        + f"初期化単語のトークン長がnum_vectors_per_tokenと合わないため、繰り返しまたは切り捨てが発生します:  tokenizer {i+1}, length {len(init_token_ids)}"
                    )
                init_token_ids_list.append(init_token_ids)
        else:
            init_token_ids_list = [None] * len(tokenizers)

        # tokenizerに新しい単語を追加する。追加する単語の数はnum_vectors_per_token
        # token_stringが hoge の場合、"hoge", "hoge1", "hoge2", ... が追加される
        # add new word to tokenizer, count is num_vectors_per_token
        # if token_string is hoge, "hoge", "hoge1", "hoge2", ... are added

        self.assert_token_string(args.token_string, tokenizers)

        token_strings = [args.token_string] + [f"{args.token_string}{i+1}" for i in range(args.num_vectors_per_token - 1)]
        token_ids_list = []
        token_embeds_list = []
        for i, (tokenizer, text_encoder, init_token_ids) in enumerate(zip(tokenizers, text_encoders, init_token_ids_list)):
            num_added_tokens = tokenizer.add_tokens(token_strings)
            assert (
                num_added_tokens == args.num_vectors_per_token
            ), f"tokenizer has same word to token string. please use another one / 指定したargs.token_stringは既に存在します。別の単語を使ってください: tokenizer {i+1}, {args.token_string}"

            token_ids = tokenizer.convert_tokens_to_ids(token_strings)
            accelerator.print(f"tokens are added for tokenizer {i+1}: {token_ids}")
            assert (
                min(token_ids) == token_ids[0] and token_ids[-1] == token_ids[0] + len(token_ids) - 1
            ), f"token ids is not ordered : tokenizer {i+1}, {token_ids}"
            assert (
                len(tokenizer) - 1 == token_ids[-1]
            ), f"token ids is not end of tokenize: tokenizer {i+1}, {token_ids}, {len(tokenizer)}"
            token_ids_list.append(token_ids)

            # Resize the token embeddings as we are adding new special tokens to the tokenizer
            text_encoder.resize_token_embeddings(len(tokenizer))

            # Initialise the newly added placeholder token with the embeddings of the initializer token
            token_embeds = text_encoder.get_input_embeddings().weight.data
            if init_token_ids is not None:
                for i, token_id in enumerate(token_ids):
                    token_embeds[token_id] = token_embeds[init_token_ids[i % len(init_token_ids)]]
                    # accelerator.print(token_id, token_embeds[token_id].mean(), token_embeds[token_id].min())
            token_embeds_list.append(token_embeds)

        # load weights
        if args.weights is not None:
            embeddings_list = self.load_weights(args.weights)
            assert len(token_ids) == len(
                embeddings_list[0]
            ), f"num_vectors_per_token is mismatch for weights / 指定した重みとnum_vectors_per_tokenの値が異なります: {len(embeddings)}"
            # accelerator.print(token_ids, embeddings.size())
            for token_ids, embeddings, token_embeds in zip(token_ids_list, embeddings_list, token_embeds_list):
                for token_id, embedding in zip(token_ids, embeddings):
                    token_embeds[token_id] = embedding
                    # accelerator.print(token_id, token_embeds[token_id].mean(), token_embeds[token_id].min())
            accelerator.print(f"weighs loaded")

        accelerator.print(f"create embeddings for {args.num_vectors_per_token} tokens, for {args.token_string}")

        # データセットを準備する
        if args.dataset_class is None:
            blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, False))
            if args.dataset_config is not None:
                accelerator.print(f"Load dataset config from {args.dataset_config}")
                user_config = config_util.load_user_config(args.dataset_config)
                ignored = ["train_data_dir", "reg_data_dir", "in_json"]
                if any(getattr(args, attr) is not None for attr in ignored):
                    accelerator.print(
                        "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                            ", ".join(ignored)
                        )
                    )
            else:
                use_dreambooth_method = args.in_json is None
                if use_dreambooth_method:
                    accelerator.print("Use DreamBooth method.")
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
                    logger.info("Train with captions.")
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

        self.assert_extra_args(args, train_dataset_group, val_dataset_group)

        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

        # make captions: tokenstring tokenstring1 tokenstring2 ...tokenstringn という文字列に書き換える超乱暴な実装
        if use_template:
            accelerator.print(f"use template for training captions. is object: {args.use_object_template}")
            templates = imagenet_templates_small if args.use_object_template else imagenet_style_templates_small
            replace_to = " ".join(token_strings)
            captions = []
            for tmpl in templates:
                captions.append(tmpl.format(replace_to))
            train_dataset_group.add_replacement("", captions)

            # サンプル生成用
            if args.num_vectors_per_token > 1:
                prompt_replacement = (args.token_string, replace_to)
            else:
                prompt_replacement = None
        else:
            # サンプル生成用
            if args.num_vectors_per_token > 1:
                replace_to = " ".join(token_strings)
                train_dataset_group.add_replacement(args.token_string, replace_to)
                prompt_replacement = (args.token_string, replace_to)
            else:
                prompt_replacement = None

        if args.debug_dataset:
            train_util.debug_dataset(train_dataset_group, show_input_ids=True)
            return
        if len(train_dataset_group) == 0:
            accelerator.print("No data found. Please verify arguments / 画像がありません。引数指定を確認してください")
            return

        if cache_latents:
            assert (
                train_dataset_group.is_latent_cacheable()
            ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

        # モデルに xformers とか memory efficient attention を組み込む
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

        # 学習を準備する
        if cache_latents:
            vae.to(accelerator.device, dtype=vae_dtype)
            vae.requires_grad_(False)
            vae.eval()

            train_dataset_group.new_cache_latents(vae, accelerator)

            clean_memory_on_device(accelerator.device)
            accelerator.wait_for_everyone()

        if args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            for text_encoder in text_encoders:
                text_encoder.gradient_checkpointing_enable()

        # 学習に必要なクラスを準備する
        accelerator.print("prepare optimizer, data loader etc.")
        trainable_params = []
        for text_encoder in text_encoders:
            trainable_params += text_encoder.get_input_embeddings().parameters()
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

        # acceleratorがなんかよろしくやってくれるらしい
        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)
        text_encoders = [accelerator.prepare(text_encoder) for text_encoder in text_encoders]

        index_no_updates_list = []
        orig_embeds_params_list = []
        for tokenizer, token_ids, text_encoder in zip(tokenizers, token_ids_list, text_encoders):
            index_no_updates = torch.arange(len(tokenizer)) < token_ids[0]
            index_no_updates_list.append(index_no_updates)

            # accelerator.print(len(index_no_updates), torch.sum(index_no_updates))
            orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.detach().clone()
            orig_embeds_params_list.append(orig_embeds_params)

            # Freeze all parameters except for the token embeddings in text encoder
            text_encoder.requires_grad_(True)
            unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
            unwrapped_text_encoder.text_model.encoder.requires_grad_(False)
            unwrapped_text_encoder.text_model.final_layer_norm.requires_grad_(False)
            unwrapped_text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
            # text_encoder.text_model.embeddings.token_embedding.requires_grad_(True)

        unet.requires_grad_(False)
        unet.to(accelerator.device, dtype=weight_dtype)
        if args.gradient_checkpointing:  # according to TI example in Diffusers, train is required
            # TODO U-Netをオリジナルに置き換えたのでいらないはずなので、後で確認して消す
            unet.train()
        else:
            unet.eval()

        text_encoding_strategy = self.get_text_encoding_strategy(args)
        strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

        if not cache_latents:  # キャッシュしない場合はVAEを使うのでVAEを準備する
            vae.requires_grad_(False)
            vae.eval()
            vae.to(accelerator.device, dtype=vae_dtype)

        # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
        if args.full_fp16:
            train_util.patch_accelerator_for_fp16_training(accelerator)
            for text_encoder in text_encoders:
                text_encoder.to(weight_dtype)
        if args.full_bf16:
            for text_encoder in text_encoders:
                text_encoder.to(weight_dtype)

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
                "textual_inversion" if args.log_tracker_name is None else args.log_tracker_name,
                config=train_util.get_sanitized_config_or_none(args),
                init_kwargs=init_kwargs,
            )

        # function for saving/removing
        def save_model(ckpt_name, embs_list, steps, epoch_no, force_sync_upload=False):
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_file = os.path.join(args.output_dir, ckpt_name)

            accelerator.print(f"\nsaving checkpoint: {ckpt_file}")

            sai_metadata = train_util.get_sai_model_spec(None, args, self.is_sdxl, False, True)

            self.save_weights(ckpt_file, embs_list, save_dtype, sai_metadata)
            if args.huggingface_repo_id is not None:
                huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

        def remove_model(old_ckpt_name):
            old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
            if os.path.exists(old_ckpt_file):
                accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)

        # For --sample_at_first
        self.sample_images(
            accelerator,
            args,
            0,
            global_step,
            accelerator.device,
            vae,
            tokenizers,
            text_encoders,
            unet,
            prompt_replacement,
        )
        if len(accelerator.trackers) > 0:
            # log empty object to commit the sample images to wandb
            accelerator.log({}, step=0)

        # training loop
        for epoch in range(num_train_epochs):
            accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
            current_epoch.value = epoch + 1

            for text_encoder in text_encoders:
                text_encoder.train()

            loss_total = 0

            for step, batch in enumerate(train_dataloader):
                current_step.value = global_step
                with accelerator.accumulate(text_encoders[0]):
                    with torch.no_grad():
                        if "latents" in batch and batch["latents"] is not None:
                            latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
                        else:
                            # latentに変換
                            latents = vae.encode(batch["images"].to(dtype=vae_dtype)).latent_dist.sample().to(dtype=weight_dtype)
                        latents = latents * self.vae_scale_factor

                    # Get the text embedding for conditioning
                    input_ids = [ids.to(accelerator.device) for ids in batch["input_ids_list"]]
                    text_encoder_conds = text_encoding_strategy.encode_tokens(
                        tokenize_strategy, self.get_models_for_text_encoding(args, accelerator, text_encoders), input_ids
                    )
                    if args.full_fp16:
                        text_encoder_conds = [c.to(weight_dtype) for c in text_encoder_conds]

                    # Sample noise, sample a random timestep for each image, and add noise to the latents,
                    # with noise offset and/or multires noise if specified
                    noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(
                        args, noise_scheduler, latents
                    )

                    # Predict the noise residual
                    with accelerator.autocast():
                        noise_pred = self.call_unet(
                            args, accelerator, unet, noisy_latents, timesteps, text_encoder_conds, batch, weight_dtype
                        )

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
                    if args.v_pred_like_loss:
                        loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)
                    if args.debiased_estimation_loss:
                        loss = apply_debiased_estimation(loss, timesteps, noise_scheduler, args.v_parameterization)

                    loss = loss.mean()  # 平均なのでbatch_sizeで割る必要なし

                    accelerator.backward(loss)
                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        params_to_clip = accelerator.unwrap_model(text_encoder).get_input_embeddings().parameters()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    # Let's make sure we don't update any embedding weights besides the newly added token
                    with torch.no_grad():
                        for text_encoder, orig_embeds_params, index_no_updates in zip(
                            text_encoders, orig_embeds_params_list, index_no_updates_list
                        ):
                            # if full_fp16/bf16, input_embeddings_weight is fp16/bf16, orig_embeds_params is fp32
                            input_embeddings_weight = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight
                            input_embeddings_weight[index_no_updates] = orig_embeds_params.to(input_embeddings_weight.dtype)[
                                index_no_updates
                            ]

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    self.sample_images(
                        accelerator,
                        args,
                        None,
                        global_step,
                        accelerator.device,
                        vae,
                        tokenizers,
                        text_encoders,
                        unet,
                        prompt_replacement,
                    )

                    # 指定ステップごとにモデルを保存
                    if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            updated_embs_list = []
                            for text_encoder, token_ids in zip(text_encoders, token_ids_list):
                                updated_embs = (
                                    accelerator.unwrap_model(text_encoder)
                                    .get_input_embeddings()
                                    .weight[token_ids]
                                    .data.detach()
                                    .clone()
                                )
                                updated_embs_list.append(updated_embs)

                            ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, global_step)
                            save_model(ckpt_name, updated_embs_list, global_step, epoch)

                            if args.save_state:
                                train_util.save_and_remove_state_stepwise(args, accelerator, global_step)

                            remove_step_no = train_util.get_remove_step_no(args, global_step)
                            if remove_step_no is not None:
                                remove_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no)
                                remove_model(remove_ckpt_name)

                current_loss = loss.detach().item()
                if len(accelerator.trackers) > 0:
                    logs = {"loss": current_loss, "lr": float(lr_scheduler.get_last_lr()[0])}
                    if (
                        args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower()
                    ):  # tracking d*lr value
                        logs["lr/d*lr"] = (
                            lr_scheduler.optimizers[0].param_groups[0]["d"] * lr_scheduler.optimizers[0].param_groups[0]["lr"]
                        )
                    accelerator.log(logs, step=global_step)

                loss_total += current_loss
                avr_loss = loss_total / (step + 1)
                logs = {"loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= args.max_train_steps:
                    break

            if len(accelerator.trackers) > 0:
                logs = {"loss/epoch": loss_total / len(train_dataloader)}
                accelerator.log(logs, step=epoch + 1)

            accelerator.wait_for_everyone()

            updated_embs_list = []
            for text_encoder, token_ids in zip(text_encoders, token_ids_list):
                updated_embs = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[token_ids].data.detach().clone()
                updated_embs_list.append(updated_embs)

            if args.save_every_n_epochs is not None:
                saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
                if accelerator.is_main_process and saving:
                    ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
                    save_model(ckpt_name, updated_embs_list, epoch + 1, global_step)

                    remove_epoch_no = train_util.get_remove_epoch_no(args, epoch + 1)
                    if remove_epoch_no is not None:
                        remove_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no)
                        remove_model(remove_ckpt_name)

                    if args.save_state:
                        train_util.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)

            self.sample_images(
                accelerator,
                args,
                epoch + 1,
                global_step,
                accelerator.device,
                vae,
                tokenizers,
                text_encoders,
                unet,
                prompt_replacement,
            )
            accelerator.log({})

            # end of epoch

        is_main_process = accelerator.is_main_process
        if is_main_process:
            text_encoder = accelerator.unwrap_model(text_encoder)
            updated_embs = text_encoder.get_input_embeddings().weight[token_ids].data.detach().clone()

        accelerator.end_training()

        if is_main_process and (args.save_state or args.save_state_on_train_end):
            train_util.save_state_on_train_end(args, accelerator)

        if is_main_process:
            ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as)
            save_model(ckpt_name, updated_embs_list, global_step, num_train_epochs, force_sync_upload=True)

            logger.info("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    sai_model_spec.add_model_spec_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, False)
    train_util.add_training_arguments(parser, True)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser, False)

    parser.add_argument(
        "--save_model_as",
        type=str,
        default="pt",
        choices=[None, "ckpt", "pt", "safetensors"],
        help="format to save the model (default is .pt) / モデル保存時の形式（デフォルトはpt）",
    )

    parser.add_argument(
        "--weights", type=str, default=None, help="embedding weights to initialize / 学習するネットワークの初期重み"
    )
    parser.add_argument(
        "--num_vectors_per_token", type=int, default=1, help="number of vectors per token / トークンに割り当てるembeddingsの要素数"
    )
    parser.add_argument(
        "--token_string",
        type=str,
        default=None,
        help="token string used in training, must not exist in tokenizer / 学習時に使用されるトークン文字列、tokenizerに存在しない文字であること",
    )
    parser.add_argument(
        "--init_word", type=str, default=None, help="words to initialize vector / ベクトルを初期化に使用する単語、複数可"
    )
    parser.add_argument(
        "--use_object_template",
        action="store_true",
        help="ignore caption and use default templates for object / キャプションは使わずデフォルトの物体用テンプレートで学習する",
    )
    parser.add_argument(
        "--use_style_template",
        action="store_true",
        help="ignore caption and use default templates for stype / キャプションは使わずデフォルトのスタイル用テンプレートで学習する",
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

    trainer = TextualInversionTrainer()
    trainer.train(args)
