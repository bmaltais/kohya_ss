# training with captions

import argparse
import math
import os
import random
import json
import importlib
import time

from tqdm import tqdm
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import CLIPTokenizer
import diffusers
from diffusers import DDPMScheduler, StableDiffusionPipeline
import numpy as np
from einops import rearrange
from torch import einsum

import library.model_util as model_util
import library.train_util as train_util


def collate_fn(examples):
  return examples[0]


class FineTuningDataset(torch.utils.data.Dataset):
  def __init__(self, metadata, train_data_dir, batch_size, tokenizer, max_token_length, shuffle_caption, shuffle_keep_tokens, dataset_repeats, debug) -> None:
    super().__init__()

    self.metadata = metadata
    self.train_data_dir = train_data_dir
    self.batch_size = batch_size
    self.tokenizer: CLIPTokenizer = tokenizer
    self.max_token_length = max_token_length
    self.shuffle_caption = shuffle_caption
    self.shuffle_keep_tokens = shuffle_keep_tokens
    self.debug = debug

    self.tokenizer_max_length = self.tokenizer.model_max_length if max_token_length is None else max_token_length + 2

    print("make buckets")

    # 最初に数を数える
    self.bucket_resos = set()
    for img_md in metadata.values():
      if 'train_resolution' in img_md:
        self.bucket_resos.add(tuple(img_md['train_resolution']))
    self.bucket_resos = list(self.bucket_resos)
    self.bucket_resos.sort()
    print(f"number of buckets: {len(self.bucket_resos)}")

    reso_to_index = {}
    for i, reso in enumerate(self.bucket_resos):
      reso_to_index[reso] = i

    # bucketに割り当てていく
    self.buckets = [[] for _ in range(len(self.bucket_resos))]
    n = 1 if dataset_repeats is None else dataset_repeats
    images_count = 0
    for image_key, img_md in metadata.items():
      if 'train_resolution' not in img_md:
        continue
      if not os.path.exists(self.image_key_to_npz_file(image_key)):
        continue

      reso = tuple(img_md['train_resolution'])
      for _ in range(n):
        self.buckets[reso_to_index[reso]].append(image_key)
      images_count += n

    # 参照用indexを作る
    self.buckets_indices = []
    for bucket_index, bucket in enumerate(self.buckets):
      batch_count = int(math.ceil(len(bucket) / self.batch_size))
      for batch_index in range(batch_count):
        self.buckets_indices.append((bucket_index, batch_index))

    self.shuffle_buckets()
    self._length = len(self.buckets_indices)
    self.images_count = images_count

  def show_buckets(self):
    for i, (reso, bucket) in enumerate(zip(self.bucket_resos, self.buckets)):
      print(f"bucket {i}: resolution {reso}, count: {len(bucket)}")

  def shuffle_buckets(self):
    random.shuffle(self.buckets_indices)
    for bucket in self.buckets:
      random.shuffle(bucket)

  def image_key_to_npz_file(self, image_key):
    npz_file_norm = os.path.splitext(image_key)[0] + '.npz'
    if os.path.exists(npz_file_norm):
      if random.random() < .5:
        npz_file_flip = os.path.splitext(image_key)[0] + '_flip.npz'
        if os.path.exists(npz_file_flip):
          return npz_file_flip
      return npz_file_norm

    npz_file_norm = os.path.join(self.train_data_dir, image_key + '.npz')
    if random.random() < .5:
      npz_file_flip = os.path.join(self.train_data_dir, image_key + '_flip.npz')
      if os.path.exists(npz_file_flip):
        return npz_file_flip
    return npz_file_norm

  def load_latent(self, image_key):
    return np.load(self.image_key_to_npz_file(image_key))['arr_0']

  def __len__(self):
    return self._length

  def __getitem__(self, index):
    if index == 0:
      self.shuffle_buckets()

    bucket = self.buckets[self.buckets_indices[index][0]]
    image_index = self.buckets_indices[index][1] * self.batch_size

    input_ids_list = []
    latents_list = []
    captions = []
    for image_key in bucket[image_index:image_index + self.batch_size]:
      img_md = self.metadata[image_key]
      caption = img_md.get('caption')
      tags = img_md.get('tags')

      if caption is None:
        caption = tags
      elif tags is not None and len(tags) > 0:
        caption = caption + ', ' + tags
      assert caption is not None and len(caption) > 0, f"caption or tag is required / キャプションまたはタグは必須です:{image_key}"

      latents = self.load_latent(image_key)

      if self.shuffle_caption:
        tokens = caption.strip().split(",")
        if self.shuffle_keep_tokens is None:
          random.shuffle(tokens)
        else:
          if len(tokens) > self.shuffle_keep_tokens:
            keep_tokens = tokens[:self.shuffle_keep_tokens]
            tokens = tokens[self.shuffle_keep_tokens:]
            random.shuffle(tokens)
            tokens = keep_tokens + tokens
        caption = ",".join(tokens).strip()

      captions.append(caption)

      input_ids = self.tokenizer(caption, padding="max_length", truncation=True,
                                 max_length=self.tokenizer_max_length, return_tensors="pt").input_ids

      if self.tokenizer_max_length > self.tokenizer.model_max_length:
        input_ids = input_ids.squeeze(0)
        iids_list = []
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
          # v1
          # 77以上の時は "<BOS> .... <EOS> <EOS> <EOS>" でトータル227とかになっているので、"<BOS>...<EOS>"の三連に変換する
          # 1111氏のやつは , で区切る、とかしているようだが　とりあえず単純に
          for i in range(1, self.tokenizer_max_length - self.tokenizer.model_max_length + 2, self.tokenizer.model_max_length - 2):  # (1, 152, 75)
            ids_chunk = (input_ids[0].unsqueeze(0),
                         input_ids[i:i + self.tokenizer.model_max_length - 2],
                         input_ids[-1].unsqueeze(0))
            ids_chunk = torch.cat(ids_chunk)
            iids_list.append(ids_chunk)
        else:
          # v2
          # 77以上の時は "<BOS> .... <EOS> <PAD> <PAD>..." でトータル227とかになっているので、"<BOS>...<EOS> <PAD> <PAD> ..."の三連に変換する
          for i in range(1, self.tokenizer_max_length - self.tokenizer.model_max_length + 2, self.tokenizer.model_max_length - 2):
            ids_chunk = (input_ids[0].unsqueeze(0),       # BOS
                         input_ids[i:i + self.tokenizer.model_max_length - 2],
                         input_ids[-1].unsqueeze(0))      # PAD or EOS
            ids_chunk = torch.cat(ids_chunk)

            # 末尾が <EOS> <PAD> または <PAD> <PAD> の場合は、何もしなくてよい
            # 末尾が x <PAD/EOS> の場合は末尾を <EOS> に変える（x <EOS> なら結果的に変化なし）
            if ids_chunk[-2] != self.tokenizer.eos_token_id and ids_chunk[-2] != self.tokenizer.pad_token_id:
              ids_chunk[-1] = self.tokenizer.eos_token_id
            # 先頭が <BOS> <PAD> ... の場合は <BOS> <EOS> <PAD> ... に変える
            if ids_chunk[1] == self.tokenizer.pad_token_id:
              ids_chunk[1] = self.tokenizer.eos_token_id

            iids_list.append(ids_chunk)

        input_ids = torch.stack(iids_list)      # 3,77

      input_ids_list.append(input_ids)
      latents_list.append(torch.FloatTensor(latents))

    example = {}
    example['input_ids'] = torch.stack(input_ids_list)
    example['latents'] = torch.stack(latents_list)
    if self.debug:
      example['image_keys'] = bucket[image_index:image_index + self.batch_size]
      example['captions'] = captions
    return example


def save_hypernetwork(output_file, hypernetwork):
  state_dict = hypernetwork.get_state_dict()
  torch.save(state_dict, output_file)


def train(args):
  fine_tuning = args.hypernetwork_module is None            # fine tuning or hypernetwork training

  # その他のオプション設定を確認する
  if args.v_parameterization and not args.v2:
    print("v_parameterization should be with v2 / v1でv_parameterizationを使用することは想定されていません")
  if args.v2 and args.clip_skip is not None:
    print("v2 with clip_skip will be unexpected / v2でclip_skipを使用することは想定されていません")

  # モデル形式のオプション設定を確認する
  load_stable_diffusion_format = os.path.isfile(args.pretrained_model_name_or_path)

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
    save_stable_diffusion_format = args.save_model_as.lower() == 'ckpt' or args.save_model_as.lower() == 'safetensors'
    use_safetensors = args.use_safetensors or ("safetensors" in args.save_model_as.lower())

  # 乱数系列を初期化する
  if args.seed is not None:
    set_seed(args.seed)

  # メタデータを読み込む
  if os.path.exists(args.in_json):
    print(f"loading existing metadata: {args.in_json}")
    with open(args.in_json, "rt", encoding='utf-8') as f:
      metadata = json.load(f)
  else:
    print(f"no metadata / メタデータファイルがありません: {args.in_json}")
    return

  # tokenizerを読み込む
  print("prepare tokenizer")
  if args.v2:
    tokenizer = CLIPTokenizer.from_pretrained(train_util.V2_STABLE_DIFFUSION_PATH, subfolder="tokenizer")
  else:
    tokenizer = CLIPTokenizer.from_pretrained(train_util.TOKENIZER_PATH)

  if args.max_token_length is not None:
    print(f"update token length: {args.max_token_length}")

  # datasetを用意する
  print("prepare dataset")
  train_dataset = FineTuningDataset(metadata, args.train_data_dir, args.train_batch_size,
                                    tokenizer, args.max_token_length, args.shuffle_caption, args.keep_tokens,
                                    args.dataset_repeats, args.debug_dataset)

  print(f"Total dataset length / データセットの長さ: {len(train_dataset)}")
  print(f"Total images / 画像数: {train_dataset.images_count}")

  if len(train_dataset) == 0:
    print("No data found. Please verify the metadata file and train_data_dir option. / 画像がありません。メタデータおよびtrain_data_dirオプションを確認してください。")
    return

  if args.debug_dataset:
    train_dataset.show_buckets()
    i = 0
    for example in train_dataset:
      print(f"image: {example['image_keys']}")
      print(f"captions: {example['captions']}")
      print(f"latents: {example['latents'].shape}")
      print(f"input_ids: {example['input_ids'].shape}")
      print(example['input_ids'])
      i += 1
      if i >= 8:
        break
    return

  # acceleratorを準備する
  print("prepare accelerator")
  if args.logging_dir is None:
    log_with = None
    logging_dir = None
  else:
    log_with = "tensorboard"
    log_prefix = "" if args.log_prefix is None else args.log_prefix
    logging_dir = args.logging_dir + "/" + log_prefix + time.strftime('%Y%m%d%H%M%S', time.localtime())
  accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                            mixed_precision=args.mixed_precision, log_with=log_with, logging_dir=logging_dir)

  # accelerateの互換性問題を解決する
  accelerator_0_15 = True
  try:
    accelerator.unwrap_model("dummy", True)
    print("Using accelerator 0.15.0 or above.")
  except TypeError:
    accelerator_0_15 = False

  def unwrap_model(model):
    if accelerator_0_15:
      return accelerator.unwrap_model(model, True)
    return accelerator.unwrap_model(model)

  # mixed precisionに対応した型を用意しておき適宜castする
  weight_dtype = torch.float32
  if args.mixed_precision == "fp16":
    weight_dtype = torch.float16
  elif args.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

  save_dtype = None
  if args.save_precision == "fp16":
    save_dtype = torch.float16
  elif args.save_precision == "bf16":
    save_dtype = torch.bfloat16
  elif args.save_precision == "float":
    save_dtype = torch.float32

  # モデルを読み込む
  if load_stable_diffusion_format:
    print("load StableDiffusion checkpoint")
    text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(args.v2, args.pretrained_model_name_or_path)
  else:
    print("load Diffusers pretrained models")
    pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, tokenizer=None, safety_checker=None)
    # , torch_dtype=weight_dtype) ここでtorch_dtypeを指定すると学習時にエラーになる
    text_encoder = pipe.text_encoder
    unet = pipe.unet
    vae = pipe.vae
    del pipe
  vae.to("cpu")                     # 保存時にしか使わないので、メモリを開けるためCPUに移しておく

  # Diffusers版のxformers使用フラグを設定する関数
  def set_diffusers_xformers_flag(model, valid):
    #   model.set_use_memory_efficient_attention_xformers(valid)            # 次のリリースでなくなりそう
    # pipeが自動で再帰的にset_use_memory_efficient_attention_xformersを探すんだって(;´Д｀)
    # U-Netだけ使う時にはどうすればいいのか……仕方ないからコピって使うか
    # 0.10.2でなんか巻き戻って個別に指定するようになった(;^ω^)

    # Recursively walk through all the children.
    # Any children which exposes the set_use_memory_efficient_attention_xformers method
    # gets the message
    def fn_recursive_set_mem_eff(module: torch.nn.Module):
      if hasattr(module, "set_use_memory_efficient_attention_xformers"):
        module.set_use_memory_efficient_attention_xformers(valid)

      for child in module.children():
        fn_recursive_set_mem_eff(child)

    fn_recursive_set_mem_eff(model)

  # モデルに xformers とか memory efficient attention を組み込む
  if args.diffusers_xformers:
    print("Use xformers by Diffusers")
    set_diffusers_xformers_flag(unet, True)
  else:
    # Windows版のxformersはfloatで学習できないのでxformersを使わない設定も可能にしておく必要がある
    print("Disable Diffusers' xformers")
    set_diffusers_xformers_flag(unet, False)
    train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers)

  if not fine_tuning:
    # Hypernetwork
    print("import hypernetwork module:", args.hypernetwork_module)
    hyp_module = importlib.import_module(args.hypernetwork_module)

    hypernetwork = hyp_module.Hypernetwork()

    if args.hypernetwork_weights is not None:
      print("load hypernetwork weights from:", args.hypernetwork_weights)
      hyp_sd = torch.load(args.hypernetwork_weights, map_location='cpu')
      success = hypernetwork.load_from_state_dict(hyp_sd)
      assert success, "hypernetwork weights loading failed."

    print("apply hypernetwork")
    hypernetwork.apply_to_diffusers(None, text_encoder, unet)

  # 学習を準備する：モデルを適切な状態にする
  training_models = []
  if fine_tuning:
    if args.gradient_checkpointing:
      unet.enable_gradient_checkpointing()
    training_models.append(unet)

    if args.train_text_encoder:
      print("enable text encoder training")
      if args.gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()
      training_models.append(text_encoder)
    else:
      text_encoder.to(accelerator.device, dtype=weight_dtype)
      text_encoder.requires_grad_(False)             # text encoderは学習しない
      text_encoder.eval()
  else:
    unet.to(accelerator.device)  # , dtype=weight_dtype)     # dtypeを指定すると学習できない
    unet.requires_grad_(False)
    unet.eval()
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    training_models.append(hypernetwork)

  for m in training_models:
    m.requires_grad_(True)
  params = []
  for m in training_models:
    params.extend(m.parameters())
  params_to_optimize = params

  # 学習に必要なクラスを準備する
  print("prepare optimizer, data loader etc.")

  # 8-bit Adamを使う
  if args.use_8bit_adam:
    try:
      import bitsandbytes as bnb
    except ImportError:
      raise ImportError("No bitsand bytes / bitsandbytesがインストールされていないようです")
    print("use 8-bit Adam optimizer")
    optimizer_class = bnb.optim.AdamW8bit
  else:
    optimizer_class = torch.optim.AdamW

  # betaやweight decayはdiffusers DreamBoothもDreamBooth SDもデフォルト値のようなのでオプションはとりあえず省略
  optimizer = optimizer_class(params_to_optimize, lr=args.learning_rate)

  # dataloaderを準備する
  # DataLoaderのプロセス数：0はメインプロセスになる
  n_workers = min(8, os.cpu_count() - 1)      # cpu_count-1 ただし最大8
  train_dataloader = torch.utils.data.DataLoader(
      train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=n_workers)

  # lr schedulerを用意する
  lr_scheduler = diffusers.optimization.get_scheduler(
      args.lr_scheduler, optimizer, num_warmup_steps=args.lr_warmup_steps, num_training_steps=args.max_train_steps * args.gradient_accumulation_steps)

  # acceleratorがなんかよろしくやってくれるらしい
  if args.full_fp16:
    assert args.mixed_precision == "fp16", "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
    print("enable full fp16 training.")

  if fine_tuning:
    # 実験的機能：勾配も含めたfp16学習を行う　モデル全体をfp16にする
    if args.full_fp16:
      unet.to(weight_dtype)
      text_encoder.to(weight_dtype)

    if args.train_text_encoder:
      unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
          unet, text_encoder, optimizer, train_dataloader, lr_scheduler)
    else:
      unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)
  else:
    if args.full_fp16:
      unet.to(weight_dtype)
      hypernetwork.to(weight_dtype)

    unet, hypernetwork, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, hypernetwork, optimizer, train_dataloader, lr_scheduler)

  # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
  if args.full_fp16:
    org_unscale_grads = accelerator.scaler._unscale_grads_

    def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
      return org_unscale_grads(optimizer, inv_scale, found_inf, True)

    accelerator.scaler._unscale_grads_ = _unscale_grads_replacer

  # TODO accelerateのconfigに指定した型とオプション指定の型とをチェックして異なれば警告を出す

  # resumeする
  if args.resume is not None:
    print(f"resume training from state: {args.resume}")
    accelerator.load_state(args.resume)

  # epoch数を計算する
  num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
  num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

  # 学習する
  total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
  print("running training / 学習開始")
  print(f"  num examples / サンプル数: {train_dataset.images_count}")
  print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
  print(f"  num epochs / epoch数: {num_train_epochs}")
  print(f"  batch size per device / バッチサイズ: {args.train_batch_size}")
  print(f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}")
  print(f"  gradient ccumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
  print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

  progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
  global_step = 0

  # v4で更新：clip_sample=Falseに
  # Diffusersのtrain_dreambooth.pyがconfigから持ってくるように変更されたので、clip_sample=Falseになるため、それに合わせる
  # 既存の1.4/1.5/2.0/2.1はすべてschedulerのconfigは（クラス名を除いて）同じ
  # よくソースを見たら学習時はclip_sampleは関係ないや(;'∀')
  noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                  num_train_timesteps=1000, clip_sample=False)

  if accelerator.is_main_process:
    accelerator.init_trackers("finetuning" if fine_tuning else "hypernetwork")

  # 以下 train_dreambooth.py からほぼコピペ
  for epoch in range(num_train_epochs):
    print(f"epoch {epoch+1}/{num_train_epochs}")
    for m in training_models:
      m.train()

    loss_total = 0
    for step, batch in enumerate(train_dataloader):
      with accelerator.accumulate(training_models[0]):  # 複数モデルに対応していない模様だがとりあえずこうしておく
        latents = batch["latents"].to(accelerator.device)
        latents = latents * 0.18215
        b_size = latents.shape[0]

        # with torch.no_grad():
        with torch.set_grad_enabled(args.train_text_encoder):
          # Get the text embedding for conditioning
          input_ids = batch["input_ids"].to(accelerator.device)
          input_ids = input_ids.reshape((-1, tokenizer.model_max_length))     # batch_size*3, 77

          if args.clip_skip is None:
            encoder_hidden_states = text_encoder(input_ids)[0]
          else:
            enc_out = text_encoder(input_ids, output_hidden_states=True, return_dict=True)
            encoder_hidden_states = enc_out['hidden_states'][-args.clip_skip]
            encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)

          # bs*3, 77, 768 or 1024
          encoder_hidden_states = encoder_hidden_states.reshape((b_size, -1, encoder_hidden_states.shape[-1]))

          if args.max_token_length is not None:
            if args.v2:
              # v2: <BOS>...<EOS> <PAD> ... の三連を <BOS>...<EOS> <PAD> ... へ戻す　正直この実装でいいのかわからん
              states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]                              # <BOS>
              for i in range(1, args.max_token_length, tokenizer.model_max_length):
                chunk = encoder_hidden_states[:, i:i + tokenizer.model_max_length - 2]              # <BOS> の後から 最後の前まで
                if i > 0:
                  for j in range(len(chunk)):
                    if input_ids[j, 1] == tokenizer.eos_token:                                      # 空、つまり <BOS> <EOS> <PAD> ...のパターン
                      chunk[j, 0] = chunk[j, 1]                                                     # 次の <PAD> の値をコピーする
                states_list.append(chunk)  # <BOS> の後から <EOS> の前まで
              states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))                         # <EOS> か <PAD> のどちらか
              encoder_hidden_states = torch.cat(states_list, dim=1)
            else:
              # v1: <BOS>...<EOS> の三連を <BOS>...<EOS> へ戻す
              states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]                              # <BOS>
              for i in range(1, args.max_token_length, tokenizer.model_max_length):
                states_list.append(encoder_hidden_states[:, i:i + tokenizer.model_max_length - 2])  # <BOS> の後から <EOS> の前まで
              states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))                         # <EOS>
              encoder_hidden_states = torch.cat(states_list, dim=1)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents, device=latents.device)

        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b_size,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict the noise residual
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        if args.v_parameterization:
          # v-parameterization training
          # Diffusers 0.10.0からv_parameterizationの学習に対応したのでそちらを使う
          target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
          target = noise

        loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

        accelerator.backward(loss)
        if accelerator.sync_gradients:
          params_to_clip = []
          for m in training_models:
            params_to_clip.extend(m.parameters())
          accelerator.clip_grad_norm_(params_to_clip, 1.0)  # args.max_grad_norm)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

      # Checks if the accelerator has performed an optimization step behind the scenes
      if accelerator.sync_gradients:
        progress_bar.update(1)
        global_step += 1

      current_loss = loss.detach().item()        # 平均なのでbatch sizeは関係ないはず
      if args.logging_dir is not None:
        logs = {"loss": current_loss, "lr": lr_scheduler.get_last_lr()[0]}
        accelerator.log(logs, step=global_step)

      loss_total += current_loss
      avr_loss = loss_total / (step+1)
      logs = {"loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
      progress_bar.set_postfix(**logs)

      if global_step >= args.max_train_steps:
        break

    if args.logging_dir is not None:
      logs = {"epoch_loss": loss_total / len(train_dataloader)}
      accelerator.log(logs, step=epoch+1)

    accelerator.wait_for_everyone()

    if args.save_every_n_epochs is not None:
      if (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs:
        print("saving checkpoint.")
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, model_util.get_epoch_ckpt_name(use_safetensors, epoch + 1))

        if fine_tuning:
          if save_stable_diffusion_format:
            model_util.save_stable_diffusion_checkpoint(args.v2, ckpt_file, unwrap_model(text_encoder), unwrap_model(unet),
                                                        src_stable_diffusion_ckpt, epoch + 1, global_step, save_dtype, vae)
          else:
            out_dir = os.path.join(args.output_dir, train_util.EPOCH_DIFFUSERS_DIR_NAME.format(epoch + 1))
            os.makedirs(out_dir, exist_ok=True)
            model_util.save_diffusers_checkpoint(args.v2, out_dir, unwrap_model(text_encoder), unwrap_model(unet),
                                                 src_diffusers_model_path, vae=vae, use_safetensors=use_safetensors)
        else:
          save_hypernetwork(ckpt_file, unwrap_model(hypernetwork))

        if args.save_state:
          print("saving state.")
          accelerator.save_state(os.path.join(args.output_dir, train_util.EPOCH_STATE_NAME.format(epoch + 1)))

  is_main_process = accelerator.is_main_process
  if is_main_process:
    if fine_tuning:
      unet = unwrap_model(unet)
      text_encoder = unwrap_model(text_encoder)
    else:
      hypernetwork = unwrap_model(hypernetwork)

  accelerator.end_training()

  if args.save_state:
    print("saving last state.")
    accelerator.save_state(os.path.join(args.output_dir, train_util.LAST_STATE_NAME))

  del accelerator                         # この後メモリを使うのでこれは消す

  if is_main_process:
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_file = os.path.join(args.output_dir, model_util.get_last_ckpt_name(use_safetensors))

    if fine_tuning:
      if save_stable_diffusion_format:
        print(f"save trained model as StableDiffusion checkpoint to {ckpt_file}")
        model_util.save_stable_diffusion_checkpoint(args.v2, ckpt_file, text_encoder, unet,
                                                    src_stable_diffusion_ckpt, epoch, global_step, save_dtype, vae)
      else:
        # Create the pipeline using using the trained modules and save it.
        print(f"save trained model as Diffusers to {args.output_dir}")
        out_dir = os.path.join(args.output_dir, train_util.LAST_DIFFUSERS_DIR_NAME)
        os.makedirs(out_dir, exist_ok=True)
        model_util.save_diffusers_checkpoint(args.v2, out_dir, text_encoder, unet,
                                             src_diffusers_model_path, vae=vae, use_safetensors=use_safetensors)
    else:
      print(f"save trained model to {ckpt_file}")
      save_hypernetwork(ckpt_file, hypernetwork)

    print("model saved.")


if __name__ == '__main__':
  # torch.cuda.set_per_process_memory_fraction(0.48)
  parser = argparse.ArgumentParser()
  parser.add_argument("--v2", action='store_true',
                      help='load Stable Diffusion v2.0 model / Stable Diffusion 2.0のモデルを読み込む')
  parser.add_argument("--v_parameterization", action='store_true',
                      help='enable v-parameterization training / v-parameterization学習を有効にする')
  parser.add_argument("--pretrained_model_name_or_path", type=str, default=None,
                      help="pretrained model to train, directory to Diffusers model or StableDiffusion checkpoint / 学習元モデル、Diffusers形式モデルのディレクトリまたはStableDiffusionのckptファイル")
  parser.add_argument("--in_json", type=str, default=None, help="metadata file to input / 読みこむメタデータファイル")
  parser.add_argument("--shuffle_caption", action="store_true",
                      help="shuffle comma-separated caption when fine tuning / fine tuning時にコンマで区切られたcaptionの各要素をshuffleする")
  parser.add_argument("--keep_tokens", type=int, default=None,
                      help="keep heading N tokens when shuffling caption tokens / captionのシャッフル時に、先頭からこの個数のトークンをシャッフルしないで残す")
  parser.add_argument("--train_data_dir", type=str, default=None, help="directory for train images / 学習画像データのディレクトリ")
  parser.add_argument("--dataset_repeats", type=int, default=None, help="num times to repeat dataset / 学習にデータセットを繰り返す回数")
  parser.add_argument("--output_dir", type=str, default=None,
                      help="directory to output trained model, save as same format as input / 学習後のモデル出力先ディレクトリ（入力と同じ形式で保存）")
  parser.add_argument("--save_precision", type=str, default=None,
                      choices=[None, "float", "fp16", "bf16"], help="precision in saving (available in StableDiffusion checkpoint) / 保存時に精度を変更して保存する（StableDiffusion形式での保存時のみ有効）")
  parser.add_argument("--save_model_as", type=str, default=None, choices=[None, "ckpt", "safetensors", "diffusers", "diffusers_safetensors"],
                      help="format to save the model (default is same to original) / モデル保存時の形式（未指定時は元モデルと同じ）")
  parser.add_argument("--use_safetensors", action='store_true',
                      help="use safetensors format to save (if save_model_as is not specified) / checkpoint、モデルをsafetensors形式で保存する（save_model_as未指定時）")
  parser.add_argument("--train_text_encoder", action="store_true", help="train text encoder / text encoderも学習する")
  parser.add_argument("--hypernetwork_module", type=str, default=None,
                      help='train hypernetwork instead of fine tuning, module to use / fine tuningの代わりにHypernetworkの学習をする場合、そのモジュール')
  parser.add_argument("--hypernetwork_weights", type=str, default=None,
                      help='hypernetwork weights to initialize for additional training / Hypernetworkの学習時に読み込む重み（Hypernetworkの追加学習）')
  parser.add_argument("--save_every_n_epochs", type=int, default=None,
                      help="save checkpoint every N epochs / 学習中のモデルを指定エポックごとに保存する")
  parser.add_argument("--save_state", action="store_true",
                      help="save training state additionally (including optimizer states etc.) / optimizerなど学習状態も含めたstateを追加で保存する")
  parser.add_argument("--resume", type=str, default=None,
                      help="saved state to resume training / 学習再開するモデルのstate")
  parser.add_argument("--max_token_length", type=int, default=None, choices=[None, 150, 225],
                      help="max token length of text encoder (default for 75, 150 or 225) / text encoderのトークンの最大長（未指定で75、150または225が指定可）")
  parser.add_argument("--train_batch_size", type=int, default=1,
                      help="batch size for training / 学習時のバッチサイズ")
  parser.add_argument("--use_8bit_adam", action="store_true",
                      help="use 8bit Adam optimizer (requires bitsandbytes) / 8bit Adamオプティマイザを使う（bitsandbytesのインストールが必要）")
  parser.add_argument("--mem_eff_attn", action="store_true",
                      help="use memory efficient attention for CrossAttention / CrossAttentionに省メモリ版attentionを使う")
  parser.add_argument("--xformers", action="store_true",
                      help="use xformers for CrossAttention / CrossAttentionにxformersを使う")
  parser.add_argument("--diffusers_xformers", action='store_true',
                      help='use xformers by diffusers (Hypernetworks doesn\'t work) / Diffusersでxformersを使用する（Hypernetwork利用不可）')
  parser.add_argument("--learning_rate", type=float, default=2.0e-6, help="learning rate / 学習率")
  parser.add_argument("--max_train_steps", type=int, default=1600, help="training steps / 学習ステップ数")
  parser.add_argument("--seed", type=int, default=None, help="random seed for training / 学習時の乱数のseed")
  parser.add_argument("--gradient_checkpointing", action="store_true",
                      help="enable gradient checkpointing / grandient checkpointingを有効にする")
  parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                      help="Number of updates steps to accumulate before performing a backward/update pass / 学習時に逆伝播をする前に勾配を合計するステップ数")
  parser.add_argument("--mixed_precision", type=str, default="no",
                      choices=["no", "fp16", "bf16"], help="use mixed precision / 混合精度を使う場合、その精度")
  parser.add_argument("--full_fp16", action="store_true", help="fp16 training including gradients / 勾配も含めてfp16で学習する")
  parser.add_argument("--clip_skip", type=int, default=None,
                      help="use output of nth layer from back of text encoder (n>=1) / text encoderの後ろからn番目の層の出力を用いる（nは1以上）")
  parser.add_argument("--debug_dataset", action="store_true",
                      help="show images for debugging (do not train) / デバッグ用に学習データを画面表示する（学習は行わない）")
  parser.add_argument("--logging_dir", type=str, default=None,
                      help="enable logging and output TensorBoard log to this directory / ログ出力を有効にしてこのディレクトリにTensorBoard用のログを出力する")
  parser.add_argument("--log_prefix", type=str, default=None, help="add prefix for each log directory / ログディレクトリ名の先頭に追加する文字列")
  parser.add_argument("--lr_scheduler", type=str, default="constant",
                      help="scheduler to use for learning rate / 学習率のスケジューラ: linear, cosine, cosine_with_restarts, polynomial, constant (default), constant_with_warmup")
  parser.add_argument("--lr_warmup_steps", type=int, default=0,
                      help="Number of steps for the warmup in the lr scheduler (default is 0) / 学習率のスケジューラをウォームアップするステップ数（デフォルト0）")

  args = parser.parse_args()
  train(args)
