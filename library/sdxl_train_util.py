import argparse
import gc
import math
import os
from types import SimpleNamespace
from typing import Any
import torch
from tqdm import tqdm
from transformers import CLIPTokenizer
import open_clip
from diffusers import StableDiffusionXLPipeline
from library import model_util, sdxl_model_util, train_util, sdxl_original_unet
from library.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline

TOKENIZER_PATH = "openai/clip-vit-large-patch14"

DEFAULT_NOISE_OFFSET = 0.0357


# TODO: separate checkpoint for each U-Net/Text Encoder/VAE
def load_target_model(args, accelerator, model_version: str, weight_dtype):
    # load models for each process
    for pi in range(accelerator.state.num_processes):
        if pi == accelerator.state.local_process_index:
            print(f"loading model for process {accelerator.state.local_process_index}/{accelerator.state.num_processes}")

            (
                load_stable_diffusion_format,
                text_encoder1,
                text_encoder2,
                vae,
                unet,
                logit_scale,
                ckpt_info,
            ) = _load_target_model(args, model_version, weight_dtype, accelerator.device if args.lowram else "cpu")

            # work on low-ram device
            if args.lowram:
                text_encoder1.to(accelerator.device)
                text_encoder2.to(accelerator.device)
                unet.to(accelerator.device)
                vae.to(accelerator.device)

            gc.collect()
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    text_encoder1, text_encoder2, unet = train_util.transform_models_if_DDP([text_encoder1, text_encoder2, unet])

    return load_stable_diffusion_format, text_encoder1, text_encoder2, vae, unet, logit_scale, ckpt_info


def _load_target_model(args: argparse.Namespace, model_version: str, weight_dtype, device="cpu"):
    name_or_path = args.pretrained_model_name_or_path
    name_or_path = os.readlink(name_or_path) if os.path.islink(name_or_path) else name_or_path
    load_stable_diffusion_format = os.path.isfile(name_or_path)  # determine SD or Diffusers

    if load_stable_diffusion_format:
        print(f"load StableDiffusion checkpoint: {name_or_path}")
        (
            text_encoder1,
            text_encoder2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = sdxl_model_util.load_models_from_sdxl_checkpoint(model_version, name_or_path, device)
    else:
        # Diffusers model is loaded to CPU
        variant = "fp16" if weight_dtype == torch.float16 else None
        print(f"load Diffusers pretrained models: {name_or_path}, variant={variant}")
        try:
            try:
                pipe = StableDiffusionXLPipeline.from_pretrained(name_or_path, variant=variant, tokenizer=None)
            except EnvironmentError as ex:
                if variant is not None:
                    print("try to load fp32 model")
                    pipe = StableDiffusionXLPipeline.from_pretrained(name_or_path, variant=None, tokenizer=None)
                else:
                    raise ex
        except EnvironmentError as ex:
            print(
                f"model is not found as a file or in Hugging Face, perhaps file name is wrong? / 指定したモデル名のファイル、またはHugging Faceのモデルが見つかりません。ファイル名が誤っているかもしれません: {name_or_path}"
            )
            raise ex

        text_encoder1 = pipe.text_encoder
        text_encoder2 = pipe.text_encoder_2
        vae = pipe.vae
        unet = pipe.unet
        del pipe

        # Diffusers U-Net to original U-Net
        original_unet = sdxl_original_unet.SdxlUNet2DConditionModel()
        state_dict = sdxl_model_util.convert_diffusers_unet_state_dict_to_sdxl(unet.state_dict())
        original_unet.load_state_dict(state_dict)
        unet = original_unet
        print("U-Net converted to original U-Net")

        logit_scale = None
        ckpt_info = None

    # VAEを読み込む
    if args.vae is not None:
        vae = model_util.load_vae(args.vae, weight_dtype)
        print("additional VAE loaded")

    return load_stable_diffusion_format, text_encoder1, text_encoder2, vae, unet, logit_scale, ckpt_info


class WrapperTokenizer:
    # open clipのtokenizerをHuggingFaceのtokenizerと同じ形で使えるようにする
    # make open clip tokenizer compatible with HuggingFace tokenizer
    def __init__(self):
        open_clip_tokenizer = open_clip.tokenizer._tokenizer
        self.model_max_length = 77
        self.bos_token_id = open_clip_tokenizer.all_special_ids[0]
        self.eos_token_id = open_clip_tokenizer.all_special_ids[1]
        self.pad_token_id = 0  # 結果から推定している assumption from result

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.tokenize(*args, **kwds)

    def tokenize(self, text, padding=False, truncation=None, max_length=None, return_tensors=None):
        if padding == "max_length":
            # for training
            assert max_length is not None
            assert truncation == True
            assert return_tensors == "pt"
            input_ids = open_clip.tokenize(text, context_length=max_length)
            return SimpleNamespace(**{"input_ids": input_ids})

        # for weighted prompt
        assert isinstance(text, str), f"input must be str: {text}"

        input_ids = open_clip.tokenize(text, context_length=self.model_max_length)[0]  # tokenizer returns list

        # find eos
        eos_index = (input_ids == self.eos_token_id).nonzero().max()
        input_ids = input_ids[: eos_index + 1]  # include eos
        return SimpleNamespace(**{"input_ids": input_ids})

    # for Textual Inversion
    # わりと面倒くさいな……これWeb UIとかでどうするんだろう / this is a bit annoying... how to do this in Web UI?

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        input_ids = open_clip.tokenizer._tokenizer.encode(text)
        return input_ids

    def add_tokens(self, new_tokens):
        tokens_to_add = []
        for token in new_tokens:
            token = token.lower()
            if token + "</w>" not in open_clip.tokenizer._tokenizer.encoder:
                tokens_to_add.append(token)

        # open clipのtokenizerに直接追加する / add tokens to open clip tokenizer
        for token in tokens_to_add:
            open_clip.tokenizer._tokenizer.encoder[token + "</w>"] = len(open_clip.tokenizer._tokenizer.encoder)
            open_clip.tokenizer._tokenizer.decoder[len(open_clip.tokenizer._tokenizer.decoder)] = token + "</w>"
            open_clip.tokenizer._tokenizer.vocab_size += 1

            # open clipのtokenizerのcacheに直接設定することで、bpeとかいうやつに含まれていなくてもtokenizeできるようにする
            # めちゃくちゃ乱暴なので、open clipのtokenizerの仕様が変わったら動かなくなる
            # set cache of open clip tokenizer directly to enable tokenization even if the token is not included in bpe
            # this is very rough, so it will not work if the specification of open clip tokenizer changes
            open_clip.tokenizer._tokenizer.cache[token] = token + "</w>"

        return len(tokens_to_add)

    def convert_tokens_to_ids(self, tokens):
        input_ids = [open_clip.tokenizer._tokenizer.encoder[token + "</w>"] for token in tokens]
        return input_ids

    def __len__(self):
        return open_clip.tokenizer._tokenizer.vocab_size


def load_tokenizers(args: argparse.Namespace):
    print("prepare tokenizers")
    original_path = TOKENIZER_PATH

    tokenizer1: CLIPTokenizer = None
    if args.tokenizer_cache_dir:
        local_tokenizer_path = os.path.join(args.tokenizer_cache_dir, original_path.replace("/", "_"))
        if os.path.exists(local_tokenizer_path):
            print(f"load tokenizer from cache: {local_tokenizer_path}")
            tokenizer1 = CLIPTokenizer.from_pretrained(local_tokenizer_path)

    if tokenizer1 is None:
        tokenizer1 = CLIPTokenizer.from_pretrained(original_path)

    if args.tokenizer_cache_dir and not os.path.exists(local_tokenizer_path):
        print(f"save Tokenizer to cache: {local_tokenizer_path}")
        tokenizer1.save_pretrained(local_tokenizer_path)

    if hasattr(args, "max_token_length") and args.max_token_length is not None:
        print(f"update token length: {args.max_token_length}")

    # tokenizer2 is from open_clip
    # TODO caching
    tokenizer2 = WrapperTokenizer()

    return [tokenizer1, tokenizer2]


def get_hidden_states(
    args: argparse.Namespace, input_ids1, input_ids2, tokenizer1, tokenizer2, text_encoder1, text_encoder2, weight_dtype=None
):
    # input_ids: b,n,77 -> b*n, 77
    b_size = input_ids1.size()[0]
    input_ids1 = input_ids1.reshape((-1, tokenizer1.model_max_length))  # batch_size*n, 77
    input_ids2 = input_ids2.reshape((-1, tokenizer2.model_max_length))  # batch_size*n, 77

    # text_encoder1
    enc_out = text_encoder1(input_ids1, output_hidden_states=True, return_dict=True)
    hidden_states1 = enc_out["hidden_states"][11]

    # text_encoder2
    enc_out = text_encoder2(input_ids2, output_hidden_states=True, return_dict=True)
    hidden_states2 = enc_out["hidden_states"][-2]  # penuultimate layer
    pool2 = enc_out["text_embeds"]

    # b*n, 77, 768 or 1280 -> b, n*77, 768 or 1280
    n_size = 1 if args.max_token_length is None else args.max_token_length // 75
    hidden_states1 = hidden_states1.reshape((b_size, -1, hidden_states1.shape[-1]))
    hidden_states2 = hidden_states2.reshape((b_size, -1, hidden_states2.shape[-1]))

    if args.max_token_length is not None:
        # bs*3, 77, 768 or 1024
        # encoder1: <BOS>...<EOS> の三連を <BOS>...<EOS> へ戻す
        states_list = [hidden_states1[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, args.max_token_length, tokenizer1.model_max_length):
            states_list.append(hidden_states1[:, i : i + tokenizer1.model_max_length - 2])  # <BOS> の後から <EOS> の前まで
        states_list.append(hidden_states1[:, -1].unsqueeze(1))  # <EOS>
        hidden_states1 = torch.cat(states_list, dim=1)

        # v2: <BOS>...<EOS> <PAD> ... の三連を <BOS>...<EOS> <PAD> ... へ戻す　正直この実装でいいのかわからん
        states_list = [hidden_states2[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, args.max_token_length, tokenizer2.model_max_length):
            chunk = hidden_states2[:, i : i + tokenizer2.model_max_length - 2]  # <BOS> の後から 最後の前まで
            # this causes an error:
            # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
            # if i > 1:
            #     for j in range(len(chunk)):  # batch_size
            #         if input_ids2[n_index + j * n_size, 1] == tokenizer2.eos_token_id:  # 空、つまり <BOS> <EOS> <PAD> ...のパターン
            #             chunk[j, 0] = chunk[j, 1]  # 次の <PAD> の値をコピーする
            states_list.append(chunk)  # <BOS> の後から <EOS> の前まで
        states_list.append(hidden_states2[:, -1].unsqueeze(1))  # <EOS> か <PAD> のどちらか
        hidden_states2 = torch.cat(states_list, dim=1)

        # pool はnの最初のものを使う
        pool2 = pool2[::n_size]

    if weight_dtype is not None:
        # this is required for additional network training
        hidden_states1 = hidden_states1.to(weight_dtype)
        hidden_states2 = hidden_states2.to(weight_dtype)

    return hidden_states1, hidden_states2, pool2


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def get_timestep_embedding(x, outdim):
    assert len(x.shape) == 2
    b, dims = x.shape[0], x.shape[1]
    x = torch.flatten(x)
    emb = timestep_embedding(x, outdim)
    emb = torch.reshape(emb, (b, dims * outdim))
    return emb


def get_size_embeddings(orig_size, crop_size, target_size, device):
    emb1 = get_timestep_embedding(orig_size, 256)
    emb2 = get_timestep_embedding(crop_size, 256)
    emb3 = get_timestep_embedding(target_size, 256)
    vector = torch.cat([emb1, emb2, emb3], dim=1).to(device)
    return vector


def save_sd_model_on_train_end(
    args: argparse.Namespace,
    src_path: str,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    save_dtype: torch.dtype,
    epoch: int,
    global_step: int,
    text_encoder1,
    text_encoder2,
    unet,
    vae,
    logit_scale,
    ckpt_info,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sdxl_model_util.save_stable_diffusion_checkpoint(
            ckpt_file,
            text_encoder1,
            text_encoder2,
            unet,
            epoch_no,
            global_step,
            ckpt_info,
            vae,
            logit_scale,
            save_dtype,
        )

    def diffusers_saver(out_dir):
        sdxl_model_util.save_diffusers_checkpoint(
            out_dir,
            text_encoder1,
            text_encoder2,
            unet,
            src_path,
            vae,
            use_safetensors=use_safetensors,
            save_dtype=save_dtype,
        )

    train_util.save_sd_model_on_train_end_common(
        args, save_stable_diffusion_format, use_safetensors, epoch, global_step, sd_saver, diffusers_saver
    )


# epochとstepの保存、メタデータにepoch/stepが含まれ引数が同じになるため、統合している
# on_epoch_end: Trueならepoch終了時、Falseならstep経過時
def save_sd_model_on_epoch_end_or_stepwise(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator,
    src_path,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    save_dtype: torch.dtype,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    text_encoder1,
    text_encoder2,
    unet,
    vae,
    logit_scale,
    ckpt_info,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sdxl_model_util.save_stable_diffusion_checkpoint(
            ckpt_file,
            text_encoder1,
            text_encoder2,
            unet,
            epoch_no,
            global_step,
            ckpt_info,
            vae,
            logit_scale,
            save_dtype,
        )

    def diffusers_saver(out_dir):
        sdxl_model_util.save_diffusers_checkpoint(
            out_dir,
            text_encoder1,
            text_encoder2,
            unet,
            src_path,
            vae,
            use_safetensors=use_safetensors,
            save_dtype=save_dtype,
        )

    train_util.save_sd_model_on_epoch_end_or_stepwise_common(
        args,
        on_epoch_end,
        accelerator,
        save_stable_diffusion_format,
        use_safetensors,
        epoch,
        num_train_epochs,
        global_step,
        sd_saver,
        diffusers_saver,
    )


# TextEncoderの出力をキャッシュする
# weight_dtypeを指定するとText Encoderそのもの、およひ出力がweight_dtypeになる
def cache_text_encoder_outputs(args, accelerator, tokenizers, text_encoders, dataset, weight_dtype):
    print("caching text encoder outputs")

    tokenizer1, tokenizer2 = tokenizers
    text_encoder1, text_encoder2 = text_encoders
    text_encoder1.to(accelerator.device)
    text_encoder2.to(accelerator.device)
    if weight_dtype is not None:
        text_encoder1.to(dtype=weight_dtype)
        text_encoder2.to(dtype=weight_dtype)

    text_encoder1_cache = {}
    text_encoder2_cache = {}
    for batch in tqdm(dataset):
        input_ids1_batch = batch["input_ids"].to(accelerator.device)
        input_ids2_batch = batch["input_ids2"].to(accelerator.device)

        # split batch to avoid OOM
        # TODO specify batch size by args
        for input_id1, input_id2 in zip(input_ids1_batch.split(1), input_ids2_batch.split(1)):
            # remove input_ids already in cache
            input_id1_cache_key = tuple(input_id1.flatten().tolist())
            input_id2_cache_key = tuple(input_id2.flatten().tolist())
            if input_id1_cache_key in text_encoder1_cache:
                assert input_id2_cache_key in text_encoder2_cache
                continue

            with torch.no_grad():
                encoder_hidden_states1, encoder_hidden_states2, pool2 = get_hidden_states(
                    args,
                    input_id1,
                    input_id2,
                    tokenizer1,
                    tokenizer2,
                    text_encoder1,
                    text_encoder2,
                    None if not args.full_fp16 else weight_dtype,
                )
            encoder_hidden_states1 = encoder_hidden_states1.detach().to("cpu").squeeze(0)  # n*75+2,768
            encoder_hidden_states2 = encoder_hidden_states2.detach().to("cpu").squeeze(0)  # n*75+2,1280
            pool2 = pool2.detach().to("cpu").squeeze(0)  # 1280
            text_encoder1_cache[input_id1_cache_key] = encoder_hidden_states1
            text_encoder2_cache[input_id2_cache_key] = (encoder_hidden_states2, pool2)
    return text_encoder1_cache, text_encoder2_cache


def add_sdxl_training_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--cache_text_encoder_outputs", action="store_true", help="cache text encoder outputs / text encoderの出力をキャッシュする"
    )


def verify_sdxl_training_args(args: argparse.Namespace):
    assert (
        not args.v2 and not args.v_parameterization
    ), "v2 or v_parameterization cannot be enabled in SDXL training / SDXL学習ではv2とv_parameterizationを有効にすることはできません"
    if args.clip_skip is not None:
        print("clip_skip will be unexpected / SDXL学習ではclip_skipは動作しません")

    if args.multires_noise_iterations:
        print(
            f"Warning: SDXL has been trained with noise_offset={DEFAULT_NOISE_OFFSET}, but noise_offset is disabled due to multires_noise_iterations / SDXLはnoise_offset={DEFAULT_NOISE_OFFSET}で学習されていますが、multires_noise_iterationsが有効になっているためnoise_offsetは無効になります"
        )
    else:
        if args.noise_offset is None:
            args.noise_offset = DEFAULT_NOISE_OFFSET
        elif args.noise_offset != DEFAULT_NOISE_OFFSET:
            print(
                f"Warning: SDXL has been trained with noise_offset={DEFAULT_NOISE_OFFSET} / SDXLはnoise_offset={DEFAULT_NOISE_OFFSET}で学習されています"
            )
        print(f"noise_offset is set to {args.noise_offset} / noise_offsetが{args.noise_offset}に設定されました")

    assert (
        not hasattr(args, "weighted_captions") or not args.weighted_captions
    ), "weighted_captions cannot be enabled in SDXL training currently / SDXL学習では今のところweighted_captionsを有効にすることはできません"


def sample_images(*args, **kwargs):
    return train_util.sample_images_common(SdxlStableDiffusionLongPromptWeightingPipeline, *args, **kwargs)
