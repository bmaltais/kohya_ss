# convert Diffusers v1.x/v2.0 model to original Stable Diffusion

import argparse
import os
import torch
from diffusers import StableDiffusionPipeline

import library.model_util as model_util


def convert(args):
  # 引数を確認する
  load_dtype = torch.float16 if args.fp16 else None

  save_dtype = None
  if args.fp16:
    save_dtype = torch.float16
  elif args.bf16:
    save_dtype = torch.bfloat16
  elif args.float:
    save_dtype = torch.float

  is_load_ckpt = os.path.isfile(args.model_to_load)
  is_save_ckpt = len(os.path.splitext(args.model_to_save)[1]) > 0

  assert not is_load_ckpt or args.v1 != args.v2, f"v1 or v2 is required to load checkpoint / checkpointの読み込みにはv1/v2指定が必要です"
  assert is_save_ckpt or args.reference_model is not None, f"reference model is required to save as Diffusers / Diffusers形式での保存には参照モデルが必要です"

  # モデルを読み込む
  msg = "checkpoint" if is_load_ckpt else ("Diffusers" + (" as fp16" if args.fp16 else ""))
  print(f"loading {msg}: {args.model_to_load}")

  if is_load_ckpt:
    v2_model = args.v2
    text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(v2_model, args.model_to_load)
  else:
    pipe = StableDiffusionPipeline.from_pretrained(args.model_to_load, torch_dtype=load_dtype, tokenizer=None, safety_checker=None)
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet

    if args.v1 == args.v2:
      # 自動判定する
      v2_model = unet.config.cross_attention_dim == 1024
      print("checking model version: model is " + ('v2' if v2_model else 'v1'))
    else:
      v2_model = not args.v1

  # 変換して保存する
  msg = ("checkpoint" + ("" if save_dtype is None else f" in {save_dtype}")) if is_save_ckpt else "Diffusers"
  print(f"converting and saving as {msg}: {args.model_to_save}")

  if is_save_ckpt:
    original_model = args.model_to_load if is_load_ckpt else None
    key_count = model_util.save_stable_diffusion_checkpoint(v2_model, args.model_to_save, text_encoder, unet,
                                                            original_model, args.epoch, args.global_step, save_dtype, vae)
    print(f"model saved. total converted state_dict keys: {key_count}")
  else:
    print(f"copy scheduler/tokenizer config from: {args.reference_model}")
    model_util.save_diffusers_checkpoint(v2_model, args.model_to_save, text_encoder, unet, args.reference_model, vae, args.use_safetensors)
    print(f"model saved.")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--v1", action='store_true',
                      help='load v1.x model (v1 or v2 is required to load checkpoint) / 1.xのモデルを読み込む')
  parser.add_argument("--v2", action='store_true',
                      help='load v2.0 model (v1 or v2 is required to load checkpoint) / 2.0のモデルを読み込む')
  parser.add_argument("--fp16", action='store_true',
                      help='load as fp16 (Diffusers only) and save as fp16 (checkpoint only) / fp16形式で読み込み（Diffusers形式のみ対応）、保存する（checkpointのみ対応）')
  parser.add_argument("--bf16", action='store_true', help='save as bf16 (checkpoint only) / bf16形式で保存する（checkpointのみ対応）')
  parser.add_argument("--float", action='store_true',
                      help='save as float (checkpoint only) / float(float32)形式で保存する（checkpointのみ対応）')
  parser.add_argument("--epoch", type=int, default=0, help='epoch to write to checkpoint / checkpointに記録するepoch数の値')
  parser.add_argument("--global_step", type=int, default=0,
                      help='global_step to write to checkpoint / checkpointに記録するglobal_stepの値')
  parser.add_argument("--reference_model", type=str, default=None,
                      help="reference model for schduler/tokenizer, required in saving Diffusers, copy schduler/tokenizer from this / scheduler/tokenizerのコピー元のDiffusersモデル、Diffusers形式で保存するときに必要")
  parser.add_argument("--use_safetensors", action='store_true',
                      help="use safetensors format to save Diffusers model (checkpoint depends on the file extension) / Duffusersモデルをsafetensors形式で保存する（checkpointは拡張子で自動判定）")

  parser.add_argument("model_to_load", type=str, default=None,
                      help="model to load: checkpoint file or Diffusers model's directory / 読み込むモデル、checkpointかDiffusers形式モデルのディレクトリ")
  parser.add_argument("model_to_save", type=str, default=None,
                      help="model to save: checkpoint (with extension) or Diffusers model's directory (without extension) / 変換後のモデル、拡張子がある場合はcheckpoint、ない場合はDiffusesモデルとして保存")

  args = parser.parse_args()
  convert(args)
