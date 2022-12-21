import argparse
import glob
import os
import json
import random

from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from blip.blip import blip_decoder
# from Salesforce_BLIP.models.blip import blip_decoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
  # fix the seed for reproducibility
  seed = args.seed # + utils.get_rank()
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
    
  if not os.path.exists("blip"):
    args.train_data_dir = os.path.abspath(args.train_data_dir)        # convert to absolute path

    cwd = os.getcwd()
    print('Current Working Directory is: ', cwd)
    os.chdir('finetune')

  print(f"load images from {args.train_data_dir}")
  image_paths = glob.glob(os.path.join(args.train_data_dir, "*.jpg")) + \
      glob.glob(os.path.join(args.train_data_dir, "*.png")) + glob.glob(os.path.join(args.train_data_dir, "*.webp"))
  print(f"found {len(image_paths)} images.")

  print(f"loading BLIP caption: {args.caption_weights}")
  image_size = 384
  model = blip_decoder(pretrained=args.caption_weights, image_size=image_size, vit='large', med_config="./blip/med_config.json")
  model.eval()
  model = model.to(DEVICE)
  print("BLIP loaded")

  # 正方形でいいのか？　という気がするがソースがそうなので
  transform = transforms.Compose([
      transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
      transforms.ToTensor(),
      transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
  ])

  # captioningする
  def run_batch(path_imgs):
    imgs = torch.stack([im for _, im in path_imgs]).to(DEVICE)

    with torch.no_grad():
      if args.beam_search:
        captions = model.generate(imgs, sample=False, num_beams=args.num_beams,
                                  max_length=args.max_length, min_length=args.min_length)
      else:
        captions = model.generate(imgs, sample=True, top_p=args.top_p, max_length=args.max_length, min_length=args.min_length)

    for (image_path, _), caption in zip(path_imgs, captions):
      with open(os.path.splitext(image_path)[0] + args.caption_extension, "wt", encoding='utf-8') as f:
        f.write(caption + "\n")
        if args.debug:
          print(image_path, caption)

  b_imgs = []
  for image_path in tqdm(image_paths, smoothing=0.0):
    raw_image = Image.open(image_path)
    if raw_image.mode != "RGB":
      print(f"convert image mode {raw_image.mode} to RGB: {image_path}")
      raw_image = raw_image.convert("RGB")

    image = transform(raw_image)
    b_imgs.append((image_path, image))
    if len(b_imgs) >= args.batch_size:
      run_batch(b_imgs)
      b_imgs.clear()
  if len(b_imgs) > 0:
    run_batch(b_imgs)

  print("done!")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
  parser.add_argument("--caption_weights", type=str, default="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth",
                      help="BLIP caption weights (model_large_caption.pth) / BLIP captionの重みファイル(model_large_caption.pth)")
  parser.add_argument("--caption_extention", type=str, default=None,
                      help="extension of caption file (for backward compatibility) / 出力されるキャプションファイルの拡張子（スペルミスしていたのを残してあります）")
  parser.add_argument("--caption_extension", type=str, default=".caption", help="extension of caption file / 出力されるキャプションファイルの拡張子")
  parser.add_argument("--beam_search", action="store_true",
                      help="use beam search (default Nucleus sampling) / beam searchを使う（このオプション未指定時はNucleus sampling）")
  parser.add_argument("--batch_size", type=int, default=1, help="batch size in inference / 推論時のバッチサイズ")
  parser.add_argument("--num_beams", type=int, default=1, help="num of beams in beam search /beam search時のビーム数（多いと精度が上がるが時間がかかる）")
  parser.add_argument("--top_p", type=float, default=0.9, help="top_p in Nucleus sampling / Nucleus sampling時のtop_p")
  parser.add_argument("--max_length", type=int, default=75, help="max length of caption / captionの最大長")
  parser.add_argument("--min_length", type=int, default=5, help="min length of caption / captionの最小長")
  parser.add_argument('--seed', default=42, type=int, help='seed for reproducibility / 再現性を確保するための乱数seed')
  parser.add_argument("--debug", action="store_true", help="debug mode")

  args = parser.parse_args()

  # スペルミスしていたオプションを復元する
  if args.caption_extention is not None:
    args.caption_extension = args.caption_extention

  main(args)
