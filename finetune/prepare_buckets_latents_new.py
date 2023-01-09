# このスクリプトのライセンスは、Apache License 2.0とします
# (c) 2022 Kohya S. @kohya_ss

import argparse
import glob
import os
import json

from tqdm import tqdm
import numpy as np
from diffusers import AutoencoderKL
from PIL import Image
import cv2
import torch
from torchvision import transforms

import library.model_util as model_util

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def get_latents(vae, images, weight_dtype):
  img_tensors = [IMAGE_TRANSFORMS(image) for image in images]
  img_tensors = torch.stack(img_tensors)
  img_tensors = img_tensors.to(DEVICE, weight_dtype)
  with torch.no_grad():
    latents = vae.encode(img_tensors).latent_dist.sample().float().to("cpu").numpy()
  return latents


def main(args):
  # image_paths = glob.glob(os.path.join(args.train_data_dir, "*.jpg")) + \
  #     glob.glob(os.path.join(args.train_data_dir, "*.png")) + glob.glob(os.path.join(args.train_data_dir, "*.webp"))
  # print(f"found {len(image_paths)} images.")
  

  if os.path.exists(args.in_json):
    print(f"loading existing metadata: {args.in_json}")
    with open(args.in_json, "rt", encoding='utf-8') as f:
      metadata = json.load(f)
  else:
    print(f"no metadata / メタデータファイルがありません: {args.in_json}")
    return

  weight_dtype = torch.float32
  if args.mixed_precision == "fp16":
    weight_dtype = torch.float16
  elif args.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

  vae = model_util.load_vae(args.model_name_or_path, weight_dtype)
  vae.eval()
  vae.to(DEVICE, dtype=weight_dtype)

  # bucketのサイズを計算する
  max_reso = tuple([int(t) for t in args.max_resolution.split(',')])
  assert len(max_reso) == 2, f"illegal resolution (not 'width,height') / 画像サイズに誤りがあります。'幅,高さ'で指定してください: {args.max_resolution}"

  bucket_resos, bucket_aspect_ratios = model_util.make_bucket_resolutions(
      max_reso, args.min_bucket_reso, args.max_bucket_reso)

  # 画像をひとつずつ適切なbucketに割り当てながらlatentを計算する
  bucket_aspect_ratios = np.array(bucket_aspect_ratios)
  buckets_imgs = [[] for _ in range(len(bucket_resos))]
  bucket_counts = [0 for _ in range(len(bucket_resos))]
  img_ar_errors = []
  for i, image_path in enumerate(tqdm(metadata, smoothing=0.0)):
    image_key = image_path
    if image_key not in metadata:
      metadata[image_key] = {}

    image = Image.open(image_path)
    if image.mode != 'RGB':
      image = image.convert("RGB")

    aspect_ratio = image.width / image.height
    ar_errors = bucket_aspect_ratios - aspect_ratio
    bucket_id = np.abs(ar_errors).argmin()
    reso = bucket_resos[bucket_id]
    ar_error = ar_errors[bucket_id]
    img_ar_errors.append(abs(ar_error))

    # どのサイズにリサイズするか→トリミングする方向で
    if ar_error <= 0:                   # 横が長い→縦を合わせる
      scale = reso[1] / image.height
    else:
      scale = reso[0] / image.width

    resized_size = (int(image.width * scale + .5), int(image.height * scale + .5))

    # print(image.width, image.height, bucket_id, bucket_resos[bucket_id], ar_errors[bucket_id], resized_size,
    #       bucket_resos[bucket_id][0] - resized_size[0], bucket_resos[bucket_id][1] - resized_size[1])

    assert resized_size[0] == reso[0] or resized_size[1] == reso[
        1], f"internal error, resized size not match: {reso}, {resized_size}, {image.width}, {image.height}"
    assert resized_size[0] >= reso[0] and resized_size[1] >= reso[
        1], f"internal error, resized size too small: {reso}, {resized_size}, {image.width}, {image.height}"

    # 画像をリサイズしてトリミングする
    # PILにinter_areaがないのでcv2で……
    image = np.array(image)
    image = cv2.resize(image, resized_size, interpolation=cv2.INTER_AREA)
    if resized_size[0] > reso[0]:
      trim_size = resized_size[0] - reso[0]
      image = image[:, trim_size//2:trim_size//2 + reso[0]]
    elif resized_size[1] > reso[1]:
      trim_size = resized_size[1] - reso[1]
      image = image[trim_size//2:trim_size//2 + reso[1]]
    assert image.shape[0] == reso[1] and image.shape[1] == reso[0], f"internal error, illegal trimmed size: {image.shape}, {reso}"

    # # debug
    # cv2.imwrite(f"r:\\test\\img_{i:05d}.jpg", image[:, :, ::-1])

    # バッチへ追加
    buckets_imgs[bucket_id].append((image_key, reso, image))
    bucket_counts[bucket_id] += 1
    metadata[image_key]['train_resolution'] = reso

    # バッチを推論するか判定して推論する
    is_last = i == len(metadata) - 1
    for j in range(len(buckets_imgs)):
      bucket = buckets_imgs[j]
      if (is_last and len(bucket) > 0) or len(bucket) >= args.batch_size:
        latents = get_latents(vae, [img for _, _, img in bucket], weight_dtype)

        for (image_key, reso, _), latent in zip(bucket, latents):
          npz_file_name = os.path.splitext(os.path.basename(image_key))[0]
          np.savez(os.path.join(os.path.dirname(image_key), npz_file_name), latent)

        # flip
        if args.flip_aug:
          latents = get_latents(vae, [img[:, ::-1].copy() for _, _, img in bucket], weight_dtype)   # copyがないとTensor変換できない

          for (image_key, reso, _), latent in zip(bucket, latents):
            npz_file_name = os.path.splitext(os.path.basename(image_key))[0]
            np.savez(os.path.join(os.path.dirname(image_key), npz_file_name + '_flip'), latent)

        bucket.clear()

  for i, (reso, count) in enumerate(zip(bucket_resos, bucket_counts)):
    print(f"bucket {i} {reso}: {count}")
  img_ar_errors = np.array(img_ar_errors)
  print(f"mean ar error: {np.mean(img_ar_errors)}")

  # metadataを書き出して終わり
  print(f"writing metadata: {args.out_json}")
  with open(args.out_json, "wt", encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
  print("done!")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
  parser.add_argument("in_json", type=str, help="metadata file to input / 読み込むメタデータファイル")
  parser.add_argument("out_json", type=str, help="metadata file to output / メタデータファイル書き出し先")
  parser.add_argument("model_name_or_path", type=str, help="model name or path to encode latents / latentを取得するためのモデル")
  parser.add_argument("--v2", action='store_true',
                      help='load Stable Diffusion v2.0 model / Stable Diffusion 2.0のモデルを読み込む')
  parser.add_argument("--batch_size", type=int, default=1, help="batch size in inference / 推論時のバッチサイズ")
  parser.add_argument("--max_resolution", type=str, default="512,512",
                      help="max resolution in fine tuning (width,height) / fine tuning時の最大画像サイズ 「幅,高さ」（使用メモリ量に関係します）")
  parser.add_argument("--min_bucket_reso", type=int, default=256, help="minimum resolution for buckets / bucketの最小解像度")
  parser.add_argument("--max_bucket_reso", type=int, default=1024, help="maximum resolution for buckets / bucketの最小解像度")
  parser.add_argument("--mixed_precision", type=str, default="no",
                      choices=["no", "fp16", "bf16"], help="use mixed precision / 混合精度を使う場合、その精度")
  parser.add_argument("--full_path", action="store_true",
                      help="use full path as image-key in metadata (supports multiple directories) / メタデータで画像キーをフルパスにする（複数の学習画像ディレクトリに対応）")
  parser.add_argument("--flip_aug", action="store_true",
                      help="flip augmentation, save latents for flipped images / 左右反転した画像もlatentを取得、保存する")

  args = parser.parse_args()
  main(args)
