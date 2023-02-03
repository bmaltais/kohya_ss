import argparse
import os
import json

from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms

import library.model_util as model_util
import library.train_util as train_util

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def collate_fn_remove_corrupted(batch):
  """Collate function that allows to remove corrupted examples in the
  dataloader. It expects that the dataloader returns 'None' when that occurs.
  The 'None's in the batch are removed.
  """
  # Filter out all the Nones (corrupted examples)
  batch = list(filter(lambda x: x is not None, batch))
  return batch


def get_latents(vae, images, weight_dtype):
  img_tensors = [IMAGE_TRANSFORMS(image) for image in images]
  img_tensors = torch.stack(img_tensors)
  img_tensors = img_tensors.to(DEVICE, weight_dtype)
  with torch.no_grad():
    latents = vae.encode(img_tensors).latent_dist.sample().float().to("cpu").numpy()
  return latents


def get_npz_filename_wo_ext(data_dir, image_key, is_full_path, flip):
  if is_full_path:
    base_name = os.path.splitext(os.path.basename(image_key))[0]
  else:
    base_name = image_key
  if flip:
    base_name += '_flip'
  return os.path.join(data_dir, base_name)


def main(args):
  image_paths = train_util.glob_images(args.train_data_dir)
  print(f"found {len(image_paths)} images.")

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

  def process_batch(is_last):
    for j in range(len(buckets_imgs)):
      bucket = buckets_imgs[j]
      if (is_last and len(bucket) > 0) or len(bucket) >= args.batch_size:
        latents = get_latents(vae, [img for _, _, img in bucket], weight_dtype)

        for (image_key, _, _), latent in zip(bucket, latents):
          npz_file_name = get_npz_filename_wo_ext(args.train_data_dir, image_key, args.full_path, False)
          np.savez(npz_file_name, latent)

        # flip
        if args.flip_aug:
          latents = get_latents(vae, [img[:, ::-1].copy() for _, _, img in bucket], weight_dtype)   # copyがないとTensor変換できない

          for (image_key, _, _), latent in zip(bucket, latents):
            npz_file_name = get_npz_filename_wo_ext(args.train_data_dir, image_key, args.full_path, True)
            np.savez(npz_file_name, latent)

        bucket.clear()

  # 読み込みの高速化のためにDataLoaderを使うオプション
  if args.max_data_loader_n_workers is not None:
    dataset = train_util.ImageLoadingDataset(image_paths)
    data = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                       num_workers=args.max_data_loader_n_workers, collate_fn=collate_fn_remove_corrupted, drop_last=False)
  else:
    data = [[(None, ip)] for ip in image_paths]

  for data_entry in tqdm(data, smoothing=0.0):
    if data_entry[0] is None:
      continue

    img_tensor, image_path = data_entry[0]
    if img_tensor is not None:
      image = transforms.functional.to_pil_image(img_tensor)
    else:
      try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
          image = image.convert("RGB")
      except Exception as e:
        print(f"Could not load image path / 画像を読み込めません: {image_path}, error: {e}")
        continue

    image_key = image_path if args.full_path else os.path.splitext(os.path.basename(image_path))[0]
    if image_key not in metadata:
      metadata[image_key] = {}

    # 本当はこの部分もDataSetに持っていけば高速化できるがいろいろ大変
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

    # 既に存在するファイルがあればshapeを確認して同じならskipする
    if args.skip_existing:
      npz_files = [get_npz_filename_wo_ext(args.train_data_dir, image_key, args.full_path, False) + ".npz"]
      if args.flip_aug:
        npz_files.append(get_npz_filename_wo_ext(args.train_data_dir, image_key, args.full_path, True) + ".npz")

      found = True
      for npz_file in npz_files:
        if not os.path.exists(npz_file):
          found = False
          break

        dat = np.load(npz_file)['arr_0']
        if dat.shape[1] != reso[1] // 8 or dat.shape[2] != reso[0] // 8:     # latentsのshapeを確認
          found = False
          break
      if found:
        continue

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
    process_batch(False)

  # 残りを処理する
  process_batch(True)

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
                      help='not used (for backward compatibility) / 使用されません（互換性のため残してあります）')
  parser.add_argument("--batch_size", type=int, default=1, help="batch size in inference / 推論時のバッチサイズ")
  parser.add_argument("--max_data_loader_n_workers", type=int, default=None,
                      help="enable image reading by DataLoader with this number of workers (faster) / DataLoaderによる画像読み込みを有効にしてこのワーカー数を適用する（読み込みを高速化）")
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
  parser.add_argument("--skip_existing", action="store_true",
                      help="skip images if npz already exists (both normal and flipped exists if flip_aug is enabled) / npzが既に存在する画像をスキップする（flip_aug有効時は通常、反転の両方が存在する画像をスキップ）")

  args = parser.parse_args()
  main(args)
