# このスクリプトのライセンスは、Apache License 2.0とします
# (c) 2022 Kohya S. @kohya_ss

import argparse
import glob
import os
import json

from tqdm import tqdm


def main(args):
  image_paths = glob.glob(os.path.join(args.train_data_dir, "*.jpg")) + \
      glob.glob(os.path.join(args.train_data_dir, "*.png")) + glob.glob(os.path.join(args.train_data_dir, "*.webp"))
  print(f"found {len(image_paths)} images.")

  if args.in_json is None and os.path.isfile(args.out_json):
    args.in_json = args.out_json

  if args.in_json is not None:
    print(f"loading existing metadata: {args.in_json}")
    with open(args.in_json, "rt", encoding='utf-8') as f:
      metadata = json.load(f)
    print("tags data for existing images will be overwritten / 既存の画像のタグは上書きされます")
  else:
    print("new metadata will be created / 新しいメタデータファイルが作成されます")
    metadata = {}

  print("merge tags to metadata json.")
  for image_path in tqdm(image_paths):
    tags_path = os.path.splitext(image_path)[0] + '.txt'
    with open(tags_path, "rt", encoding='utf-8') as f:
      tags = f.readlines()[0].strip()

    image_key = image_path if args.full_path else os.path.splitext(os.path.basename(image_path))[0]
    if image_key not in metadata:
      metadata[image_key] = {}

    metadata[image_key]['tags'] = tags
    if args.debug:
      print(image_key, tags)

  # metadataを書き出して終わり
  print(f"writing metadata: {args.out_json}")
  with open(args.out_json, "wt", encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
  print("done!")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
  parser.add_argument("out_json", type=str, help="metadata file to output / メタデータファイル書き出し先")
  parser.add_argument("--in_json", type=str, help="metadata file to input (if omitted and out_json exists, existing out_json is read) / 読み込むメタデータファイル（省略時、out_jsonが存在すればそれを読み込む）")
  parser.add_argument("--full_path", action="store_true",
                      help="use full path as image-key in metadata (supports multiple directories) / メタデータで画像キーをフルパスにする（複数の学習画像ディレクトリに対応）")
  parser.add_argument("--debug", action="store_true", help="debug mode, print tags")

  args = parser.parse_args()
  main(args)
