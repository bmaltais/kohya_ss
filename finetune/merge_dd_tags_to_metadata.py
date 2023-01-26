# このスクリプトのライセンスは、Apache License 2.0とします
# (c) 2022 Kohya S. @kohya_ss

import argparse
import json
from pathlib import Path

from tqdm import tqdm


def main(args):
  
  image_paths = None
  train_data_dir_path = Path(args.train_data_dir)
  if args.recursive:
    image_paths = list(train_data_dir_path.rglob('*.jpg')) + \
                  list(train_data_dir_path.rglob('*.png')) + \
                  list(train_data_dir_path.rglob('*.webp')) 
  else:
    image_paths = list(train_data_dir_path.glob('*.jpg')) + \
                  list(train_data_dir_path.glob('*.png')) + \
                  list(train_data_dir_path.glob('*.webp')) 

  print(f"found {len(image_paths)} images.")

  if args.in_json is None and Path(args.out_json).is_file():
    args.in_json = args.out_json

  if args.in_json is not None:
    print(f"loading existing metadata: {args.in_json}")
    metadata = json.loads(Path(args.in_json).read_text(encoding='utf-8'))
    print("tags data for existing images will be overwritten / 既存の画像のタグは上書きされます")
  else:
    print("new metadata will be created / 新しいメタデータファイルが作成されます")
    metadata = {}

  print("merge tags to metadata json.")
  for image_path in tqdm(image_paths):
    tags_path = image_path.with_suffix('.txt')
    tags = tags_path.read_text(encoding='utf-8').strip()

    image_key = image_path if args.full_path else image_path.stem
    if str(image_key) not in metadata:
      metadata[str(image_key)] = {}

    metadata[str(image_key)]['tags'] = tags
    if args.debug:
      print(image_key, tags)

  # metadataを書き出して終わり
  print(f"writing metadata: {args.out_json}")
  Path(args.out_json).write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    
  print("done!")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
  parser.add_argument("out_json", type=str, help="metadata file to output / メタデータファイル書き出し先")
  parser.add_argument("--in_json", type=str, help="metadata file to input (if omitted and out_json exists, existing out_json is read) / 読み込むメタデータファイル（省略時、out_jsonが存在すればそれを読み込む）")
  parser.add_argument("--full_path", action="store_true",
                      help="use full path as image-key in metadata (supports multiple directories) / メタデータで画像キーをフルパスにする（複数の学習画像ディレクトリに対応）")
  parser.add_argument("--recursive", action="store_true", help="recursively look for training tags in all child folders of train_data_dir / train_data_dirのすべての子フォルダにある学習タグを再帰的に探す")
  parser.add_argument("--debug", action="store_true", help="debug mode, print tags")

  args = parser.parse_args()
  main(args)
