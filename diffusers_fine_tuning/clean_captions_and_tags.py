# このスクリプトのライセンスは、Apache License 2.0とします
# (c) 2022 Kohya S. @kohya_ss

import argparse
import glob
import os
import json

from tqdm import tqdm


def clean_tags(image_key, tags):
  # replace '_' to ' '
  tags = tags.replace('_', ' ')

  # remove rating: deepdanbooruのみ
  tokens = tags.split(", rating")
  if len(tokens) == 1:
    # WD14 taggerのときはこちらになるのでメッセージは出さない
    # print("no rating:")
    # print(f"{image_key} {tags}")
    pass
  else:
    if len(tokens) > 2:
      print("multiple ratings:")
      print(f"{image_key} {tags}")
    tags = tokens[0]

  return tags


# 上から順に検索、置換される
# ('置換元文字列', '置換後文字列')
CAPTION_REPLACEMENTS = [
    ('anime anime', 'anime'),
    ('young ', ''),
    ('anime girl', 'girl'),
    ('cartoon female', 'girl'),
    ('cartoon lady', 'girl'),
    ('cartoon character', 'girl'),      # a or ~s
    ('cartoon woman', 'girl'),
    ('cartoon women', 'girls'),
    ('cartoon girl', 'girl'),
    ('anime female', 'girl'),
    ('anime lady', 'girl'),
    ('anime character', 'girl'),      # a or ~s
    ('anime woman', 'girl'),
    ('anime women', 'girls'),
    ('lady', 'girl'),
    ('female', 'girl'),
    ('woman', 'girl'),
    ('women', 'girls'),
    ('people', 'girls'),
    ('person', 'girl'),
    ('a cartoon figure', 'a figure'),
    ('a cartoon image', 'an image'),
    ('a cartoon picture', 'a picture'),
    ('an anime cartoon image', 'an image'),
    ('a cartoon anime drawing', 'a drawing'),
    ('a cartoon drawing', 'a drawing'),
    ('girl girl', 'girl'),
]


def clean_caption(caption):
  for rf, rt in CAPTION_REPLACEMENTS:
    replaced = True
    while replaced:
      bef = caption
      caption = caption.replace(rf, rt)
      replaced = bef != caption
  return caption


def main(args):
  image_paths = glob.glob(os.path.join(args.train_data_dir, "*.jpg")) + glob.glob(os.path.join(args.train_data_dir, "*.png"))
  print(f"found {len(image_paths)} images.")

  if os.path.exists(args.in_json):
    print(f"loading existing metadata: {args.in_json}")
    with open(args.in_json, "rt", encoding='utf-8') as f:
      metadata = json.load(f)
  else:
    print("no metadata / メタデータファイルがありません")
    return

  print("cleaning captions and tags.")
  for image_path in tqdm(image_paths):
    tags_path = os.path.splitext(image_path)[0] + '.txt'
    with open(tags_path, "rt", encoding='utf-8') as f:
      tags = f.readlines()[0].strip()

    image_key = os.path.splitext(os.path.basename(image_path))[0]
    if image_key not in metadata:
      print(f"image not in metadata / メタデータに画像がありません: {image_path}")
      return

    tags = metadata[image_key].get('tags')
    if tags is None:
      print(f"image does not have tags / メタデータにタグがありません: {image_path}")
    else:
      metadata[image_key]['tags'] = clean_tags(image_key, tags)

    caption = metadata[image_key].get('caption')
    if caption is None:
      print(f"image does not have caption / メタデータにキャプションがありません: {image_path}")
    else:
      metadata[image_key]['caption'] = clean_caption(caption)

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
  # parser.add_argument("--debug", action="store_true", help="debug mode")

  args = parser.parse_args()
  main(args)
