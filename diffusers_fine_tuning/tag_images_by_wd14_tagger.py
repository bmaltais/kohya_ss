# このスクリプトのライセンスは、Apache License 2.0とします
# (c) 2022 Kohya S. @kohya_ss

import argparse
import csv
import glob
import os
import json

from PIL import Image
from tqdm import tqdm
import numpy as np
from tensorflow.keras.models import load_model
from Utils import dbimutils


# from wd14 tagger
IMAGE_SIZE = 448


def main(args):
  image_paths = glob.glob(os.path.join(args.train_data_dir, "*.jpg")) + \
      glob.glob(os.path.join(args.train_data_dir, "*.png")) + glob.glob(os.path.join(args.train_data_dir, "*.webp"))
  print(f"found {len(image_paths)} images.")

  print("loading model and labels")
  model = load_model(args.model)

  # label_names = pd.read_csv("2022_0000_0899_6549/selected_tags.csv")
  # 依存ライブラリを増やしたくないので自力で読むよ
  with open(args.tag_csv, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    l = [row for row in reader]
    header = l[0]             # tag_id,name,category,count
    rows = l[1:]
  assert header[0] == 'tag_id' and header[1] == 'name' and header[2] == 'category', f"unexpected csv format: {header}"

  tags = [row[1] for row in rows[1:] if row[2] == '0']      # categoryが0、つまり通常のタグのみ

  # 推論する
  def run_batch(path_imgs):
    imgs = np.array([im for _, im in path_imgs])

    probs = model(imgs, training=False)
    probs = probs.numpy()

    for (image_path, _), prob in zip(path_imgs, probs):
      # 最初の4つはratingなので無視する
      # # First 4 labels are actually ratings: pick one with argmax
      # ratings_names = label_names[:4]
      # rating_index = ratings_names["probs"].argmax()
      # found_rating = ratings_names[rating_index: rating_index + 1][["name", "probs"]]

      # それ以降はタグなのでconfidenceがthresholdより高いものを追加する
      # Everything else is tags: pick any where prediction confidence > threshold
      tag_text = ""
      for i, p in enumerate(prob[4:]):                # numpyとか使うのが良いけど、まあそれほど数も多くないのでループで
        if p >= args.thresh:
          tag_text += ", " + tags[i]

      if len(tag_text) > 0:
        tag_text = tag_text[2:]                   # 最初の ", " を消す

      with open(os.path.splitext(image_path)[0] + args.caption_extension, "wt", encoding='utf-8') as f:
        f.write(tag_text + '\n')
        if args.debug:
          print(image_path, tag_text)

  b_imgs = []
  for image_path in tqdm(image_paths):
    img = dbimutils.smart_imread(image_path)
    img = dbimutils.smart_24bit(img)
    img = dbimutils.make_square(img, IMAGE_SIZE)
    img = dbimutils.smart_resize(img, IMAGE_SIZE)
    img = img.astype(np.float32)
    b_imgs.append((image_path, img))

    if len(b_imgs) >= args.batch_size:
      run_batch(b_imgs)
      b_imgs.clear()
  if len(b_imgs) > 0:
    run_batch(b_imgs)

  print("done!")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
  parser.add_argument("--model", type=str, default="networks/ViTB16_11_03_2022_07h05m53s",
                      help="model path to load / 読み込むモデルファイル")
  parser.add_argument("--tag_csv", type=str, default="2022_0000_0899_6549/selected_tags.csv",
                      help="csv file for tags / タグ一覧のCSVファイル")
  parser.add_argument("--thresh", type=float, default=0.35, help="threshold of confidence to add a tag / タグを追加するか判定する閾値")
  parser.add_argument("--batch_size", type=int, default=1, help="batch size in inference / 推論時のバッチサイズ")
  parser.add_argument("--caption_extention", type=str, default=None,
                      help="extension of caption file (for backward compatibility) / 出力されるキャプションファイルの拡張子（スペルミスしていたのを残してあります）")
  parser.add_argument("--caption_extension", type=str, default=".txt", help="extension of caption file / 出力されるキャプションファイルの拡張子")
  parser.add_argument("--debug", action="store_true", help="debug mode")

  args = parser.parse_args()

  # スペルミスしていたオプションを復元する
  if args.caption_extention is not None:
    args.caption_extension = args.caption_extention

  main(args)
