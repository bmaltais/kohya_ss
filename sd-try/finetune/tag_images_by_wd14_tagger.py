import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

import library.train_util as train_util
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

# from wd14 tagger
IMAGE_SIZE = 448

# wd-v1-4-swinv2-tagger-v2 / wd-v1-4-vit-tagger / wd-v1-4-vit-tagger-v2/ wd-v1-4-convnext-tagger / wd-v1-4-convnext-tagger-v2
DEFAULT_WD14_TAGGER_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
FILES = ["keras_metadata.pb", "saved_model.pb", "selected_tags.csv"]
FILES_ONNX = ["model.onnx"]
SUB_DIR = "variables"
SUB_DIR_FILES = ["variables.data-00000-of-00001", "variables.index"]
CSV_FILE = FILES[-1]


def preprocess_image(image):
    image = np.array(image)
    image = image[:, :, ::-1]  # RGB->BGR

    # pad to square
    size = max(image.shape[0:2])
    pad_x = size - image.shape[1]
    pad_y = size - image.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode="constant", constant_values=255)

    interp = cv2.INTER_AREA if size > IMAGE_SIZE else cv2.INTER_LANCZOS4
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interp)

    image = image.astype(np.float32)
    return image


class ImageLoadingPrepDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])

        try:
            image = Image.open(img_path).convert("RGB")
            image = preprocess_image(image)
            # tensor = torch.tensor(image) # これ Tensor に変換する必要ないな……(;･∀･)
        except Exception as e:
            logger.error(f"Could not load image path / 画像を読み込めません: {img_path}, error: {e}")
            return None

        return (image, img_path)


def collate_fn_remove_corrupted(batch):
    """Collate function that allows to remove corrupted examples in the
    dataloader. It expects that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are removed.
    """
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    return batch


def main(args):
    # model location is model_dir + repo_id
    # repo id may be like "user/repo" or "user/repo/branch", so we need to remove slash
    model_location = os.path.join(args.model_dir, args.repo_id.replace("/", "_"))

    # hf_hub_downloadをそのまま使うとsymlink関係で問題があるらしいので、キャッシュディレクトリとforce_filenameを指定してなんとかする
    # depreacatedの警告が出るけどなくなったらその時
    # https://github.com/toriato/stable-diffusion-webui-wd14-tagger/issues/22
    if not os.path.exists(model_location) or args.force_download:
        os.makedirs(args.model_dir, exist_ok=True)
        logger.info(f"downloading wd14 tagger model from hf_hub. id: {args.repo_id}")
        files = FILES
        if args.onnx:
            files = ["selected_tags.csv"]
            files += FILES_ONNX
        else:
            for file in SUB_DIR_FILES:
                hf_hub_download(
                    args.repo_id,
                    file,
                    subfolder=SUB_DIR,
                    cache_dir=os.path.join(model_location, SUB_DIR),
                    force_download=True,
                    force_filename=file,
                )
        for file in files:
            hf_hub_download(args.repo_id, file, cache_dir=model_location, force_download=True, force_filename=file)
    else:
        logger.info("using existing wd14 tagger model")

    # モデルを読み込む
    if args.onnx:
        import torch
        import onnx
        import onnxruntime as ort

        onnx_path = f"{model_location}/model.onnx"
        logger.info("Running wd14 tagger with onnx")
        logger.info(f"loading onnx model: {onnx_path}")

        if not os.path.exists(onnx_path):
            raise Exception(
                f"onnx model not found: {onnx_path}, please redownload the model with --force_download"
                + " / onnxモデルが見つかりませんでした。--force_downloadで再ダウンロードしてください"
            )

        model = onnx.load(onnx_path)
        input_name = model.graph.input[0].name
        try:
            batch_size = model.graph.input[0].type.tensor_type.shape.dim[0].dim_value
        except Exception:
            batch_size = model.graph.input[0].type.tensor_type.shape.dim[0].dim_param

        if args.batch_size != batch_size and not isinstance(batch_size, str) and batch_size > 0:
            # some rebatch model may use 'N' as dynamic axes
            logger.warning(
                f"Batch size {args.batch_size} doesn't match onnx model batch size {batch_size}, use model batch size {batch_size}"
            )
            args.batch_size = batch_size

        del model

        if "OpenVINOExecutionProvider" in ort.get_available_providers():
            # requires provider options for gpu support
            # fp16 causes nonsense outputs
            ort_sess = ort.InferenceSession(
                onnx_path,
                providers=(["OpenVINOExecutionProvider"]),
                provider_options=[{'device_type' : "GPU_FP32"}],
            )
        else:
            ort_sess = ort.InferenceSession(
                onnx_path,
                providers=(
                    ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else
                    ["ROCMExecutionProvider"] if "ROCMExecutionProvider" in ort.get_available_providers() else
                    ["CPUExecutionProvider"]
                ),
            )
    else:
        from tensorflow.keras.models import load_model

        model = load_model(f"{model_location}")

    # label_names = pd.read_csv("2022_0000_0899_6549/selected_tags.csv")
    # 依存ライブラリを増やしたくないので自力で読むよ

    with open(os.path.join(model_location, CSV_FILE), "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        line = [row for row in reader]
        header = line[0]  # tag_id,name,category,count
        rows = line[1:]
    assert header[0] == "tag_id" and header[1] == "name" and header[2] == "category", f"unexpected csv format: {header}"

    rating_tags = [row[1] for row in rows[0:] if row[2] == "9"]
    general_tags = [row[1] for row in rows[0:] if row[2] == "0"]
    character_tags = [row[1] for row in rows[0:] if row[2] == "4"]

    # preprocess tags in advance
    if args.character_tag_expand:
        for i, tag in enumerate(character_tags):
            if tag.endswith(")"):
                # chara_name_(series) -> chara_name, series
                # chara_name_(costume)_(series) -> chara_name_(costume), series
                tags = tag.split("(")
                character_tag = "(".join(tags[:-1])
                if character_tag.endswith("_"):
                    character_tag = character_tag[:-1]
                series_tag = tags[-1].replace(")", "")
                character_tags[i] = character_tag + args.caption_separator + series_tag

    if args.remove_underscore:
        rating_tags = [tag.replace("_", " ") if len(tag) > 3 else tag for tag in rating_tags]
        general_tags = [tag.replace("_", " ") if len(tag) > 3 else tag for tag in general_tags]
        character_tags = [tag.replace("_", " ") if len(tag) > 3 else tag for tag in character_tags]

    if args.tag_replacement is not None:
        # escape , and ; in tag_replacement: wd14 tag names may contain , and ;
        escaped_tag_replacements = args.tag_replacement.replace("\\,", "@@@@").replace("\\;", "####")
        tag_replacements = escaped_tag_replacements.split(";")
        for tag_replacement in tag_replacements:
            tags = tag_replacement.split(",")  # source, target
            assert len(tags) == 2, f"tag replacement must be in the format of `source,target` / タグの置換は `置換元,置換先` の形式で指定してください: {args.tag_replacement}"

            source, target = [tag.replace("@@@@", ",").replace("####", ";") for tag in tags]
            logger.info(f"replacing tag: {source} -> {target}")

            if source in general_tags:
                general_tags[general_tags.index(source)] = target
            elif source in character_tags:
                character_tags[character_tags.index(source)] = target
            elif source in rating_tags:
                rating_tags[rating_tags.index(source)] = target

    # 画像を読み込む
    train_data_dir_path = Path(args.train_data_dir)
    image_paths = train_util.glob_images_pathlib(train_data_dir_path, args.recursive)
    logger.info(f"found {len(image_paths)} images.")

    tag_freq = {}

    caption_separator = args.caption_separator
    stripped_caption_separator = caption_separator.strip()
    undesired_tags = args.undesired_tags.split(stripped_caption_separator)
    undesired_tags = set([tag.strip() for tag in undesired_tags if tag.strip() != ""])

    always_first_tags = None
    if args.always_first_tags is not None:
        always_first_tags = [tag for tag in args.always_first_tags.split(stripped_caption_separator) if tag.strip() != ""]

    def run_batch(path_imgs):
        imgs = np.array([im for _, im in path_imgs])

        if args.onnx:
            # if len(imgs) < args.batch_size:
            #     imgs = np.concatenate([imgs, np.zeros((args.batch_size - len(imgs), IMAGE_SIZE, IMAGE_SIZE, 3))], axis=0)
            probs = ort_sess.run(None, {input_name: imgs})[0]  # onnx output numpy
            probs = probs[: len(path_imgs)]
        else:
            probs = model(imgs, training=False)
            probs = probs.numpy()

        for (image_path, _), prob in zip(path_imgs, probs):
            combined_tags = []
            rating_tag_text = ""
            character_tag_text = ""
            general_tag_text = ""

            # 最初の4つ以降はタグなのでconfidenceがthreshold以上のものを追加する
            # First 4 labels are ratings, the rest are tags: pick any where prediction confidence >= threshold
            for i, p in enumerate(prob[4:]):
                if i < len(general_tags) and p >= args.general_threshold:
                    tag_name = general_tags[i]

                    if tag_name not in undesired_tags:
                        tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1
                        general_tag_text += caption_separator + tag_name
                        combined_tags.append(tag_name)
                elif i >= len(general_tags) and p >= args.character_threshold:
                    tag_name = character_tags[i - len(general_tags)]

                    if tag_name not in undesired_tags:
                        tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1
                        character_tag_text += caption_separator + tag_name
                        if args.character_tags_first: # insert to the beginning
                            combined_tags.insert(0, tag_name)
                        else:
                            combined_tags.append(tag_name)

            # 最初の4つはratingなのでargmaxで選ぶ
            # First 4 labels are actually ratings: pick one with argmax
            if args.use_rating_tags or args.use_rating_tags_as_last_tag:
                ratings_probs = prob[:4]
                rating_index = ratings_probs.argmax()
                found_rating = rating_tags[rating_index]

                if found_rating not in undesired_tags:
                    tag_freq[found_rating] = tag_freq.get(found_rating, 0) + 1
                    rating_tag_text = found_rating
                    if args.use_rating_tags:
                        combined_tags.insert(0, found_rating) # insert to the beginning
                    else:
                        combined_tags.append(found_rating)

            # 一番最初に置くタグを指定する
            # Always put some tags at the beginning
            if always_first_tags is not None:
                for tag in always_first_tags:
                    if tag in combined_tags:
                        combined_tags.remove(tag)
                        combined_tags.insert(0, tag)

            # 先頭のカンマを取る
            if len(general_tag_text) > 0:
                general_tag_text = general_tag_text[len(caption_separator) :]
            if len(character_tag_text) > 0:
                character_tag_text = character_tag_text[len(caption_separator) :]

            caption_file = os.path.splitext(image_path)[0] + args.caption_extension

            tag_text = caption_separator.join(combined_tags)

            if args.append_tags:
                # Check if file exists
                if os.path.exists(caption_file):
                    with open(caption_file, "rt", encoding="utf-8") as f:
                        # Read file and remove new lines
                        existing_content = f.read().strip("\n")  # Remove newlines

                    # Split the content into tags and store them in a list
                    existing_tags = [tag.strip() for tag in existing_content.split(stripped_caption_separator) if tag.strip()]

                    # Check and remove repeating tags in tag_text
                    new_tags = [tag for tag in combined_tags if tag not in existing_tags]

                    # Create new tag_text
                    tag_text = caption_separator.join(existing_tags + new_tags)

            with open(caption_file, "wt", encoding="utf-8") as f:
                f.write(tag_text + "\n")
                if args.debug:
                    logger.info("")
                    logger.info(f"{image_path}:")
                    logger.info(f"\tRating tags: {rating_tag_text}")
                    logger.info(f"\tCharacter tags: {character_tag_text}")
                    logger.info(f"\tGeneral tags: {general_tag_text}")

    # 読み込みの高速化のためにDataLoaderを使うオプション
    if args.max_data_loader_n_workers is not None:
        dataset = ImageLoadingPrepDataset(image_paths)
        data = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.max_data_loader_n_workers,
            collate_fn=collate_fn_remove_corrupted,
            drop_last=False,
        )
    else:
        data = [[(None, ip)] for ip in image_paths]

    b_imgs = []
    for data_entry in tqdm(data, smoothing=0.0):
        for data in data_entry:
            if data is None:
                continue

            image, image_path = data
            if image is None:
                try:
                    image = Image.open(image_path)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    image = preprocess_image(image)
                except Exception as e:
                    logger.error(f"Could not load image path / 画像を読み込めません: {image_path}, error: {e}")
                    continue
            b_imgs.append((image_path, image))

            if len(b_imgs) >= args.batch_size:
                b_imgs = [(str(image_path), image) for image_path, image in b_imgs]  # Convert image_path to string
                run_batch(b_imgs)
                b_imgs.clear()

    if len(b_imgs) > 0:
        b_imgs = [(str(image_path), image) for image_path, image in b_imgs]  # Convert image_path to string
        run_batch(b_imgs)

    if args.frequency_tags:
        sorted_tags = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)
        print("Tag frequencies:")
        for tag, freq in sorted_tags:
            print(f"{tag}: {freq}")

    logger.info("done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=DEFAULT_WD14_TAGGER_REPO,
        help="repo id for wd14 tagger on Hugging Face / Hugging Faceのwd14 taggerのリポジトリID",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="wd14_tagger_model",
        help="directory to store wd14 tagger model / wd14 taggerのモデルを格納するディレクトリ",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="force downloading wd14 tagger models / wd14 taggerのモデルを再ダウンロードします",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="batch size in inference / 推論時のバッチサイズ"
    )
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=None,
        help="enable image reading by DataLoader with this number of workers (faster) / DataLoaderによる画像読み込みを有効にしてこのワーカー数を適用する（読み込みを高速化）",
    )
    parser.add_argument(
        "--caption_extention",
        type=str,
        default=None,
        help="extension of caption file (for backward compatibility) / 出力されるキャプションファイルの拡張子（スペルミスしていたのを残してあります）",
    )
    parser.add_argument(
        "--caption_extension", type=str, default=".txt", help="extension of caption file / 出力されるキャプションファイルの拡張子"
    )
    parser.add_argument(
        "--thresh", type=float, default=0.35, help="threshold of confidence to add a tag / タグを追加するか判定する閾値"
    )
    parser.add_argument(
        "--general_threshold",
        type=float,
        default=None,
        help="threshold of confidence to add a tag for general category, same as --thresh if omitted / generalカテゴリのタグを追加するための確信度の閾値、省略時は --thresh と同じ",
    )
    parser.add_argument(
        "--character_threshold",
        type=float,
        default=None,
        help="threshold of confidence to add a tag for character category, same as --thres if omitted / characterカテゴリのタグを追加するための確信度の閾値、省略時は --thresh と同じ",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="search for images in subfolders recursively / サブフォルダを再帰的に検索する"
    )
    parser.add_argument(
        "--remove_underscore",
        action="store_true",
        help="replace underscores with spaces in the output tags / 出力されるタグのアンダースコアをスペースに置き換える",
    )
    parser.add_argument(
        "--debug", action="store_true", help="debug mode"
    )
    parser.add_argument(
        "--undesired_tags",
        type=str,
        default="",
        help="comma-separated list of undesired tags to remove from the output / 出力から除外したいタグのカンマ区切りのリスト",
    )
    parser.add_argument(
        "--frequency_tags", action="store_true", help="Show frequency of tags for images / タグの出現頻度を表示する"
    )
    parser.add_argument(
        "--onnx", action="store_true", help="use onnx model for inference / onnxモデルを推論に使用する"
    )
    parser.add_argument(
        "--append_tags", action="store_true", help="Append captions instead of overwriting / 上書きではなくキャプションを追記する"
    )
    parser.add_argument(
        "--use_rating_tags", action="store_true", help="Adds rating tags as the first tag / レーティングタグを最初のタグとして追加する",
    )
    parser.add_argument(
        "--use_rating_tags_as_last_tag", action="store_true", help="Adds rating tags as the last tag / レーティングタグを最後のタグとして追加する",
    )
    parser.add_argument(
        "--character_tags_first", action="store_true", help="Always inserts character tags before the general tags / characterタグを常にgeneralタグの前に出力する",
    )
    parser.add_argument(
        "--always_first_tags",
        type=str,
        default=None,
        help="comma-separated list of tags to always put at the beginning, e.g. `1girl,1boy`"
        + " / 必ず先頭に置くタグのカンマ区切りリスト、例 : `1girl,1boy`",
    )
    parser.add_argument(
        "--caption_separator",
        type=str,
        default=", ",
        help="Separator for captions, include space if needed / キャプションの区切り文字、必要ならスペースを含めてください",
    )
    parser.add_argument(
        "--tag_replacement",
        type=str,
        default=None,
        help="tag replacement in the format of `source1,target1;source2,target2; ...`. Escape `,` and `;` with `\`. e.g. `tag1,tag2;tag3,tag4`"
        + " / タグの置換を `置換元1,置換先1;置換元2,置換先2; ...`で指定する。`\` で `,` と `;` をエスケープできる。例: `tag1,tag2;tag3,tag4`",
    )
    parser.add_argument(
        "--character_tag_expand",
        action="store_true",
        help="expand tag tail parenthesis to another tag for character tags. `chara_name_(series)` becomes `chara_name, series`"
        + " / キャラクタタグの末尾の括弧を別のタグに展開する。`chara_name_(series)` は `chara_name, series` になる",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()

    # スペルミスしていたオプションを復元する
    if args.caption_extention is not None:
        args.caption_extension = args.caption_extention

    if args.general_threshold is None:
        args.general_threshold = args.thresh
    if args.character_threshold is None:
        args.character_threshold = args.thresh

    main(args)
