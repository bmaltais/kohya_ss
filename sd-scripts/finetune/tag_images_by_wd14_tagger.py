import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

import library.train_util as train_util
from library.utils import setup_logging, resize_image

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

TAG_JSON_FILE = "tag_mapping.json"


def preprocess_image(image: Image.Image) -> np.ndarray:
    # If image has transparency, convert to RGBA. If not, convert to RGB
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        image = image.convert("RGBA")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    # If image is RGBA, combine with white background
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
        image = background

    image = np.array(image)
    image = image[:, :, ::-1]  # RGB->BGR

    # pad to square
    size = max(image.shape[0:2])
    pad_x = size - image.shape[1]
    pad_y = size - image.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode="constant", constant_values=255)

    image = resize_image(image, image.shape[0], image.shape[1], IMAGE_SIZE, IMAGE_SIZE)

    image = image.astype(np.float32)
    return image


class ImageLoadingPrepDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths: list[str], batch_size: int):
        self.image_paths = image_paths
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.image_paths) / self.batch_size)

    def __getitem__(self, batch_index: int) -> tuple[str, np.ndarray, tuple[int, int]]:
        image_index_start = batch_index * self.batch_size
        image_index_end = min((batch_index + 1) * self.batch_size, len(self.image_paths))

        batch_image_paths = []
        images = []
        image_sizes = []
        for idx in range(image_index_start, image_index_end):
            img_path = str(self.image_paths[idx])

            try:
                image = Image.open(img_path)
                image_size = image.size
                image = preprocess_image(image)

                batch_image_paths.append(img_path)
                images.append(image)
                image_sizes.append(image_size)
            except Exception as e:
                logger.error(f"Could not load image path / 画像を読み込めません: {img_path}, error: {e}")

        images = np.stack(images) if len(images) > 0 else np.zeros((0, IMAGE_SIZE, IMAGE_SIZE, 3))
        return batch_image_paths, images, image_sizes


def collate_fn_no_op(batch):
    """Collate function that does nothing and returns the batch as is."""
    return batch


def main(args):
    # model location is model_dir + repo_id
    # given repo_id may be "namespace/repo_name" or "namespace/repo_name/subdir"
    # so we split it to "namespace/reponame" and "subdir"
    tokens = args.repo_id.split("/")

    if len(tokens) > 2:
        repo_id = "/".join(tokens[:2])
        subdir = "/".join(tokens[2:])
        model_location = os.path.join(args.model_dir, repo_id.replace("/", "_"), subdir)
        onnx_model_name = "model_optimized.onnx"
        default_format = False
    else:
        repo_id = args.repo_id
        subdir = None
        model_location = os.path.join(args.model_dir, repo_id.replace("/", "_"))
        onnx_model_name = "model.onnx"
        default_format = True

    # https://github.com/toriato/stable-diffusion-webui-wd14-tagger/issues/22

    if not os.path.exists(model_location) or args.force_download:
        os.makedirs(args.model_dir, exist_ok=True)
        logger.info(f"downloading wd14 tagger model from hf_hub. id: {args.repo_id}")

        if subdir is None:
            # SmilingWolf structure
            files = FILES
            if args.onnx:
                files = ["selected_tags.csv"]
                files += FILES_ONNX
            else:
                for file in SUB_DIR_FILES:
                    hf_hub_download(
                        repo_id=args.repo_id,
                        filename=file,
                        subfolder=SUB_DIR,
                        local_dir=os.path.join(model_location, SUB_DIR),
                        force_download=True,
                    )

            for file in files:
                hf_hub_download(
                    repo_id=args.repo_id,
                    filename=file,
                    local_dir=model_location,
                    force_download=True,
                )
        else:
            # another structure
            files = [onnx_model_name, "tag_mapping.json"]

            for file in files:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=file,
                    subfolder=subdir,
                    local_dir=os.path.join(args.model_dir, repo_id.replace("/", "_")),  # because subdir is specified
                    force_download=True,
                )
    else:
        logger.info("using existing wd14 tagger model")

    # モデルを読み込む
    if args.onnx:
        import onnx
        import onnxruntime as ort

        onnx_path = os.path.join(model_location, onnx_model_name)
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
                provider_options=[{"device_type": "GPU", "precision": "FP32"}],
            )
        else:
            providers = (
                ["CUDAExecutionProvider"]
                if "CUDAExecutionProvider" in ort.get_available_providers()
                else (
                    ["ROCMExecutionProvider"]
                    if "ROCMExecutionProvider" in ort.get_available_providers()
                    else ["CPUExecutionProvider"]
                )
            )
            logger.info(f"Using onnxruntime providers: {providers}")
            ort_sess = ort.InferenceSession(onnx_path, providers=providers)
    else:
        from tensorflow.keras.models import load_model

        model = load_model(f"{model_location}")

    # We read the CSV file manually to avoid adding dependencies.
    # label_names = pd.read_csv("2022_0000_0899_6549/selected_tags.csv")

    def expand_character_tags(char_tags):
        for i, tag in enumerate(char_tags):
            if tag.endswith(")"):
                # chara_name_(series) -> chara_name, series
                # chara_name_(costume)_(series) -> chara_name_(costume), series
                tags = tag.split("(")
                character_tag = "(".join(tags[:-1])
                if character_tag.endswith("_"):
                    character_tag = character_tag[:-1]
                series_tag = tags[-1].replace(")", "")
                char_tags[i] = character_tag + args.caption_separator + series_tag

    def remove_underscore(tags):
        return [tag.replace("_", " ") if len(tag) > 3 else tag for tag in tags]

    def process_tag_replacement(tags: list[str], tag_replacements_arg: str) -> list[str]:
        # escape , and ; in tag_replacement: wd14 tag names may contain , and ;,
        # so user must be specified them like `aa\,bb,AA\,BB;cc\;dd,CC\;DD` which means
        # `aa,bb` is replaced with `AA,BB` and `cc;dd` is replaced with `CC;DD`
        escaped_tag_replacements = tag_replacements_arg.replace("\\,", "@@@@").replace("\\;", "####")
        tag_replacements = escaped_tag_replacements.split(";")

        for tag_replacements_arg in tag_replacements:
            tags = tag_replacements_arg.split(",")  # source, target
            assert (
                len(tags) == 2
            ), f"tag replacement must be in the format of `source,target` / タグの置換は `置換元,置換先` の形式で指定してください: {args.tag_replacement}"

            source, target = [tag.replace("@@@@", ",").replace("####", ";") for tag in tags]
            logger.info(f"replacing tag: {source} -> {target}")

            if source in tags:
                tags[tags.index(source)] = target

        return tags

    if default_format:
        with open(os.path.join(model_location, CSV_FILE), "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            line = [row for row in reader]
            header = line[0]  # tag_id,name,category,count
            rows = line[1:]
        assert header[0] == "tag_id" and header[1] == "name" and header[2] == "category", f"unexpected csv format: {header}"

        rating_tags = [row[1] for row in rows[0:] if row[2] == "9"]
        general_tags = [row[1] for row in rows[0:] if row[2] == "0"]
        character_tags = [row[1] for row in rows[0:] if row[2] == "4"]

        if args.character_tag_expand:
            expand_character_tags(character_tags)
        if args.remove_underscore:
            rating_tags = remove_underscore(rating_tags)
            character_tags = remove_underscore(character_tags)
            general_tags = remove_underscore(general_tags)
        if args.tag_replacement is not None:
            process_tag_replacement(rating_tags, args.tag_replacement)
            process_tag_replacement(general_tags, args.tag_replacement)
            process_tag_replacement(character_tags, args.tag_replacement)
    else:
        with open(os.path.join(model_location, TAG_JSON_FILE), "r", encoding="utf-8") as f:
            tag_mapping = json.load(f)

        rating_tags = []
        general_tags = []
        character_tags = []

        tag_id_to_tag_mapping = {}
        tag_id_to_category_mapping = {}
        for tag_id, tag_info in tag_mapping.items():
            tag = tag_info["tag"]
            category = tag_info["category"]
            assert category in [
                "Rating",
                "General",
                "Character",
                "Copyright",
                "Meta",
                "Model",
                "Quality",
                "Artist",
            ], f"unexpected category: {category}"

            if args.remove_underscore:
                tag = remove_underscore([tag])[0]
            if args.tag_replacement is not None:
                tag = process_tag_replacement([tag], args.tag_replacement)[0]
            if category == "Character" and args.character_tag_expand:
                tag_list = [tag]
                expand_character_tags(tag_list)
                tag = tag_list[0]

            tag_id_to_tag_mapping[int(tag_id)] = tag
            tag_id_to_category_mapping[int(tag_id)] = category

    # 画像を読み込む
    train_data_dir_path = Path(args.train_data_dir)
    image_paths = train_util.glob_images_pathlib(train_data_dir_path, args.recursive)
    logger.info(f"found {len(image_paths)} images.")
    image_paths = [str(ip) for ip in image_paths]

    tag_freq = {}

    caption_separator = args.caption_separator
    stripped_caption_separator = caption_separator.strip()
    undesired_tags = args.undesired_tags.split(stripped_caption_separator)
    undesired_tags = set([tag.strip() for tag in undesired_tags if tag.strip() != ""])

    always_first_tags = None
    if args.always_first_tags is not None:
        always_first_tags = [tag for tag in args.always_first_tags.split(stripped_caption_separator) if tag.strip() != ""]

    def run_batch(path_imgs: tuple[list[str], np.ndarray, list[tuple[int, int]]]) -> Optional[dict[str, dict]]:
        nonlocal args, default_format, model, ort_sess, input_name, tag_freq

        imgs = path_imgs[1]
        result = {}

        if args.onnx:
            # if len(imgs) < args.batch_size:
            #     imgs = np.concatenate([imgs, np.zeros((args.batch_size - len(imgs), IMAGE_SIZE, IMAGE_SIZE, 3))], axis=0)
            if not default_format:
                imgs = imgs.transpose(0, 3, 1, 2)  # to NCHW
                imgs = imgs / 127.5 - 1.0
            probs = ort_sess.run(None, {input_name: imgs})[0]  # onnx output numpy
            probs = probs[: len(imgs)]  # remove padding
        else:
            probs = model(imgs, training=False)
            probs = probs.numpy()

        for image_path, image_size, prob in zip(path_imgs[0], path_imgs[2], probs):
            combined_tags = []
            rating_tag_text = ""
            character_tag_text = ""
            general_tag_text = ""
            other_tag_text = ""

            if default_format:
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
                            if args.character_tags_first:  # insert to the beginning
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
                            combined_tags.insert(0, found_rating)  # insert to the beginning
                        else:
                            combined_tags.append(found_rating)
            else:
                # apply sigmoid to probabilities
                prob = 1 / (1 + np.exp(-prob))

                rating_max_prob = -1
                rating_tag = None
                quality_max_prob = -1
                quality_tag = None
                character_tags = []

                min_thres = min(
                    args.thresh,
                    args.general_threshold,
                    args.character_threshold,
                    args.copyright_threshold,
                    args.meta_threshold,
                    args.model_threshold,
                    args.artist_threshold,
                )
                prob_indices = np.where(prob >= min_thres)[0]
                # for i, p in enumerate(prob):
                for i in prob_indices:
                    if i not in tag_id_to_tag_mapping:
                        continue
                    p = prob[i]

                    tag_name = tag_id_to_tag_mapping[i]
                    category = tag_id_to_category_mapping[i]
                    if tag_name in undesired_tags:
                        continue

                    if category == "Rating":
                        if p > rating_max_prob:
                            rating_max_prob = p
                            rating_tag = tag_name
                            rating_tag_text = tag_name
                        continue
                    elif category == "Quality":
                        if p > quality_max_prob:
                            quality_max_prob = p
                            quality_tag = tag_name
                            if args.use_quality_tags or args.use_quality_tags_as_last_tag:
                                other_tag_text += caption_separator + tag_name
                        continue

                    if category == "General" and p >= args.general_threshold:
                        tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1
                        general_tag_text += caption_separator + tag_name
                        combined_tags.append((tag_name, p))
                    elif category == "Character" and p >= args.character_threshold:
                        tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1
                        character_tag_text += caption_separator + tag_name
                        if args.character_tags_first:  # we separate character tags
                            character_tags.append((tag_name, p))
                        else:
                            combined_tags.append((tag_name, p))
                    elif (
                        (category == "Copyright" and p >= args.copyright_threshold)
                        or (category == "Meta" and p >= args.meta_threshold)
                        or (category == "Model" and p >= args.model_threshold)
                        or (category == "Artist" and p >= args.artist_threshold)
                    ):
                        tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1
                        other_tag_text += f"{caption_separator}{tag_name} ({category})"
                        combined_tags.append((tag_name, p))

                # sort by probability
                combined_tags.sort(key=lambda x: x[1], reverse=True)
                if character_tags:
                    character_tags.sort(key=lambda x: x[1], reverse=True)
                    combined_tags = character_tags + combined_tags
                combined_tags = [t[0] for t in combined_tags]  # remove probability

                if quality_tag is not None:
                    if args.use_quality_tags_as_last_tag:
                        combined_tags.append(quality_tag)
                    elif args.use_quality_tags:
                        combined_tags.insert(0, quality_tag)
                if rating_tag is not None:
                    if args.use_rating_tags_as_last_tag:
                        combined_tags.append(rating_tag)
                    elif args.use_rating_tags:
                        combined_tags.insert(0, rating_tag)

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
            if len(other_tag_text) > 0:
                other_tag_text = other_tag_text[len(caption_separator) :]

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

            if not args.output_path:
                with open(caption_file, "wt", encoding="utf-8") as f:
                    f.write(tag_text + "\n")
            else:
                entry = {"tags": tag_text, "image_size": list(image_size)}
                result[image_path] = entry

            if args.debug:
                logger.info("")
                logger.info(f"{image_path}:")
                logger.info(f"\tRating tags: {rating_tag_text}")
                logger.info(f"\tCharacter tags: {character_tag_text}")
                logger.info(f"\tGeneral tags: {general_tag_text}")
                if other_tag_text:
                    logger.info(f"\tOther tags: {other_tag_text}")

        return result

    # 読み込みの高速化のためにDataLoaderを使うオプション
    if args.max_data_loader_n_workers is not None:
        dataset = ImageLoadingPrepDataset(image_paths, args.batch_size)
        data = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.max_data_loader_n_workers,
            collate_fn=collate_fn_no_op,
            drop_last=False,
        )
    else:
        # data = [[(ip, None, None)] for ip in image_paths]
        data = [[]]
        for ip in image_paths:
            if len(data[-1]) >= args.batch_size:
                data.append([])
            data[-1].append((ip, None, None))

    results = {}
    for data_entry in tqdm(data, smoothing=0.0):
        if data_entry is None or len(data_entry) == 0:
            continue

        if data_entry[0][1] is None:
            # No preloaded image, need to load
            images = []
            image_sizes = []
            for image_path, _, _ in data_entry:
                image = Image.open(image_path)
                image_size = image.size
                image = preprocess_image(image)
                images.append(image)
                image_sizes.append(image_size)
            b_imgs = ([ip for ip, _, _ in data_entry], np.stack(images), image_sizes)
        else:
            b_imgs = data_entry[0]

        r = run_batch(b_imgs)
        if args.output_path and r is not None:
            results.update(r)

    if args.output_path:
        if args.output_path.endswith(".jsonl"):
            # optional JSONL metadata
            with open(args.output_path, "wt", encoding="utf-8") as f:
                for image_path, entry in results.items():
                    f.write(
                        json.dumps({"image_path": image_path, "caption": entry["tags"], "image_size": entry["image_size"]}) + "\n"
                    )
        else:
            # standard JSON metadata
            with open(args.output_path, "wt", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            logger.info(f"captions saved to {args.output_path}")

    if args.frequency_tags:
        sorted_tags = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)
        print("Tag frequencies:")
        for tag, freq in sorted_tags:
            print(f"{tag}: {freq}")

    logger.info("done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
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
    parser.add_argument("--batch_size", type=int, default=1, help="batch size in inference / 推論時のバッチサイズ")
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=None,
        help="enable image reading by DataLoader with this number of workers (faster) / DataLoaderによる画像読み込みを有効にしてこのワーカー数を適用する（読み込みを高速化）",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="path for output captions (json format). if this is set, captions will be saved to this file / 出力キャプションのパス（json形式）。このオプションが設定されている場合、キャプションはこのファイルに保存されます",
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
        help="threshold of confidence to add a tag for character category, same as --thres if omitted. set above 1 to disable character tags"
        " / characterカテゴリのタグを追加するための確信度の閾値、省略時は --thresh と同じ。1以上にするとcharacterタグを無効化できる",
    )
    parser.add_argument(
        "--meta_threshold",
        type=float,
        default=None,
        help="threshold of confidence to add a tag for meta category, same as --thresh if omitted. set above 1 to disable meta tags"
        " / metaカテゴリのタグを追加するための確信度の閾値、省略時は --thresh と同じ。1以上にするとmetaタグを無効化できる",
    )
    parser.add_argument(
        "--model_threshold",
        type=float,
        default=None,
        help="threshold of confidence to add a tag for model category, same as --thresh if omitted. set above 1 to disable model tags"
        " / modelカテゴリのタグを追加するための確信度の閾値、省略時は --thresh と同じ。1以上にするとmodelタグを無効化できる",
    )
    parser.add_argument(
        "--copyright_threshold",
        type=float,
        default=None,
        help="threshold of confidence to add a tag for copyright category, same as --thresh if omitted. set above 1 to disable copyright tags"
        " / copyrightカテゴリのタグを追加するための確信度の閾値、省略時は --thresh と同じ。1以上にするとcopyrightタグを無効化できる",
    )
    parser.add_argument(
        "--artist_threshold",
        type=float,
        default=None,
        help="threshold of confidence to add a tag for artist category, same as --thresh if omitted. set above 1 to disable artist tags"
        " / artistカテゴリのタグを追加するための確信度の閾値、省略時は --thresh と同じ。1以上にするとartistタグを無効化できる",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="search for images in subfolders recursively / サブフォルダを再帰的に検索する"
    )
    parser.add_argument(
        "--remove_underscore",
        action="store_true",
        help="replace underscores with spaces in the output tags / 出力されるタグのアンダースコアをスペースに置き換える",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--undesired_tags",
        type=str,
        default="",
        help="comma-separated list of undesired tags to remove from the output / 出力から除外したいタグのカンマ区切りのリスト",
    )
    parser.add_argument(
        "--frequency_tags", action="store_true", help="Show frequency of tags for images / タグの出現頻度を表示する"
    )
    parser.add_argument("--onnx", action="store_true", help="use onnx model for inference / onnxモデルを推論に使用する")
    parser.add_argument(
        "--append_tags", action="store_true", help="Append captions instead of overwriting / 上書きではなくキャプションを追記する"
    )
    parser.add_argument(
        "--use_rating_tags",
        action="store_true",
        help="Adds rating tags as the first tag / レーティングタグを最初のタグとして追加する",
    )
    parser.add_argument(
        "--use_rating_tags_as_last_tag",
        action="store_true",
        help="Adds rating tags as the last tag / レーティングタグを最後のタグとして追加する",
    )
    parser.add_argument(
        "--use_quality_tags",
        action="store_true",
        help="Adds quality tags as the first tag / クオリティタグを最初のタグとして追加する",
    )
    parser.add_argument(
        "--use_quality_tags_as_last_tag",
        action="store_true",
        help="Adds quality tags as the last tag / クオリティタグを最後のタグとして追加する",
    )
    parser.add_argument(
        "--character_tags_first",
        action="store_true",
        help="Always inserts character tags before the general tags / characterタグを常にgeneralタグの前に出力する",
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
    if args.meta_threshold is None:
        args.meta_threshold = args.thresh
    if args.model_threshold is None:
        args.model_threshold = args.thresh
    if args.copyright_threshold is None:
        args.copyright_threshold = args.thresh
    if args.artist_threshold is None:
        args.artist_threshold = args.thresh

    main(args)
