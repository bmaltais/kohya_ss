# text encoder出力のdiskへの事前キャッシュを行う / cache text encoder outputs to disk in advance

import argparse
import math
from multiprocessing import Value
import os

from accelerate.utils import set_seed
import torch
from tqdm import tqdm

from library import config_util
from library import train_util
from library import sdxl_train_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
from library.utils import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

def cache_to_disk(args: argparse.Namespace) -> None:
    train_util.prepare_dataset_args(args, True)

    # check cache arg
    assert (
        args.cache_text_encoder_outputs_to_disk
    ), "cache_text_encoder_outputs_to_disk must be True / cache_text_encoder_outputs_to_diskはTrueである必要があります"

    # できるだけ準備はしておくが今のところSDXLのみしか動かない
    assert (
        args.sdxl
    ), "cache_text_encoder_outputs_to_disk is only available for SDXL / cache_text_encoder_outputs_to_diskはSDXLのみ利用可能です"

    use_dreambooth_method = args.in_json is None

    if args.seed is not None:
        set_seed(args.seed)  # 乱数系列を初期化する

    # tokenizerを準備する：datasetを動かすために必要
    if args.sdxl:
        tokenizer1, tokenizer2 = sdxl_train_util.load_tokenizers(args)
        tokenizers = [tokenizer1, tokenizer2]
    else:
        tokenizer = train_util.load_tokenizer(args)
        tokenizers = [tokenizer]

    # データセットを準備する
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, False, True))
        if args.dataset_config is not None:
            logger.info(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning(
                    "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                        ", ".join(ignored)
                    )
                )
        else:
            if use_dreambooth_method:
                logger.info("Using DreamBooth method.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                args.train_data_dir, args.reg_data_dir
                            )
                        }
                    ]
                }
            else:
                logger.info("Training with captions.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": [
                                {
                                    "image_dir": args.train_data_dir,
                                    "metadata_file": args.in_json,
                                }
                            ]
                        }
                    ]
                }

        blueprint = blueprint_generator.generate(user_config, args, tokenizer=tokenizers)
        train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args, tokenizers)

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    # acceleratorを準備する
    logger.info("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, _ = train_util.prepare_dtype(args)

    # モデルを読み込む
    logger.info("load model")
    if args.sdxl:
        (_, text_encoder1, text_encoder2, _, _, _, _) = sdxl_train_util.load_target_model(args, accelerator, "sdxl", weight_dtype)
        text_encoders = [text_encoder1, text_encoder2]
    else:
        text_encoder1, _, _, _ = train_util.load_target_model(args, weight_dtype, accelerator)
        text_encoders = [text_encoder1]

    for text_encoder in text_encoders:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        text_encoder.requires_grad_(False)
        text_encoder.eval()

    # dataloaderを準備する
    train_dataset_group.set_caching_mode("text")

    # DataLoaderのプロセス数：0 は persistent_workers が使えないので注意
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    # acceleratorを使ってモデルを準備する：マルチGPUで使えるようになるはず
    train_dataloader = accelerator.prepare(train_dataloader)

    # データ取得のためのループ
    for batch in tqdm(train_dataloader):
        absolute_paths = batch["absolute_paths"]
        input_ids1_list = batch["input_ids1_list"]
        input_ids2_list = batch["input_ids2_list"]

        image_infos = []
        for absolute_path, input_ids1, input_ids2 in zip(absolute_paths, input_ids1_list, input_ids2_list):
            image_info = train_util.ImageInfo(absolute_path, 1, "dummy", False, absolute_path)
            image_info.text_encoder_outputs_npz = os.path.splitext(absolute_path)[0] + train_util.TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX
            image_info

            if args.skip_existing:
                if os.path.exists(image_info.text_encoder_outputs_npz):
                    logger.warning(f"Skipping {image_info.text_encoder_outputs_npz} because it already exists.")
                    continue
                
            image_info.input_ids1 = input_ids1
            image_info.input_ids2 = input_ids2
            image_infos.append(image_info)

        if len(image_infos) > 0:
            b_input_ids1 = torch.stack([image_info.input_ids1 for image_info in image_infos])
            b_input_ids2 = torch.stack([image_info.input_ids2 for image_info in image_infos])
            train_util.cache_batch_text_encoder_outputs(
                image_infos, tokenizers, text_encoders, args.max_token_length, True, b_input_ids1, b_input_ids2, weight_dtype
            )

    accelerator.wait_for_everyone()
    accelerator.print(f"Finished caching latents for {len(train_dataset_group)} batches.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    train_util.add_sd_models_arguments(parser)
    train_util.add_training_arguments(parser, True)
    train_util.add_dataset_arguments(parser, True, True, True)
    config_util.add_config_arguments(parser)
    sdxl_train_util.add_sdxl_training_arguments(parser)
    parser.add_argument("--sdxl", action="store_true", help="Use SDXL model / SDXLモデルを使用する")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="skip images if npz already exists (both normal and flipped exists if flip_aug is enabled) / npzが既に存在する画像をスキップする（flip_aug有効時は通常、反転の両方が存在する画像をスキップ）",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)

    cache_to_disk(args)
