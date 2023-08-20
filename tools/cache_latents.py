# latentsのdiskへの事前キャッシュを行う / cache latents to disk

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


def cache_to_disk(args: argparse.Namespace) -> None:
    train_util.prepare_dataset_args(args, True)

    # check cache latents arg
    assert args.cache_latents_to_disk, "cache_latents_to_disk must be True / cache_latents_to_diskはTrueである必要があります"

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
            print(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                print(
                    "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                        ", ".join(ignored)
                    )
                )
        else:
            if use_dreambooth_method:
                print("Using DreamBooth method.")
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
                print("Training with captions.")
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

    # datasetのcache_latentsを呼ばなければ、生の画像が返る

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collater = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collater = train_util.collater_class(current_epoch, current_step, ds_for_collater)

    # acceleratorを準備する
    print("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, _ = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    # モデルを読み込む
    print("load model")
    if args.sdxl:
        (_, _, _, vae, _, _, _) = sdxl_train_util.load_target_model(args, accelerator, "sdxl", weight_dtype)
    else:
        _, vae, _, _ = train_util.load_target_model(args, weight_dtype, accelerator)

    if torch.__version__ >= "2.0.0": # PyTorch 2.0.0 以上対応のxformersなら以下が使える
        vae.set_use_memory_efficient_attention_xformers(args.xformers)
    vae.to(accelerator.device, dtype=vae_dtype)
    vae.requires_grad_(False)
    vae.eval()

    # dataloaderを準備する
    train_dataset_group.set_caching_mode("latents")

    # DataLoaderのプロセス数：0はメインプロセスになる
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count() - 1)  # cpu_count-1 ただし最大で指定された数まで

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collater,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    # acceleratorを使ってモデルを準備する：マルチGPUで使えるようになるはず
    train_dataloader = accelerator.prepare(train_dataloader)

    # データ取得のためのループ
    for batch in tqdm(train_dataloader):
        b_size = len(batch["images"])
        vae_batch_size = b_size if args.vae_batch_size is None else args.vae_batch_size
        flip_aug = batch["flip_aug"]
        random_crop = batch["random_crop"]
        bucket_reso = batch["bucket_reso"]

        # バッチを分割して処理する
        for i in range(0, b_size, vae_batch_size):
            images = batch["images"][i : i + vae_batch_size]
            absolute_paths = batch["absolute_paths"][i : i + vae_batch_size]
            resized_sizes = batch["resized_sizes"][i : i + vae_batch_size]

            image_infos = []
            for i, (image, absolute_path, resized_size) in enumerate(zip(images, absolute_paths, resized_sizes)):
                image_info = train_util.ImageInfo(absolute_path, 1, "dummy", False, absolute_path)
                image_info.image = image
                image_info.bucket_reso = bucket_reso
                image_info.resized_size = resized_size
                image_info.latents_npz = os.path.splitext(absolute_path)[0] + ".npz"

                if args.skip_existing:
                    if train_util.is_disk_cached_latents_is_expected(image_info.bucket_reso, image_info.latents_npz, flip_aug):
                        print(f"Skipping {image_info.latents_npz} because it already exists.")
                        continue

                image_infos.append(image_info)

            if len(image_infos) > 0:
                train_util.cache_batch_latents(vae, True, image_infos, flip_aug, random_crop)

    accelerator.wait_for_everyone()
    accelerator.print(f"Finished caching latents for {len(train_dataset_group)} batches.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    train_util.add_sd_models_arguments(parser)
    train_util.add_training_arguments(parser, True)
    train_util.add_dataset_arguments(parser, True, True, True)
    config_util.add_config_arguments(parser)
    parser.add_argument("--sdxl", action="store_true", help="Use SDXL model / SDXLモデルを使用する")
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
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
