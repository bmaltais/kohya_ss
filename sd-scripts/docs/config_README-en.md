First version: A.I Translation by Model: NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO, editing by Darkstorm2150

Document is updated and maintained manually.

# Config Readme

This README is about the configuration files that can be passed with the `--dataset_config` option.

## Overview

By passing a configuration file, users can make detailed settings.

* Multiple datasets can be configured
   * For example, by setting `resolution` for each dataset, they can be mixed and trained.
   * In training methods that support both the DreamBooth approach and the fine-tuning approach, datasets of the DreamBooth method and the fine-tuning method can be mixed.
* Settings can be changed for each subset
   * A subset is a partition of the dataset by image directory or metadata. Several subsets make up a dataset.
   * Options such as `keep_tokens` and `flip_aug` can be set for each subset. On the other hand, options such as `resolution` and `batch_size` can be set for each dataset, and their values are common among subsets belonging to the same dataset. More details will be provided later.

The configuration file format can be JSON or TOML. Considering the ease of writing, it is recommended to use [TOML](https://toml.io/ja/v1.0.0-rc.2). The following explanation assumes the use of TOML.


Here is an example of a configuration file written in TOML.

```toml
[general]
shuffle_caption = true
caption_extension = '.txt'
keep_tokens = 1

# This is a DreamBooth-style dataset
[[datasets]]
resolution = 512
batch_size = 4
keep_tokens = 2

  [[datasets.subsets]]
  image_dir = 'C:\hoge'
  class_tokens = 'hoge girl'
  # This subset uses keep_tokens = 2 (the value of the parent datasets)

  [[datasets.subsets]]
  image_dir = 'C:\fuga'
  class_tokens = 'fuga boy'
  keep_tokens = 3

  [[datasets.subsets]]
  is_reg = true
  image_dir = 'C:\reg'
  class_tokens = 'human'
  keep_tokens = 1

# This is a fine-tuning dataset
[[datasets]]
resolution = [768, 768]
batch_size = 2

  [[datasets.subsets]]
  image_dir = 'C:\piyo'
  metadata_file = 'C:\piyo\piyo_md.json'
  # This subset uses keep_tokens = 1 (the value of [general])
```

In this example, three directories are trained as a DreamBooth-style dataset at 512x512 (batch size 4), and one directory is trained as a fine-tuning dataset at 768x768 (batch size 2).

## Settings for datasets and subsets

Settings for datasets and subsets are divided into several registration locations.

* `[general]`
    * This is where options that apply to all datasets or all subsets are specified.
    * If there are options with the same name in the dataset-specific or subset-specific settings, the dataset-specific or subset-specific settings take precedence.
* `[[datasets]]`
    * `datasets` is where settings for datasets are registered. This is where options that apply individually to each dataset are specified.
	* If there are subset-specific settings, the subset-specific settings take precedence.
* `[[datasets.subsets]]`
    * `datasets.subsets` is where settings for subsets are registered. This is where options that apply individually to each subset are specified.

Here is an image showing the correspondence between image directories and registration locations in the previous example.

```
C:\
├─ hoge  ->  [[datasets.subsets]] No.1  ┐                        ┐
├─ fuga  ->  [[datasets.subsets]] No.2  |->  [[datasets]] No.1   |->  [general]
├─ reg   ->  [[datasets.subsets]] No.3  ┘                        |
└─ piyo  ->  [[datasets.subsets]] No.4  -->  [[datasets]] No.2   ┘
```

The image directory corresponds to each `[[datasets.subsets]]`. Then, multiple `[[datasets.subsets]]` are combined to form one `[[datasets]]`. All `[[datasets]]` and `[[datasets.subsets]]` belong to `[general]`.

The available options for each registration location may differ, but if the same option is specified, the value in the lower registration location will take precedence. You can check how the `keep_tokens` option is handled in the previous example for better understanding.

Additionally, the available options may vary depending on the method that the learning approach supports.

* Options specific to the DreamBooth method
* Options specific to the fine-tuning method
* Options available when using the caption dropout technique

When using both the DreamBooth method and the fine-tuning method, they can be used together with a learning approach that supports both.
When using them together, a point to note is that the method is determined based on the dataset, so it is not possible to mix DreamBooth method subsets and fine-tuning method subsets within the same dataset.
In other words, if you want to use both methods together, you need to set up subsets of different methods belonging to different datasets.

In terms of program behavior, if the `metadata_file` option exists, it is determined to be a subset of fine-tuning. Therefore, for subsets belonging to the same dataset, as long as they are either "all have the `metadata_file` option" or "all have no `metadata_file` option," there is no problem.

Below, the available options will be explained. For options with the same name as the command-line argument, the explanation will be omitted in principle. Please refer to other READMEs.

### Common options for all learning methods

These are options that can be specified regardless of the learning method.

#### Data set specific options

These are options related to the configuration of the data set. They cannot be described in `datasets.subsets`.


| Option Name | Example Setting | `[general]` | `[[datasets]]` |
| ---- | ---- | ---- | ---- |
| `batch_size` | `1` | o | o |
| `bucket_no_upscale` | `true` | o | o |
| `bucket_reso_steps` | `64` | o | o |
| `enable_bucket` | `true` | o | o |
| `max_bucket_reso` | `1024` | o | o |
| `min_bucket_reso` | `128` | o | o |
| `resolution` | `256`, `[512, 512]` | o | o |

* `batch_size`
    * This corresponds to the command-line argument `--train_batch_size`.
* `max_bucket_reso`, `min_bucket_reso`
    * Specify the maximum and minimum resolutions of the bucket. It must be divisible by `bucket_reso_steps`.

These settings are fixed per dataset. That means that subsets belonging to the same dataset will share these settings. For example, if you want to prepare datasets with different resolutions, you can define them as separate datasets as shown in the example above, and set different resolutions for each.

#### Options for Subsets

These options are related to subset configuration.

| Option Name | Example | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- | ---- |
| `color_aug` | `false` | o | o | o |
| `face_crop_aug_range` | `[1.0, 3.0]` | o | o | o |
| `flip_aug` | `true` | o | o | o |
| `keep_tokens` | `2` | o | o | o |
| `num_repeats` | `10` | o | o | o |
| `random_crop` | `false` | o | o | o |
| `shuffle_caption` | `true` | o | o | o |
| `caption_prefix` | `"masterpiece, best quality, "` | o | o | o |
| `caption_suffix` | `", from side"` | o | o | o |
| `caption_separator` |  (not specified) | o | o | o |
| `keep_tokens_separator` | `“|||”` | o | o | o |
| `secondary_separator` | `“;;;”` | o | o | o |
| `enable_wildcard` | `true` | o | o | o |
| `resize_interpolation` | (not specified) | o | o | o |

* `num_repeats`
    * Specifies the number of repeats for images in a subset. This is equivalent to `--dataset_repeats` in fine-tuning but can be specified for any training method.
* `caption_prefix`, `caption_suffix`
    * Specifies the prefix and suffix strings to be appended to the captions. Shuffling is performed with these strings included. Be cautious when using `keep_tokens`.
* `caption_separator`
    * Specifies the string to separate the tags. The default is `,`. This option is usually not necessary to set.
* `keep_tokens_separator`
    * Specifies the string to separate the parts to be fixed in the caption. For example, if you specify `aaa, bbb ||| ccc, ddd, eee, fff ||| ggg, hhh`, the parts `aaa, bbb` and `ggg, hhh` will remain, and the rest will be shuffled and dropped. The comma in between is not necessary. As a result, the prompt will be `aaa, bbb, eee, ccc, fff, ggg, hhh` or `aaa, bbb, fff, ccc, eee, ggg, hhh`, etc.
* `secondary_separator`
    * Specifies an additional separator. The part separated by this separator is treated as one tag and is shuffled and dropped. It is then replaced by `caption_separator`. For example, if you specify `aaa;;;bbb;;;ccc`, it will be replaced by `aaa,bbb,ccc` or dropped together.
* `enable_wildcard`
    * Enables wildcard notation. This will be explained later.
* `resize_interpolation`
    * Specifies the interpolation method used when resizing images. Normally, there is no need to specify this. The following options can be specified: `lanczos`, `nearest`, `bilinear`, `linear`, `bicubic`, `cubic`, `area`, `box`. By default (when not specified), `area` is used for downscaling, and `lanczos` is used for upscaling. If this option is specified, the same interpolation method will be used for both upscaling and downscaling. When `lanczos` or `box` is specified, PIL is used; for other options, OpenCV is used.

### DreamBooth-specific options

DreamBooth-specific options only exist as subsets-specific options.

#### Subset-specific options

Options related to the configuration of DreamBooth subsets.

| Option Name | Example Setting | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- | ---- |
| `image_dir` | `'C:\hoge'` | - | - | o (required) |
| `caption_extension` | `".txt"` | o | o | o |
| `class_tokens` | `"sks girl"` | - | - | o |
| `cache_info` | `false` | o | o | o |
| `is_reg` | `false` | - | - | o |

Firstly, note that for `image_dir`, the path to the image files must be specified as being directly in the directory. Unlike the previous DreamBooth method, where images had to be placed in subdirectories, this is not compatible with that specification. Also, even if you name the folder something like "5_cat", the number of repeats of the image and the class name will not be reflected. If you want to set these individually, you will need to explicitly specify them using `num_repeats` and `class_tokens`.

* `image_dir`
    * Specifies the path to the image directory. This is a required option.
    * Images must be placed directly under the directory.
* `class_tokens`
    * Sets the class tokens.
    * Only used during training when a corresponding caption file does not exist. The determination of whether or not to use it is made on a per-image basis. If `class_tokens` is not specified and a caption file is not found, an error will occur.
* `cache_info`
    * Specifies whether to cache the image size and caption. If not specified, it is set to `false`. The cache is saved in `metadata_cache.json` in `image_dir`.
    * Caching speeds up the loading of the dataset after the first time. It is effective when dealing with thousands of images or more.
* `is_reg`
    * Specifies whether the subset images are for normalization. If not specified, it is set to `false`, meaning that the images are not for normalization.

### Fine-tuning method specific options

The options for the fine-tuning method only exist for subset-specific options.

#### Subset-specific options

These options are related to the configuration of the fine-tuning method's subsets.

| Option name | Example setting | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- | ---- |
| `image_dir` | `'C:\hoge'` | - | - | o |
| `metadata_file` | `'C:\piyo\piyo_md.json'` | - | - | o (required) |

* `image_dir`
    * Specify the path to the image directory. Unlike the DreamBooth method, specifying it is not mandatory, but it is recommended to do so.
        * The case where it is not necessary to specify is when the `--full_path` is added to the command line when generating the metadata file.
    * The images must be placed directly under the directory.
* `metadata_file`
    * Specify the path to the metadata file used for the subset. This is a required option.
        * It is equivalent to the command-line argument `--in_json`.
    * Due to the specification that a metadata file must be specified for each subset, it is recommended to avoid creating a metadata file with images from different directories as a single metadata file. It is strongly recommended to prepare a separate metadata file for each image directory and register them as separate subsets.

### Options available when caption dropout method can be used

The options available when the caption dropout method can be used exist only for subsets. Regardless of whether it's the DreamBooth method or fine-tuning method, if it supports caption dropout, it can be specified.

#### Subset-specific options

Options related to the setting of subsets that caption dropout can be used for.

| Option Name | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- |
| `caption_dropout_every_n_epochs` | o | o | o |
| `caption_dropout_rate` | o | o | o |
| `caption_tag_dropout_rate` | o | o | o |

## Behavior when there are duplicate subsets

In the case of the DreamBooth dataset, if there are multiple `image_dir` directories with the same content, they are considered to be duplicate subsets. For the fine-tuning dataset, if there are multiple `metadata_file` files with the same content, they are considered to be duplicate subsets. If duplicate subsets exist in the dataset, subsequent subsets will be ignored.

However, if they belong to different datasets, they are not considered duplicates. For example, if you have subsets with the same `image_dir` in different datasets, they will not be considered duplicates. This is useful when you want to train with the same image but with different resolutions.

```toml
# If data sets exist separately, they are not considered duplicates and are both used for training.

[[datasets]]
resolution = 512

  [[datasets.subsets]]
  image_dir = 'C:\hoge'

[[datasets]]
resolution = 768

  [[datasets.subsets]]
  image_dir = 'C:\hoge'
```

## Command Line Argument and Configuration File

There are options in the configuration file that have overlapping roles with command line argument options.

The following command line argument options are ignored if a configuration file is passed:

* `--train_data_dir`
* `--reg_data_dir`
* `--in_json`

For the command line options listed below, if an option is specified in both the command line arguments and the configuration file, the value from the configuration file will be given priority. Unless otherwise noted, the option names are the same.

| Command Line Argument Option   | Corresponding Configuration File Option |
| ------------------------------- | --------------------------------------- |
| `--bucket_no_upscale`           |                                       |
| `--bucket_reso_steps`           |                                       |
| `--caption_dropout_every_n_epochs` |                                       |
| `--caption_dropout_rate`        |                                       |
| `--caption_extension`           |                                       |
| `--caption_tag_dropout_rate`    |                                       |
| `--color_aug`                   |                                       |
| `--dataset_repeats`             | `num_repeats`                          |
| `--enable_bucket`               |                                       |
| `--face_crop_aug_range`         |                                       |
| `--flip_aug`                    |                                       |
| `--keep_tokens`                 |                                       |
| `--min_bucket_reso`              |                                       |
| `--random_crop`                 |                                       |
| `--resolution`                  |                                       |
| `--shuffle_caption`             |                                       |
| `--train_batch_size`            | `batch_size`                           |

## Error Guide

Currently, we are using an external library to check if the configuration file is written correctly, but the development has not been completed, and there is a problem that the error message is not clear. In the future, we plan to improve this problem.

As a temporary measure, we will list common errors and their solutions. If you encounter an error even though it should be correct or if the error content is not understandable, please contact us as it may be a bug.

* `voluptuous.error.MultipleInvalid: required key not provided @ ...`: This error occurs when a required option is not provided. It is highly likely that you forgot to specify the option or misspelled the option name.
  * The error location is indicated by `...` in the error message. For example, if you encounter an error like `voluptuous.error.MultipleInvalid: required key not provided @ data['datasets'][0]['subsets'][0]['image_dir']`, it means that the `image_dir` option does not exist in the 0th `subsets` of the 0th `datasets` setting.
* `voluptuous.error.MultipleInvalid: expected int for dictionary value @ ...`: This error occurs when the specified value format is incorrect. It is highly likely that the value format is incorrect. The `int` part changes depending on the target option. The example configurations in this README may be helpful.
* `voluptuous.error.MultipleInvalid: extra keys not allowed @ ...`: This error occurs when there is an option name that is not supported. It is highly likely that you misspelled the option name or mistakenly included it.

## Miscellaneous

### Multi-line captions

By setting `enable_wildcard = true`, multiple-line captions are also enabled. If the caption file consists of multiple lines, one line is randomly selected as the caption. 

```txt
1girl, hatsune miku, vocaloid, upper body, looking at viewer, microphone, stage
a girl with a microphone standing on a stage
detailed digital art of a girl with a microphone on a stage
```

It can be combined with wildcard notation.

In metadata files, you can also specify multiple-line captions. In the `.json` metadata file, use `\n` to represent a line break. If the caption file consists of multiple lines, `merge_captions_to_metadata.py` will create a metadata file in this format.

The tags in the metadata (`tags`) are added to each line of the caption.

```json
{
    "/path/to/image.png": {
        "caption": "a cartoon of a frog with the word frog on it\ntest multiline caption1\ntest multiline caption2",
        "tags": "open mouth, simple background, standing, no humans, animal, black background, frog, animal costume, animal focus"
    },
    ...
}
```

In this case, the actual caption will be `a cartoon of a frog with the word frog on it, open mouth, simple background ...`, `test multiline caption1, open mouth, simple background ...`, `test multiline caption2, open mouth, simple background ...`, etc.

### Example of configuration file : `secondary_separator`, wildcard notation, `keep_tokens_separator`, etc.

```toml
[general]
flip_aug = true
color_aug = false
resolution = [1024, 1024]

[[datasets]]
batch_size = 6
enable_bucket = true
bucket_no_upscale = true
caption_extension = ".txt"
keep_tokens_separator= "|||"
shuffle_caption = true
caption_tag_dropout_rate = 0.1
secondary_separator = ";;;" # subset 側に書くこともできます / can be written in the subset side
enable_wildcard = true # 同上 / same as above

  [[datasets.subsets]]
  image_dir = "/path/to/image_dir"
  num_repeats = 1

  # ||| の前後はカンマは不要です（自動的に追加されます） / No comma is required before and after ||| (it is added automatically)
  caption_prefix = "1girl, hatsune miku, vocaloid |||" 
  
  # ||| の後はシャッフル、drop されず残ります / After |||, it is not shuffled or dropped and remains
  # 単純に文字列として連結されるので、カンマなどは自分で入れる必要があります / It is simply concatenated as a string, so you need to put commas yourself
  caption_suffix = ", anime screencap ||| masterpiece, rating: general"
```

### Example of caption, secondary_separator notation: `secondary_separator = ";;;"`

```txt
1girl, hatsune miku, vocaloid, upper body, looking at viewer, sky;;;cloud;;;day, outdoors
```
The part `sky;;;cloud;;;day` is replaced with `sky,cloud,day` without shuffling or dropping. When shuffling and dropping are enabled, it is processed as a whole (as one tag). For example, it becomes `vocaloid, 1girl, upper body, sky,cloud,day, outdoors, hatsune miku` (shuffled) or `vocaloid, 1girl, outdoors, looking at viewer, upper body, hatsune miku` (dropped).

### Example of caption, enable_wildcard notation: `enable_wildcard = true`

```txt
1girl, hatsune miku, vocaloid, upper body, looking at viewer, {simple|white} background
```
`simple` or `white` is randomly selected, and it becomes `simple background` or `white background`.

```txt
1girl, hatsune miku, vocaloid, {{retro style}}
```
If you want to include `{` or `}` in the tag string, double them like `{{` or `}}` (in this example, the actual caption used for training is `{retro style}`).

### Example of caption, `keep_tokens_separator` notation: `keep_tokens_separator = "|||"`

```txt
1girl, hatsune miku, vocaloid ||| stage, microphone, white shirt, smile ||| best quality, rating: general
```
It becomes `1girl, hatsune miku, vocaloid, microphone, stage, white shirt, best quality, rating: general` or `1girl, hatsune miku, vocaloid, white shirt, smile, stage, microphone, best quality, rating: general` etc.

