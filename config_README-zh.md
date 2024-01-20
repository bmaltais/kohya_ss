 <documents>
<document index="1">
<source>paste.txt</source>
<document_content>For non-Japanese speakers: this README is provided only in Japanese in the current state. Sorry for inconvenience. We will provide English version in the near future.

`--dataset_config` 可以传递设置文件的说明。

## 概述

通过传递设置文件,用户可以进行更详细的设置。

* 可以设置多个数据集
    * 例如,可以为每个数据集设置 `resolution`,并混合它们进行训练。
    * 对于支持DreamBooth方法和fine tuning方法的训练方法,可以混合使用DreamBooth方式和fine tuning方式的数据集。
* 可以为每个子集更改设置
    * 数据子集是指根据图像目录或元数据拆分的数据集。几个子集构成一个数据集。
    * 诸如 `keep_tokens` 和 `flip_aug` 之类的选项可以为每个子集设置。另一方面,诸如 `resolution` 和 `batch_size` 之类的选项可以为每个数据集设置,属于同一数据集的子集的值将是共享的。具体细节将在下文中说明。

设置文件的格式可以是JSON或TOML。考虑到描述的便利性,建议使用 [TOML](https://toml.io/zh/v1.0.0-rc.2)。下面将以TOML为前提进行说明。

下面是一个用TOML描述的设置文件的示例。

```toml
[general]
shuffle_caption = true
caption_extension = '.txt'
keep_tokens = 1

# 这是一个DreamBooth方式的数据集 
[[datasets]]
resolution = 512  
batch_size = 4
keep_tokens = 2

  [[datasets.subsets]]
  image_dir = 'C:\hoge'
  class_tokens = 'hoge girl'
  # 此子集为keep_tokens = 2(使用属于的数据集的值)

  [[datasets.subsets]]
  image_dir = 'C:\fuga'
  class_tokens = 'fuga boy'
  keep_tokens = 3

  [[datasets.subsets]]
  is_reg = true
  image_dir = 'C:\reg'
  class_tokens = 'human'
  keep_tokens = 1

# 这是一个fine tuning方式的数据集
[[datasets]]
resolution = [768, 768] 
batch_size = 2

  [[datasets.subsets]]
  image_dir = 'C:\piyo'
  metadata_file = 'C:\piyo\piyo_md.json'
  # 此子集为keep_tokens = 1(使用general的值)
```

在此示例中,有3个目录作为DreamBooth方式的数据集以512x512(批量大小4)进行训练,有1个目录作为fine tuning方式的数据集以768x768(批量大小2)进行训练。

## 数据子集的设置

数据集和子集的相关设置分为几个可以注册的部分。

* `[general]`
    * 适用于所有数据集或所有子集的选项指定部分。
    * 如果在数据集设置或子集设置中存在同名选项,则会优先使用数据集或子集的设置。
* `[[datasets]]`
    * `datasets`是数据集设置注册部分。这是指定适用于每个数据集的选项的部分。
    * 如果存在子集设置,则会优先使用子集设置。
* `[[datasets.subsets]]`
    * `datasets.subsets`是子集设置的注册部分。这是指定适用于每个子集的选项的部分。

下面是先前示例中图像目录和注册部分的对应关系图。

```
C:\
├─ hoge  ->  [[datasets.subsets]] No.1   ┐                     ┐
├─ fuga  ->  [[datasets.subsets]] No.2   |-> [[datasets]] No.1 |-> [general]
├─ reg   ->  [[datasets.subsets]] No.3   ┘                     |
└─ piyo  ->  [[datasets.subsets]] No.4  --> [[datasets]] No.2  ┘
```

每个图像目录对应一个 `[[datasets.subsets]]`。然后1个或多个 `[[datasets.subsets]]` 构成一个 `[[datasets]]`。所有的 `[[datasets]]`、`[[datasets.subsets]]` 属于 `[general]`。

根据注册部分,可以指定不同的选项,但如果指定了同名选项,则下级注册部分中的值将被优先使用。建议参考先前 keep_tokens 选项的处理,这将有助于理解。

此外,可指定的选项会根据训练方法支持的技术而变化。

* 仅DreamBooth方式的选项
* 仅fine tuning方式的选项
* 当可以使用caption dropout技术时的选项

在可同时使用DreamBooth方法和fine tuning方法的训练方法中,它们可以一起使用。
需要注意的是,判断是DreamBooth方式还是fine tuning方式是按数据集进行的,因此同一数据集中不能混合存在DreamBooth方式子集和fine tuning方式子集。
也就是说,如果要混合使用这两种方式,则需要设置不同方式的子集属于不同的数据集。

从程序行为上看,如果存在 `metadata_file` 选项,则判断为fine tuning方式的子集。
因此,对于属于同一数据集的子集,要么全部都具有 `metadata_file` 选项,要么全部都不具有 `metadata_file` 选项,这两种情况之一。

下面解释可用的选项。对于与命令行参数名称相同的选项,基本上省略解释。请参阅其他自述文件。

### 所有训练方法通用的选项

不管训练方法如何,都可以指定的选项。

#### 数据集的选项

与数据集设置相关的选项。不能在 `datasets.subsets` 中描述。

| 选项名称 | 设置示例 | `[general]` | `[[datasets]]` |
| ---- | ---- | ---- | ---- |
| `batch_size` | `1` | o | o |
| `bucket_no_upscale` | `true` | o | o |
| `bucket_reso_steps` | `64` | o | o |
| `enable_bucket` | `true` | o | o |
| `max_bucket_reso` | `1024` | o | o |
| `min_bucket_reso` | `128` | o | o |
| `resolution` | `256`, `[512, 512]` | o | o |

* `batch_size`
    * 等同于命令行参数 `--train_batch_size`。

这些设置对每个数据集是固定的。
也就是说,属于该数据集的子集将共享这些设置。
例如,如果要准备不同分辨率的数据集,可以像上面的示例那样将它们定义为单独的数据集,以便可以为它们单独设置不同的分辨率。

#### 子集的选项

与子集设置相关的选项。

| 选项名称 | 设置示例 | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- | ---- |
| `color_aug` | `false` | o | o | o |
| `face_crop_aug_range` | `[1.0, 3.0]` | o | o | o |
| `flip_aug` | `true` | o | o | o |
| `keep_tokens` | `2` | o | o | o |
| `num_repeats` | `10` | o | o | o |
| `random_crop` | `false` | o | o | o |
| `shuffle_caption` | `true` | o | o | o |
| `caption_prefix` | `“masterpiece, best quality, ”` | o | o | o |
| `caption_suffix` | `“, from side”` | o | o | o |

* `num_repeats`
    * 指定子集中图像的重复次数。在fine tuning中相当于 `--dataset_repeats`,但 `num_repeats`
     在任何训练方法中都可以指定。
* `caption_prefix`, `caption_suffix`
    * 指定在标题前后附加的字符串。包括这些字符串的状态下进行洗牌。当指定 `keep_tokens` 时请注意。

### 仅DreamBooth方式的选项

DreamBooth方式的选项仅存在于子集选项中。  

#### 子集的选项

DreamBooth方式子集的设置相关选项。

| 选项名称 | 设置示例 | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- | ---- |  
| `image_dir` | `‘C:\hoge’` | - | - | o(必需) |
| `caption_extension` | `".txt"` | o | o | o |
| `class_tokens` | `“sks girl”` | - | - | o |
| `is_reg` | `false` | - | - | o |

首先需要注意的是,`image_dir` 必须指定图像文件直接放置在路径下。与以往的 DreamBooth 方法不同,它与放置在子目录中的图像不兼容。另外,即使使用像 `5_cat` 这样的文件夹名,也不会反映图像的重复次数和类名。如果要单独设置这些,需要使用 `num_repeats` 和 `class_tokens` 明确指定。

* `image_dir`  
    * 指定图像目录路径。必填选项。
    * 图像需要直接放置在目录下。
* `class_tokens`
    * 设置类标记。
    * 仅在不存在与图像对应的 caption 文件时才在训练中使用。是否使用是按图像判断的。如果没有指定 `class_tokens` 且也找不到 caption 文件,将报错。 
* `is_reg` 
    * 指定子集中的图像是否用于正则化。如果未指定,则视为 `false`,即视为非正则化图像。

### 仅fine tuning方式的选项

fine tuning方式的选项仅存在于子集选项中。  

#### 子集的选项

fine tuning方式子集的设置相关选项。

| 选项名称 | 设置示例 | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |  
| ---- | ---- | ---- | ---- | ---- |
| `image_dir` | `‘C:\hoge’` | - | - | o |
| `metadata_file` | `'C:\piyo\piyo_md.json'` | - | - | o(必需) |

* `image_dir`  
    * 指定图像目录路径。与 DreamBooth 方式不同,不是必填的,但建议设置。
        * 不需要指定的情况是在生成元数据时指定了 `--full_path`。
    * 图像需要直接放置在目录下。
* `metadata_file`
    * 指定子集使用的元数据文件路径。必填选项。
        * 与命令行参数 `--in_json` 等效。
    * 由于子集需要按元数据文件指定,因此避免跨目录创建一个元数据文件。强烈建议为每个图像目录准备元数据文件,并将它们作为单独的子集进行注册。

### 当可使用caption dropout技术时可以指定的选项

当可以使用 caption dropout 技术时的选项仅存在于子集选项中。
无论是 DreamBooth 方式还是 fine tuning 方式,只要训练方法支持 caption dropout,就可以指定。

#### 子集的选项  

可使用 caption dropout 的子集的设置相关选项。

| 选项名称 | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- | 
| `caption_dropout_every_n_epochs` | o | o | o |
| `caption_dropout_rate` | o | o | o |
| `caption_tag_dropout_rate` | o | o | o |

## 存在重复子集时的行为

对于 DreamBooth 方式的数据集,如果其中 `image_dir` 相同的子集将被视为重复。
对于 fine tuning 方式的数据集,如果其中 `metadata_file` 相同的子集将被视为重复。  
如果数据集中存在重复的子集,则从第二个开始将被忽略。

另一方面,如果它们属于不同的数据集,则不会被视为重复。 
例如,像下面这样在不同的数据集中放置相同的 `image_dir` 的子集,不会被视为重复。
这在想要以不同分辨率训练相同图像的情况下很有用。

```toml
# 即使存在于不同的数据集中,也不会被视为重复,两者都将用于训练

[[datasets]] 
resolution = 512

  [[datasets.subsets]]
  image_dir = 'C:\hoge'

[[datasets]]
resolution = 768

  [[datasets.subsets]]
  image_dir = 'C:\hoge'  
```

## 与命令行参数的组合使用

设置文件中的某些选项与命令行参数选项的作用相同。

如果传递了设置文件,则会忽略下列命令行参数选项。

* `--train_data_dir`
* `--reg_data_dir` 
* `--in_json`

如果同时在命令行参数和设置文件中指定了下列命令行参数选项,则设置文件中的值将优先于命令行参数的值。除非另有说明,否则它们是同名选项。  

| 命令行参数选项 | 优先的设置文件选项 |
| ---- | ---- | 
| `--bucket_no_upscale` | |
| `--bucket_reso_steps` | |  
| `--caption_dropout_every_n_epochs` | |
| `--caption_dropout_rate` | |
| `--caption_extension` | |
| `--caption_tag_dropout_rate` | | 
| `--color_aug` | |
| `--dataset_repeats` | `num_repeats` |
| `--enable_bucket` | | 
| `--face_crop_aug_range` | |
| `--flip_aug` | |
| `--keep_tokens` | |
| `--min_bucket_reso` | | 
| `--random_crop`| |
| `--resolution` | |  
| `--shuffle_caption` | |
| `--train_batch_size` | `batch_size` |  

## 错误指南

目前,正在使用外部库来检查设置文件是否正确,但维护不够充分,错误消息难以理解。
计划在将来改进此问题。

作为次佳方案,列出一些常见错误和解决方法。  
如果确信设置正确但仍出现错误,或者完全不明白错误内容,可能是bug,请联系我们。

* `voluptuous.error.MultipleInvalid: required key not provided @ ...`:: 缺少必填选项的错误。可能忘记指定或错误输入了选项名。
  * `...`部分显示了错误发生位置。例如,`voluptuous.error.MultipleInvalid: required key not provided @ data['datasets'][0]['subsets'][0]['image_dir']` 意味着在第0个 `datasets` 的第0个 `subsets` 的设置中缺失 `image_dir`。
* `voluptuous.error.MultipleInvalid: expected int for dictionary value @ ...`:值的格式错误。输入值的格式可能错误。可以参考本 README 中选项的「设置示例」部分。 
* `voluptuous.error.MultipleInvalid: extra keys not allowed @ ...`:存在不支持的选项名。可能错误输入或误输入了选项名。

</document_content>
</document>
</documents>