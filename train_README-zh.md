__由于文档正在更新中，描述可能有错误。__

# 关于学习

当在存储库中模型的fine tuning、DreamBooth、LoRA 和
文本反转 Textual Inversion（[XTI:P+](https://github.com/kohya-ss/sd-scripts/pull/327)
包括 ）。本文档解释了如何准备培训数据和他们常用的选项。
# 概要

请提前参考本仓库的README，准备好环境。


以下本节说明。

1. 准备训练数据（使用配置文件的新格式）
1. 对研究中使用的术语的非常简短的解释
1. 以前的规范格式（不使用配置文件从命令行指定）
1. 学习过程中样本图像的生成
1. 各脚本通用的常用选项
1. 微调元数据准备：字幕等

如果只执行
1. 暂时可以学习（学习可以参考各个脚本的文档）。
2. 请根据需要参考以下内容。


# 准备训练数据

在任意文件夹（或多个文件夹）中准备训练数据图像文件。支持`.png`、`.jpg`、`.jpeg`、`.webp`、`.bmp`。基本不需要调整大小等预处理。

但是，建议不要使用比学习分辨率（稍后描述）小得多的图像，或者使用超分辨率 AI 提前放大它们。另外，比起超大图像（大约3000x3000像素？），似乎有可能会出错，所以请提前缩小。

训练时，您需要组织您希望模型学习的图像数据，并将其提供给您的脚本。指定训练数据的方式有多种，具体取决于训练数据的数量、学习目标、是否有字幕（图像描述）等。有以下几种方法（每个名字都不是通用的，而是本仓库独有的定义）。正则化图像将在后面讨论。
1. DreamBooth、class+identifier方式（正則化画像使用可）

    学会将学习目标与特定的词（标识符）联系起来。无需提供字幕。例如，如果你用它来学习一个特定的角色，你不需要准备字幕，所以很容易，但是学习数据的所有元素，如发型、衣服和背景，都是通过链接到标识符来学习的。可能是不能按提示换衣服的情况。
1. DreamBooth、标题方式（正則化画像使用可）

    准备并研究一个文本文件，其中为每个图像记录了说明文字。比如在学习一个特定的人物时，通过在caption中描述图像的细节（白衣服的人物A，红衣服的人物A等），将人物与其他元素分离，使其更精确 可以期望模型当时只学习角色。
1. fine tuning方式（正則化画像使用不可）

    提前在元数据文件中收集字幕。它支持诸如单独管理标签和标题以及预缓存潜伏以加快学习速度等功能（均在单独的文档中进行了描述）。 （虽然称为微调法，但也可用于微调以外的方法。）
你想学的东西和你可以使用的规范方法的组合如下。

| 学习对象或方法 | 脚本 | DB/class+identifier | DB/caption | fine tuning |
| ----- | ----- | ----- | ----- | ----- |
| 微调模型 | `fine_tune.py`| x | x | o |
| 模型到 DreamBooth | `train_db.py`| o | o | x |
| LoRA | `train_network.py`| o | o | o |
| Textual Invesion | `train_textual_inversion.py`| o | o | o |

## 选择哪一个

对于LoRA和Textual Inversion，如果不准备caption文件也想轻松学习，DreamBooth class + identifier，能准备的话DreamBooth caption方法不错。如果训练数据的数量很大并且没有使用正则化图像，可以考虑微调方法。

DreamBooth也是一样，但是不能用微调的方法。对于微调，仅使用微调方法。
# 如何指定每个方法

此处仅说明每种规范方法的典型模式。更详细的指定方法请参考[数据集设置](./config_README-zh.md)。
# DreamBooth、class+identifier方式（正則化画像使用可）


这样一来，每张图片都相当于使用标题“类标识符”（例如“shs dog”）进行训练。

## step 1. identifier和class（确定标识符和类)

确定连接您要学习的目标和目标所属类别的单词标识符。

（有各种名称，例如instance，但暂时我会坚持使用原始论文。）

这是一个非常简短的解释（查看更多详细信息）。

class 类是学习的一般类型。例如，如果你想学习特定品种的狗，类就是狗。动漫角色将是男孩、女孩、1 个男孩或 1 个女孩，具体取决于型号。

identifier 标识符用于识别和学习学习目标。任何单词都可以，但根据原始论文，“一个不超过 3 个字母的稀有单词可以成为 tokinizer 的一个标记”是好的。

通过使用标识符和类来训练模型，例如“shs dog”，您可以通过识别类中要学习的对象来进行学习。

生成图像时，如果你说“shs dog”，就会生成学习过的狗品种的图像。

（作为参考，我最近使用的标识符是``shs sts scs cpc coc cic msm usu ici lvl cic dii muk ori hru rik koo yos wny``。实际上，它不包含在Danbooru Tag中。一个是更多可取的。）
## step 2. 决定是否使用正则化图像，如果使用则生成正则化图像
正则化图像是防止整个班级被学习目标拉动（语言漂移）的图像。如果你不使用正则化图像，例如 `shs 1girl` 来学习一个特定的字符，即使您只是提示 `1girl`，它看起来也像那个字符。这是因为 `1girl` 包含在学习说明中。

通过同时学习目标图像和正则化图像，类仍然是类，只有在提示中添加标识符时才会生成目标。

如果你只想在 LoRA 或 DreamBooth 中出现特定的字符，则不需要使用正则化图像。

不应使用文本反转（因为如果要学习的标记字符串不包含在标题中，则什么也学不到）。

作为正则化图像，通常仅在要训练的模型中使用类名生成的图像（例如 `1girl`）。但是，如果生成的图像质量较差，您可以设计提示或使用从 Internet 单独下载的图像。

（还学习了正则化图像，因此它们的质量会影响模型。）

一般来说，准备几百张图像似乎是可取的（如果数量少，类图像将不会被泛化，它们的特征将被学习）。

使用生成图像时，生成图像的大小一般应与训练分辨率（更准确地说是桶的分辨率，稍后描述）相匹配。

## step 2. 编写配置文件

创建一个文本文件并给它一个 .toml 扩展名。例如：

（#开头的部分是注释，可以原样复制粘贴，也可以删除。）

```toml
[general]
enable_bucket = true                        # 是否使用Aspect Ratio Bucketing

[[datasets]]
resolution = 512                            # 学习分辨率
batch_size = 4                              # 批量大小

  [[datasets.subsets]]
  image_dir = 'C:\hoge'                     # 指定包含训练图像的文件夹
  class_tokens = 'hoge girl'                # 指定标识符类
  num_repeats = 10                          # 训练图像的迭代次数

  # 以下仅在使用正则化图像时进行描述。不使用则删除
  [[datasets.subsets]]
  is_reg = true
  image_dir = 'C:\reg'                      # 指定包含正则化图像的文件夹
  class_tokens = 'girl'                     # 指定类别
  num_repeats = 1                           # 正则化图像的迭代次数，基本上1就可以了
```

基本上，你可以通过重写以下地方来学习。

1. 学习分辨率
    如果您指定一个数字，它将是正方形（512x512 对应 512），如果您指定两个数字，用逗号分隔，它将是水平 x 垂直（512x768 对于 `[512,768]`）。在SD1.x系列中，原始学习分辨率为512。如果您指定更大的分辨率，例如 `[512,768]`，您可以减少生成纵向和横向图像时的失败。对于 SD2.x 768 系列，它是“768”。
1. 批量大小

    指定同时学习多少数据。这取决于 GPU 的 VRAM 大小和训练分辨率。稍后会详细介绍。此外，它会根据微调/DreamBooth/LoRA 等而变化。请参阅每个脚本的说明。

1. 文件夹指定

    为训练图像和正则化图像（如果使用）指定文件夹。指定包含图像数据的文件夹本身。

1. 指定标识符和类

    就像前面的例子一样。

1. 重复计数

    我稍后会解释。

### 关于重复次数

迭代次数用于调整正则化图像的数量和训练图像的数量。由于正则化图像的数量多于训练图像的数量，因此重复训练图像以匹配图像的数量，从而可以按 1:1 的比例进行训练。

指定迭代次数，以便“__ 训练图像的迭代次数 x 训练图像的数量 ≥ 正则化图像的迭代次数 x 正则化图像的数量 __”。

一个epoch（数据绕一圈时为一个epoch）的数据数量为“训练图像的重复次数x训练图像的数量”。如果正则化图像的数量大于此，则不使用剩余的正则化图像.)

## step 3. 学習

请参考每个文档和研究。

# DreamBooth、Caption方法（可以使用正则化图片）

在这种方法中，每张图片都用字幕进行训练。

## step 1. 准备字幕文件

在训练图像文件夹中放置一个与图像同名且扩展名为“.caption”（可在设置中更改）的文件。每个文件应该只有一行。编码是“UTF-8”。

## step 2. 决定是否使用正则化图像，如果使用则生成正则化图像

类似于类+标识符格式。正规化图像也可以有说明文字，但通常是不必要的。

## 第二步，编写配置文件

创建一个文本文件并给它一个 .toml 扩展名。例如：

```toml
[general]
enable_bucket = true                        # Aspect Ratio Bucketingを使うか否か

[[datasets]]
resolution = 512                            # 学習解像度
batch_size = 4                              # 批量大小

  [[datasets.subsets]]
  image_dir = 'C:\hoge'                     # 指定包含训练图像的文件夹
  caption_extension = '.caption'            # 使用字幕文件扩展名 .txt 时重写
  num_repeats = 10                          # 训练图像的迭代次数

  # 以下仅在使用正则化图像时进行描述。不使用则删除
  [[datasets.subsets]]
  is_reg = true
  image_dir = 'C:\reg'                      #指定包含正则化图像的文件夹
  class_tokens = 'girl'                     # class を指定
  num_repeats = 1                           # 
正则化图像的迭代次数，基本上1就可以了
```

基本上，您可以通过仅重写以下位置来学习。除非另有说明，否则与类+标识符方法相同。

1. 学习率
1.  批量大小
1. 文件夹指定
1. 字幕文件扩展名

    您可以指定任何扩展名。
1. 重复计数

## step 3. 学習

请参考每个文档和研究。

# fine tuning 方式

## step 1. 准备元数据

汇总标题和标签的管理文件称为元数据。 json 格式，扩展名为 .json
 是。创建方法比较长，所以写在了这篇文档的末尾。

## step 2. 配置文件说明

创建一个文本文件并给它一个 .toml 扩展名。例如：
```toml
[general]
shuffle_caption = true
keep_tokens = 1

[[datasets]]
resolution = 512                                    # 图像分辨率
batch_size = 4                                      # 批量大小

  [[datasets.subsets]]
  image_dir = 'C:\piyo'                             # 指定包含训练图像的文件夹
  metadata_file = 'C:\piyo\piyo_md.json'            # 元数据文件名
```

基本上，您可以通过仅重写以下位置来学习。如无特别说明，与DreamBooth相同，类+标识符方式。

1. 学习率
1. 批量大小
1. 文件夹指定
1. 元数据文件名

    指定通过后述方法创建的元数据文件。


## step 3. 学習

请参考每个文档和研究。

# 对研究中使用的术语的非常简短的解释

细节略去，本人也不是很了解，请各位自行研究。
## fine tuning（微调）

它指的是学习和微调模型。含义因使用方式而异，但狭义上的fine tuning是在Stable Diffusion的情况下学习带有图像和字幕的模型。 DreamBooth可以说是一种狭义微调的特殊方法。广义上的fine tuning包括LoRA、Textual Inversion、Hypernetworks等，包括所有的学习模型。
## 步

粗略地说，一步就是对训练数据进行一次计算。第一步是通过当前模型运行训练数据说明，将生成的图像与训练数据图像进行比较，并稍微修改模型以更接近训练数据。
## 批量大小

批量大小是一个值，它指定在一个步骤中计算多少数据。因为是集体计算，所以速度相对提高了。也有人说准确率普遍较高。
`Batch size x number of steps` 是用于学习的数据数量。因此，根据增加的批量大小减少步骤数是个好主意。

（但是，例如，“1600 steps with a batch size”和“400 steps with batch size of 4”给出的结果不一样。对于相同的学习率，后者通常会学习不足。尝试增加稍微降低速率（例如“2e-6”）或将步数设置为例如 500 步。）

更大的批量大小消耗更多的 GPU 内存。如果内存用完就会出错，学习速度会降低到不出错的限度。最好使用任务管理器或“nvidia-smi”命令检查内存使用量并进行相应调整。

批处理的意思是“一个数据块”。

## 学习率

粗略地说，它表示每一步要改变多少。更高的值会训练得更快，但它可能会改变太多而破坏模型，或者达到次优状态。较小的值会减慢学习速度，并且可能仍会导致次优条件。

fine tuning、DreamBooth 和 LoRA 之间差异很大，还取决于训练数据、你要训练的模型、batch size 和步数。从一般值开始，在观察学习状态的同时增加或减少。

默认情况下，学习率在整个训练过程中是固定的。您可以通过指定调度程序来决定如何更改学习率，因此结果也会根据它们而变化。
## 时代（epoch）


训练数据学习一次（数据绕一圈）就是1个epoch。如果指定重复次数，则重复完成后数据的一个epoch。

一个epoch的步数基本上是`数据个数除以batch size`，但是如果使用Aspect Ratio Bucketing，会稍微增加（因为不同bucket的数据不能一起batch，所以步数会增加).
## Aspect Ratio Bucketing
Stable Diffusion v1 在 512\*512 下训练，但也在 256\*1024 和 384\*640 等分辨率下训练。预计这将减少裁剪部分并更正确地学习字幕和图像之间的关系。

此外，由于它可以在任何分辨率下学习，因此不再需要预先统一图像数据的纵横比。
它可以在设置中在启用和禁用之间切换，但在目前的配置文件描述示例中，它是启用的（设置了`true`）。

学习分辨率在作为参数给出的分辨率区域（=内存使用量）范围内以 64 像素（默认，可变）为单位进行垂直和水平调整。

在机器学习中，统一所有的输入大小是很常见的，但是没有特别的限制，其实只要在同一个batch内统一就可以了。 NovelAI的bucketing好像是指预先按照长宽比对每个学习分辨率的训练数据进行分类。并通过将每个桶中的图像创建一个批次，统一批次的图像大小。

# 旧规范格式（不使用配置文件从命令行指定）

这是一个没有指定 .toml 文件的命令行选项。有DreamBooth类+标识符方式、DreamBooth字幕方式、微调方式。
## DreamBooth、class+identifier方式

用文件夹名称指定重复次数。还可以使用 `train_data_dir` 和 `reg_data_dir` 选项。
### step 1. 为训练准备图像

创建一个文件夹来存储训练图像。 __另外，创建一个目录，名称如下：
```
<重复计数>_<identifier> <class>
```

不要忘记它们之间的``_``。

例如，在提示“sls frog”时，要重复数据 20 次，则为“20_sls frog”。它将如下所示。
![image](https://user-images.githubusercontent.com/52813779/210770636-1c851377-5936-4c15-90b7-8ac8ad6c2074.png)

### 多类、多对象（标识符）学习

方法很简单，training image文件夹下有多个``Repetition count_<identifier> <class>``的文件夹，regularization image文件夹下有``Repetition count_<class>``的文件夹。请准备多个

例如，同时学习“sls frog”和“cpc rabbit”会是这样的：

![image](https://user-images.githubusercontent.com/52813779/210777933-a22229db-b219-4cd8-83ca-e87320fc4192.png)

如果你有一个类和多个目标，你可以只有一个正则化图像文件夹。例如，如果1girl 有字符A 和字符B，则执行如下操作。
- train_girls
  - 10_sls 1girl
  - 10_cpc 1girl
- reg_girls
  - 1_1girl

### step 2. 准备正规化图像

这是使用规则化图像时的过程。

创建一个文件夹来存储规则化的图像。 __此外，__ 创建一个名为``<repeat count>_<class>`` 的目录。

例如，使用提示“frog”并且不重复数据（仅一次）：
![image](https://user-images.githubusercontent.com/52813779/210770897-329758e5-3675-49f1-b345-c135f1725832.png)


### step 3. 训练跑

运行每个训练脚本。 `--train_data_dir`选项设置训练数据文件夹（__不是包含图像的文件夹，其父文件夹__），`--reg_data_dir`选项设置正则化图像文件夹（__包含图像请指定其父文件夹__）而不是文件夹.
## DreamBooth、标题法

如果在训练图像和正则化图像文件夹中放置一个与图像同名且扩展名为 .caption（可在选项中更改）的文件，将从该文件中读取标题并作为提示进行学习。

* 文件夹名称（标识符类）将不再用于训练这些图像。

默认情况下，字幕文件的扩展名为 .caption。您可以使用训练脚本中的“--caption_extension”选项更改它。 `--shuffle_caption` 选项在学习时随机播放字幕的逗号分隔部分。
## fine tuning 方式

直到创建元数据为止，它与使用配置文件时相同。使用 `in_json` 选项指定元数据文件。

# 训练期间的示例输出

您可以通过使用正在训练的模型生成试用图像来检查学习进度。在训练脚本中指定以下选项：

- `--sample_every_n_steps` / `--sample_every_n_epochs`
    
    指定要采样的步数或纪元数。为这些数字中的每一个输出样本。如果两者都指定，则 epoch 数优先。
- `--sample_prompts`

    指定示例输出的提示文件。

- `--sample_sampler`

    指定用于采样输出的采样器。
    `'ddim', 'pndm', 'heun', 'dpmsolver', 'dpmsolver++', 'dpmsingle', 'k_lms', 'k_euler', 'k_euler_a', 'k_dpm_2', 'k_dpm_2_a'`が選べます。

要输出样本，您需要提前准备一个包含提示的文本文件。每行输入一个提示。

```txt
# prompt 1
masterpiece, best quality, 1girl, in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

# prompt 2
masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
```

以“#”开头的行是注释。您可以使用“`--` + 小写字母”为生成的图像指定选项，例如 `--n`。您可以使用：

- `--n` 否定提示到下一个选项。
- `--w` 指定生成图像的宽度。
- `--h` 指定生成图像的高度。
- `--d` 指定生成图像的种子。
- `--l` 指定生成图像的 CFG 比例。
- `--s` 指定生成过程中的步骤数。


# 每个脚本通用的常用选项

文档更新可能跟不上脚本更新。在这种情况下，请使用 `--help` 选项检查可用选项。
## 学习模型规范

- `--v2` / `--v_parameterization`
    
   如果使用 Hugging Face 的 stable-diffusion-2-base 或来自它的微调模型作为学习目标模型（对于在推理时指示使用 `v2-inference.yaml` 的模型），`- 当使用-v2` 选项与 stable-diffusion-2、768-v-ema.ckpt 及其微调模型（对于在推理过程中使用 `v2-inference-v.yaml` 的模型），`- 指定两个 -v2`和 `--v_parameterization` 选项。

    以下几点在 Stable Diffusion 2.0 中发生了显着变化。

    1.  使用分词器
    2. 使用哪个Text Encoder，使用哪个输出层（2.0使用倒数第二层）
    3. Text Encoder的输出维度(768->1024)
    4. U-Net的结构（CrossAttention的头数等）
    5. v-parameterization（采样方式好像变了）

    其中碱基使用1-4个，非碱基使用1-5个（768-v）。使用 1-4 进行 v2 选择，使用 5 进行 v_parameterization 选择。
-`--pretrained_model_name_or_path`
    
    指定要从中执行额外训练的模型。您可以指定稳定扩散检查点文件（.ckpt 或 .safetensors）、扩散器本地磁盘上的模型目录或扩散器模型 ID（例如“stabilityai/stable-diffusion-2”）。
## 学习设置

- `--output_dir` 

    指定训练后保存模型的文件夹。
    
- `--output_name` 
    
    指定不带扩展名的模型文件名。
    
- `--dataset_config` 

    指定描述数据集配置的 .toml 文件。

- `--max_train_steps` / `--max_train_epochs`

    指定要学习的步数或纪元数。如果两者都指定，则 epoch 数优先。
- 
- `--mixed_precision`

 训练混合精度以节省内存。指定像`--mixed_precision = "fp16"`。与无混合精度（默认）相比，精度可能较低，但训练所需的 GPU 内存明显较少。
    
    （在RTX30系列以后也可以指定`bf16`，请配合您在搭建环境时做的加速设置）。    
- `--gradient_checkpointing`

  通过逐步计算权重而不是在训练期间一次计算所有权重来减少训练所需的 GPU 内存量。关闭它不会影响准确性，但打开它允许更大的批量大小，所以那里有影响。
    
    另外，打开它通常会减慢速度，但可以增加批量大小，因此总的学习时间实际上可能会更快。

- `--xformers` / `--mem_eff_attn`

   当指定 xformers 选项时，使用 xformers 的 CrossAttention。如果未安装 xformers 或发生错误（取决于环境，例如 `mixed_precision="no"`），请指定 `mem_eff_attn` 选项而不是使用 CrossAttention 的内存节省版本（xformers 比 慢）。
- `--save_precision`

   指定保存时的数据精度。为 save_precision 选项指定 float、fp16 或 bf16 将以该格式保存模型（在 DreamBooth 中保存 Diffusers 格式时无效，微调）。当您想缩小模型的尺寸时请使用它。
- `--save_every_n_epochs` / `--save_state` / `--resume`
    为 save_every_n_epochs 选项指定一个数字可以在每个时期的训练期间保存模型。

    如果同时指定save_state选项，学习状态包括优化器的状态等都会一起保存。。保存目的地将是一个文件夹。
    
    学习状态输出到目标文件夹中名为“<output_name>-??????-state”（??????是纪元数）的文件夹中。长时间学习时请使用。

    使用 resume 选项从保存的训练状态恢复训练。指定学习状态文件夹（其中的状态文件夹，而不是 `output_dir`）。

    请注意，由于 Accelerator 规范，epoch 数和全局步数不会保存，即使恢复时它们也从 1 开始。
- `--save_model_as` （DreamBooth, fine tuning 仅有的）

  您可以从 `ckpt, safetensors, diffusers, diffusers_safetensors` 中选择模型保存格式。
 
- `--save_model_as=safetensors` 指定喜欢当读取稳定扩散格式（ckpt 或安全张量）并以扩散器格式保存时，缺少的信息通过从 Hugging Face 中删除 v1.5 或 v2.1 信息来补充。
    
- `--clip_skip`
    
    `2`  如果指定，则使用文本编码器 (CLIP) 的倒数第二层的输出。如果省略 1 或选项，则使用最后一层。

    *SD2.0默认使用倒数第二层，学习SD2.0时请不要指定。

    如果被训练的模型最初被训练为使用第二层，则 2 是一个很好的值。

    如果您使用的是最后一层，那么整个模型都会根据该假设进行训练。因此，如果再次使用第二层进行训练，可能需要一定数量的teacher数据和更长时间的学习才能得到想要的学习结果。
- `--max_token_length`

    默认值为 75。您可以通过指定“150”或“225”来扩展令牌长度来学习。使用长字幕学习时指定。
    
    但由于学习时token展开的规范与Automatic1111的web UI（除法等规范）略有不同，如非必要建议用75学习。

    与clip_skip一样，学习与模型学习状态不同的长度可能需要一定量的teacher数据和更长的学习时间。

- `--persistent_data_loader_workers`

    在 Windows 环境中指定它可以显着减少时期之间的延迟。

- `--max_data_loader_n_workers`

    指定数据加载的进程数。大量的进程会更快地加载数据并更有效地使用 GPU，但会消耗更多的主内存。默认是"`8`或者`CPU并发执行线程数 - 1`，取小者"，所以如果主存没有空间或者GPU使用率大概在90%以上，就看那些数字和 `2` 或将其降低到大约 `1`。
- `--logging_dir` / `--log_prefix`

   保存学习日志的选项。在 logging_dir 选项中指定日志保存目标文件夹。以 TensorBoard 格式保存日志。

    例如，如果您指定 --logging_dir=logs，将在您的工作文件夹中创建一个日志文件夹，并将日志保存在日期/时间文件夹中。
    此外，如果您指定 --log_prefix 选项，则指定的字符串将添加到日期和时间之前。使用“--logging_dir=logs --log_prefix=db_style1_”进行识别。

    要检查 TensorBoard 中的日志，请打开另一个命令提示符并在您的工作文件夹中键入：
    ```
    tensorboard --logdir=logs
    ```

   我觉得tensorboard会在环境搭建的时候安装，如果没有安装，请用`pip install tensorboard`安装。）

    然后打开浏览器到http://localhost:6006/就可以看到了。
- `--noise_offset`
本文的实现：https://www.crosslabs.org//blog/diffusion-with-offset-noise
    
    看起来它可能会为整体更暗和更亮的图像产生更好的结果。它似乎对 LoRA 学习也有效。指定一个大约 0.1 的值似乎很好。

- `--debug_dataset`

   通过添加此选项，您可以在学习之前检查将学习什么样的图像数据和标题。按 Esc 退出并返回命令行。按 `S` 进入下一步（批次），按 `E` 进入下一个纪元。

    *图片在 Linux 环境（包括 Colab）下不显示。

- `--vae`

   如果您在 vae 选项中指定稳定扩散检查点、VAE 检查点文件、扩散模型或 VAE（两者都可以指定本地或拥抱面模型 ID），则该 VAE 用于学习（缓存时的潜伏）或在学习过程中获得潜伏）。

    对于 DreamBooth 和微调，保存的模型将包含此 VAE

- `--cache_latents`

  在主内存中缓存 VAE 输出以减少 VRAM 使用。除 flip_aug 之外的任何增强都将不可用。此外，整体学习速度略快。
- `--min_snr_gamma`

    指定最小 SNR 加权策略。细节是[这里](https://github.com/kohya-ss/sd-scripts/pull/308)请参阅。论文中推荐`5`。

## 优化器相关

- `--optimizer_type`
    -- 指定优化器类型。您可以指定
    - AdamW : [torch.optim.AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
    - 与过去版本中未指定选项时相同
    - AdamW8bit : 同上
    - 与过去版本中指定的 --use_8bit_adam 相同
    - Lion : https://github.com/lucidrains/lion-pytorch
    - 与过去版本中指定的 --use_lion_optimizer 相同
    - SGDNesterov : [torch.optim.SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html), nesterov=True
    - SGDNesterov8bit : 引数同上
    - DAdaptation : https://github.com/facebookresearch/dadaptation
    - AdaFactor : [Transformers AdaFactor](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)
    - 任何优化器

- `--learning_rate`

   指定学习率。合适的学习率取决于学习脚本，所以请参考每个解释。
- `--lr_scheduler` / `--lr_warmup_steps` / `--lr_scheduler_num_cycles` / `--lr_scheduler_power`
  
    学习率的调度程序相关规范。

    使用 lr_scheduler 选项，您可以从线性、余弦、cosine_with_restarts、多项式、常数、constant_with_warmup 或任何调度程序中选择学习率调度程序。默认值是常量。
    
    使用 lr_warmup_steps，您可以指定预热调度程序的步数（逐渐改变学习率）。
    
    lr_scheduler_num_cycles 是 cosine with restarts 调度器中的重启次数，lr_scheduler_power 是多项式调度器中的多项式幂。

    有关详细信息，请自行研究。

    要使用任何调度程序，请像使用任何优化器一样使用“--scheduler_args”指定可选参数。
### 关于指定优化器

使用 --optimizer_args 选项指定优化器选项参数。可以以key=value的格式指定多个值。此外，您可以指定多个值，以逗号分隔。例如，要指定 AdamW 优化器的参数，``--optimizer_args weight_decay=0.01 betas=.9,.999``。

指定可选参数时，请检查每个优化器的规格。
一些优化器有一个必需的参数，如果省略它会自动添加（例如 SGDNesterov 的动量）。检查控制台输出。

D-Adaptation 优化器自动调整学习率。学习率选项指定的值不是学习率本身，而是D-Adaptation决定的学习率的应用率，所以通常指定1.0。如果您希望 Text Encoder 的学习率是 U-Net 的一半，请指定 ``--text_encoder_lr=0.5 --unet_lr=1.0``。
如果指定 relative_step=True，AdaFactor 优化器可以自动调整学习率（如果省略，将默认添加）。自动调整时，学习率调度器被迫使用 adafactor_scheduler。此外，指定 scale_parameter 和 warmup_init 似乎也不错。

自动调整的选项类似于``--optimizer_args "relative_step=True" "scale_parameter=True" "warmup_init=True"``。

如果您不想自动调整学习率，请添加可选参数 ``relative_step=False``。在那种情况下，似乎建议将 constant_with_warmup 用于学习率调度程序，而不要为梯度剪裁范数。所以参数就像``--optimizer_type=adafactor --optimizer_args "relative_step=False" --lr_scheduler="constant_with_warmup" --max_grad_norm=0.0``。

### 使用任何优化器

使用 ``torch.optim`` 优化器时，仅指定类名（例如 ``--optimizer_type=RMSprop``），使用其他模块的优化器时，指定“模块名.类名”。（例如``--optimizer_type=bitsandbytes.optim.lamb.LAMB``）。

（内部仅通过 importlib 未确认操作。如果需要，请安装包。）
<!-- 
## 使用任意大小的图像进行训练 --resolution
你可以在广场外学习。请在分辨率中指定“宽度、高度”，如“448,640”。宽度和高度必须能被 64 整除。匹配训练图像和正则化图像的大小。

就我个人而言，我经常生成垂直长的图像，所以我有时会用“448、640”来学习。

## 纵横比分桶 --enable_bucket / --min_bucket_reso / --max_bucket_reso
它通过指定 enable_bucket 选项来启用。 Stable Diffusion 在 512x512 分辨率下训练，但也在 256x768 和 384x640 等分辨率下训练。

如果指定此选项，则不需要将训练图像和正则化图像统一为特定分辨率。从多种分辨率（纵横比）中进行选择，并在该分辨率下学习。
由于分辨率为 64 像素，纵横比可能与原始图像不完全相同。

您可以使用 min_bucket_reso 选项指定分辨率的最小大小，使用 max_bucket_reso 指定最大大小。默认值分别为 256 和 1024。
例如，将最小尺寸指定为 384 将不会使用 256x1024 或 320x768 等分辨率。
如果将分辨率增加到 768x768，您可能需要将 1280 指定为最大尺寸。

启用 Aspect Ratio Ratio Bucketing 时，最好准备具有与训练图像相似的各种分辨率的正则化图像。

（因为一批中的图像不偏向于训练图像和正则化图像。

## 扩充 --color_aug / --flip_aug
增强是一种通过在学习过程中动态改变数据来提高模型性能的方法。在使用 color_aug 巧妙地改变色调并使用 flip_aug 左右翻转的同时学习。

由于数据是动态变化的，因此不能与 cache_latents 选项一起指定。

## 使用 fp16 梯度训练（实验特征）--full_fp16
如果指定 full_fp16 选项，梯度从普通 float32 变为 float16 (fp16) 并学习（它似乎是 full fp16 学习而不是混合精度）。
结果，似乎 SD1.x 512x512 大小可以在 VRAM 使用量小于 8GB 的​​情况下学习，而 SD2.x 512x512 大小可以在 VRAM 使用量小于 12GB 的情况下学习。

预先在加速配置中指定 fp16，并可选择设置 ``mixed_precision="fp16"``（bf16 不起作用）。

为了最大限度地减少内存使用，请使用 xformers、use_8bit_adam、cache_latents、gradient_checkpointing 选项并将 train_batch_size 设置为 1。

（如果你负担得起，逐步增加 train_batch_size 应该会提高一点精度。）

它是通过修补 PyTorch 源代码实现的（已通过 PyTorch 1.12.1 和 1.13.0 确认）。准确率会大幅下降，途中学习失败的概率也会增加。
学习率和步数的设置似乎很严格。请注意它们并自行承担使用它们的风险。
-->

# 创建元数据文件

## 准备教师资料

如上所述准备好你要学习的图像数据，放在任意文件夹中。

例如，存储这样的图像：

![教师数据文件夹的屏幕截图](https://user-images.githubusercontent.com/52813779/208907739-8e89d5fa-6ca8-4b60-8927-f484d2a9ae04.png)

## 自动字幕

如果您只想学习没有标题的标签，请跳过。

另外，手动准备字幕时，请准备在与教师数据图像相同的目录下，文件名相同，扩展名.caption等。每个文件应该是只有一行的文本文件。
### 使用 BLIP 添加字幕

最新版本不再需要 BLIP 下载、权重下载和额外的虚拟环境。按原样工作。

运行 finetune 文件夹中的 make_captions.py。

```
python finetune\make_captions.py --batch_size <バッチサイズ> <教師データフォルダ>
```

如果batch size为8，训练数据放在父文件夹train_data中，则会如下所示
```
python finetune\make_captions.py --batch_size 8 ..\train_data
```

字幕文件创建在与教师数据图像相同的目录中，具有相同的文件名和扩展名.caption。

根据 GPU 的 VRAM 容量增加或减少 batch_size。越大越快（我认为 12GB 的 VRAM 可以多一点）。
您可以使用 max_length 选项指定标题的最大长度。默认值为 75。如果使用 225 的令牌长度训练模型，它可能会更长。
您可以使用 caption_extension 选项更改标题扩展名。默认为 .caption（.txt 与稍后描述的 DeepDanbooru 冲突）。
如果有多个教师数据文件夹，则对每个文件夹执行。

请注意，推理是随机的，因此每次运行时结果都会发生变化。如果要修复它，请使用 --seed 选项指定一个随机数种子，例如 `--seed 42`。

其他的选项，请参考help with `--help`（好像没有文档说明参数的含义，得看源码）。

默认情况下，会生成扩展名为 .caption 的字幕文件。

![captionが生成されたフォルダ](https://user-images.githubusercontent.com/52813779/208908845-48a9d36c-f6ee-4dae-af71-9ab462d1459e.png)

例如，标题如下：

![字幕和图像](https://user-images.githubusercontent.com/52813779/208908947-af936957-5d73-4339-b6c8-945a52857373.png)

## 由 DeepDanbooru 标记

如果不想给danbooru标签本身打标签，请继续“标题和标签信息的预处理”。

标记是使用 DeepDanbooru 或 WD14Tagger 完成的。 WD14Tagger 似乎更准确。如果您想使用 WD14Tagger 进行标记，请跳至下一章。
### 环境布置

将 DeepDanbooru https://github.com/KichangKim/DeepDanbooru 克隆到您的工作文件夹中，或下载并展开 zip。我解压缩了它。
另外，从 DeepDanbooru 发布页面 https://github.com/KichangKim/DeepDanbooru/releases 上的“DeepDanbooru 预训练模型 v3-20211112-sgd-e28”的资产下载 deepdanbooru-v3-20211112-sgd-e28.zip 并解压到 DeepDanbooru 文件夹。

从下面下载。单击以打开资产并从那里下载。

![DeepDanbooru下载页面](https://user-images.githubusercontent.com/52813779/208909417-10e597df-7085-41ee-bd06-3e856a1339df.png)

做一个这样的目录结构

![DeepDanbooru的目录结构](https://user-images.githubusercontent.com/52813779/208909486-38935d8b-8dc6-43f1-84d3-fef99bc471aa.png)
为扩散器环境安装必要的库。进入 DeepDanbooru 文件夹并安装它（我认为它实际上只是添加了 tensorflow-io）。
```
pip install -r requirements.txt
```

接下来，安装 DeepDanbooru 本身。

```
pip install .
```

这样就完成了标注环境的准备工作。

### 实施标记
转到 DeepDanbooru 的文件夹并运行 deepdanbooru 进行标记。
```
deepdanbooru evaluate <教师资料夹> --project-path deepdanbooru-v3-20211112-sgd-e28 --allow-folder --save-txt
```

如果将训练数据放在父文件夹train_data中，则如下所示。
```
deepdanbooru evaluate ../train_data --project-path deepdanbooru-v3-20211112-sgd-e28 --allow-folder --save-txt
```

在与教师数据图像相同的目录中创建具有相同文件名和扩展名.txt 的标记文件。它很慢，因为它是一个接一个地处理的。

如果有多个教师数据文件夹，则对每个文件夹执行。

它生成如下。

![DeepDanbooru生成的文件](https://user-images.githubusercontent.com/52813779/208909855-d21b9c98-f2d3-4283-8238-5b0e5aad6691.png)

它会被这样标记（信息量很大...）。

![DeepDanbooru标签和图片](https://user-images.githubusercontent.com/52813779/208909908-a7920174-266e-48d5-aaef-940aba709519.png)

## WD14Tagger标记为

此过程使用 WD14Tagger 而不是 DeepDanbooru。

使用 Mr. Automatic1111 的 WebUI 中使用的标记器。我参考了这个 github 页面上的信息 (https://github.com/toriato/stable-diffusion-webui-wd14-tagger#mrsmilingwolfs-model-aka-waifu-diffusion-14-tagger)。

初始环境维护所需的模块已经安装。权重自动从 Hugging Face 下载。
### 实施标记

运行脚本以进行标记。
```
python tag_images_by_wd14_tagger.py --batch_size <バッチサイズ> <教師データフォルダ>
```

如果将训练数据放在父文件夹train_data中，则如下所示
```
python tag_images_by_wd14_tagger.py --batch_size 4 ..\train_data
```

模型文件将在首次启动时自动下载到 wd14_tagger_model 文件夹（文件夹可以在选项中更改）。它将如下所示。
![ダウンロードされたファイル](https://user-images.githubusercontent.com/52813779/208910447-f7eb0582-90d6-49d3-a666-2b508c7d1842.png)

在与教师数据图像相同的目录中创建具有相同文件名和扩展名.txt 的标记文件。
![生成的标签文件](https://user-images.githubusercontent.com/52813779/208910534-ea514373-1185-4b7d-9ae3-61eb50bc294e.png)

![标签和图片](https://user-images.githubusercontent.com/52813779/208910599-29070c15-7639-474f-b3e4-06bd5a3df29e.png)

使用 thresh 选项，您可以指定确定的标签的置信度数以附加标签。默认值为 0.35，与 WD14Tagger 示例相同。较低的值给出更多的标签，但准确性较低。

根据 GPU 的 VRAM 容量增加或减少 batch_size。越大越快（我认为 12GB 的 VRAM 可以多一点）。您可以使用 caption_extension 选项更改标记文件扩展名。默认为 .txt。

您可以使用 model_dir 选项指定保存模型的文件夹。

此外，如果指定 force_download 选项，即使有保存目标文件夹，也会重新下载模型。

如果有多个教师数据文件夹，则对每个文件夹执行。

## 预处理字幕和标签信息

将字幕和标签作为元数据合并到一个文件中，以便从脚本中轻松处理。
### 字幕预处理

要将字幕放入元数据，请在您的工作文件夹中运行以下命令（如果您不使用字幕进行学习，则不需要运行它）（它实际上是一行，依此类推）。指定 `--full_path` 选项以将图像文件的完整路径存储在元数据中。如果省略此选项，则会记录相对路径，但 .toml 文件中需要单独的文件夹规范。
```
python merge_captions_to_metadata.py --full_path <教师资料夹>
　  --in_json <要读取的元数据文件名> <元数据文件名>
```

元数据文件名是任意名称。
如果训练数据为train_data，没有读取元数据文件，元数据文件为meta_cap.json，则会如下。
```
python merge_captions_to_metadata.py --full_path train_data meta_cap.json
```

您可以使用 caption_extension 选项指定标题扩展。

如果有多个教师数据文件夹，请指定 full_path 参数并为每个文件夹执行。
```
python merge_captions_to_metadata.py --full_path 
    train_data1 meta_cap1.json
python merge_captions_to_metadata.py --full_path --in_json meta_cap1.json 
    train_data2 meta_cap2.json
```
如果省略in_json，如果有写入目标元数据文件，将从那里读取并覆盖。

__* 每次重写 in_json 选项和写入目标并写入单独的元数据文件是安全的。 __
### 标签预处理

同样，标签也收集在元数据中（如果标签不用于学习，则无需这样做）。
```
python merge_dd_tags_to_metadata.py --full_path <教師データフォルダ> 
    --in_json <要读取的元数据文件名> <書き込むメタデータファイル名>
```

同样的目录结构，读取meta_cap.json和写入meta_cap_dd.json时，会是这样的。
```
python merge_dd_tags_to_metadata.py --full_path train_data --in_json meta_cap.json meta_cap_dd.json
```

如果有多个教师数据文件夹，请指定 full_path 参数并为每个文件夹执行。

```
python merge_dd_tags_to_metadata.py --full_path --in_json meta_cap2.json
    train_data1 meta_cap_dd1.json
python merge_dd_tags_to_metadata.py --full_path --in_json meta_cap_dd1.json 
    train_data2 meta_cap_dd2.json
```

如果省略in_json，如果有写入目标元数据文件，将从那里读取并覆盖。
__※ 通过每次重写 in_json 选项和写入目标，写入单独的元数据文件是安全的。 __
### 清洁标题和标签

至此，字幕和 DeepDanbooru 标签已放在元数据文件中。但是，由于拼写变化（*），带有自动字幕的字幕很微妙，并且标签包括下划线和评级（在 DeepDanbooru 的情况下），因此编辑器的替换功能等。您应该使用它来清理您的字幕和标签。
※ 例如，如果您正在学习动漫女孩，那么字幕会有女孩/女孩/女人/女人等变体。另外，对于“动漫女孩”之类的东西，简单地使用“女孩”可能更合适。

提供了清理脚本，请根据情况编辑脚本内容使用。

（不再需要指定教师数据文件夹。元数据中的所有数据将被清除。）
```
python clean_captions_and_tags.py <要读取的元数据文件名> <要写入的元数据文件名>
```

--in_json 请注意，不包括在内。例如：

```
python clean_captions_and_tags.py meta_cap_dd.json meta_clean.json
```

标题和标签的预处理现已完成。

## latents 提前获取潜能

※ 此步骤不是必需的。即使你省略它，你也可以在学习过程中获得潜能的同时学习。
此外，在学习过程中执行 `random_crop` 或 `color_aug` 时，无法提前获取 latents（因为每次学习时图像都会改变）。如果你不预取，你可以从到目前为止的元数据中学习。

事先获取图像的潜在表示并将其保存到磁盘。这允许快速学习。同时进行bucketing（根据纵横比对训练数据进行分类）。

在您的工作文件夹中，键入：

```
python prepare_buckets_latents.py --full_path <教师资料夹>  
    <要读取的元数据文件名> <要写入的元数据文件名> 
    <要微调的模型名称或检查点> 
    --batch_size <批量大小> 
    --max_resolution <分辨率宽、高> 
    --mixed_precision <准确性>
```

如果模型是model.ckpt，batch size 4，training resolution是512\*512，precision是no(float32)，从meta_clean.json读取metadata写入meta_lat.json：
```
python prepare_buckets_latents.py --full_path 
    train_data meta_clean.json meta_lat.json model.ckpt 
    --batch_size 4 --max_resolution 512,512 --mixed_precision no
```

潜在变量以 numpy npz 格式保存在教师数据文件夹中。

您可以使用 --min_bucket_reso 选项指定最小分辨率大小，使用 --max_bucket_reso 选项指定最大分辨率大小。默认值分别为 256 和 1024。例如，指定最小大小为 384 将不会使用 256\*1024 或 320\*768 等分辨率。
如果将分辨率增加到 768\*768 之类的值，则应为最大尺寸指定 1280 之类的值。

如果指定 --flip_aug 选项，它将执行水平翻转扩充（数据扩充）。你可以人为地把数据量翻倍，但是如果你在数据不是左右对称的情况下指定它（比如人物外貌、发型等），学习就不会很顺利。


（这是一个简单的实现，获取翻​​转图像的潜伏并保存 \*\_flip.npz 文件。fline_tune.py 不需要任何选项。如果有带 \_flip 的文件，则随机加载一个没有的文件

即使使用 12GB 的 VRAM，批处理大小也可能会增加一点。
分辨率是一个能被 64 整除的数，由“width, height”指定。在微调期间，分辨率与内存大小直接相关。 512,512 似乎是 VRAM 12GB (*) 的限制。 16GB 可能会增加到 512,704 或 512,768。即使有 256、256 等，8GB 的​​ VRAM 似乎也很困难（因为参数和优化器需要一定数量的内存，而不管分辨率如何）。

※ 还有一份报告称，学习批量大小 1 适用于 12GB VRAM 和 640,640。

分桶的结果显示如下。

![bucketing的結果](https://user-images.githubusercontent.com/52813779/208911419-71c00fbb-2ce6-49d5-89b5-b78d7715e441.png)

如果有多个教师数据文件夹，请指定 full_path 参数并为每个文件夹执行

```
python prepare_buckets_latents.py --full_path  
    train_data1 meta_clean.json meta_lat1.json model.ckpt 
    --batch_size 4 --max_resolution 512,512 --mixed_precision no

python prepare_buckets_latents.py --full_path 
    train_data2 meta_lat1.json meta_lat2.json model.ckpt 
    --batch_size 4 --max_resolution 512,512 --mixed_precision no

```
可以使读源和写目标相同，但分开更安全。

__*每次重写参数并将其写入单独的元数据文件是安全的。 __
