 SDXL已得到支持。sdxl分支已合并到main分支。当更新仓库时,请执行升级步骤。由于accelerate版本也已经升级,请重新运行accelerate config。

有关SDXL训练的信息,请参见[此处](./README.md#sdxl-training)(英文)。

## 关于本仓库

用于Stable Diffusion的训练、图像生成和其他脚本的仓库。

[英文README](./README.md) <- 更新信息在这里

[bmaltais的仓库](https://github.com/bmaltais/kohya_ss)中提供了GUI和PowerShell脚本等使其更易于使用的功能(英文),也请一并参阅。衷心感谢bmaltais。

包含以下脚本:

* 支持DreamBooth、U-Net和Text Encoder的训练
* 微调,同上
* 支持LoRA的训练
* 图像生成
* 模型转换(在Stable Diffision ckpt/safetensors与Diffusers之间转换)

## 使用方法

* [通用部分的训练信息](./docs/train_README-ja.md): 数据准备和选项等
* [数据集设置](./docs/config_README-ja.md)
* [DreamBooth的训练信息](./docs/train_db_README-ja.md)  
* [微调指南](./docs/fine_tune_README_ja.md)
* [LoRA的训练信息](./docs/train_network_README-ja.md)
* [Textual Inversion的训练信息](./docs/train_ti_README-ja.md)
* [图像生成脚本](./docs/gen_img_README-ja.md)
* note.com [模型转换脚本](https://note.com/kohya_ss/n/n374f316fe4ad)

## Windows上需要的程序

需要Python 3.10.6和Git。

- Python 3.10.6: https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe
- git: https://git-scm.com/download/win  

如果要在PowerShell中使用,请按以下步骤更改安全设置以使用venv。
(不仅仅是venv,这使得脚本的执行成为可能,所以请注意。)

- 以管理员身份打开PowerShell。
- 输入“Set-ExecutionPolicy Unrestricted”,并回答Y。  
- 关闭管理员PowerShell。

## 在Windows环境下安装

脚本已在PyTorch 2.0.1上通过测试。PyTorch 1.12.1也应该可以工作。

下例中,将安装PyTorch 2.0.1/CUDA 11.8版。如果使用CUDA 11.6版或PyTorch 1.12.1,请酌情更改。  

(注意,如果python -m venv~这行只显示“python”,请将其更改为py -m venv~。)

如果使用PowerShell,请打开常规(非管理员)PowerShell并按顺序执行以下操作:  

```powershell
git clone https://github.com/kohya-ss/sd-scripts.git 
cd sd-scripts

python -m venv venv
.\venv\Scripts\activate  

pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade -r requirements.txt
pip install xformers==0.0.20  

accelerate config
```

在命令提示符下也相同。  

(注:由于 ``python -m venv venv`` 比 ``python -m venv --system-site-packages venv`` 更安全,已进行更改。如果global python中安装了package,后者会引发各种问题。) 

在accelerate config的提示中,请按以下方式回答。(如果以bf16学习,最后一个问题回答bf16。)  

※从0.15.0开始,在日语环境中按方向键选择会崩溃(......)。请使用数字键0、1、2......进行选择。  

```txt
- This machine  
- No distributed training
- NO  
- NO
- NO
- all
- fp16
```

※有时可能会出现 ``ValueError: fp16 mixed precision requires a GPU`` 错误。在这种情况下,对第6个问题 ``(What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]:``
)回答“0”。(将使用id `0`的GPU)。

### 可选:``bitsandbytes``(8位优化器)

`bitsandbytes`现在是可选的。在Linux上,可以通过pip正常安装(推荐0.41.1或更高版本)。  

在Windows上,推荐0.35.0或0.41.1。

- `bitsandbytes` 0.35.0: 似乎是稳定的版本。可以使用AdamW8bit,但不能使用其他一些8位优化器和`full_bf16`学习时的选项。
- `bitsandbytes` 0.41.1: 支持 Lion8bit、PagedAdamW8bit、PagedLion8bit。可以使用`full_bf16`。   

注意:`bitsandbytes` 从0.35.0到0.41.0之间的版本似乎存在问题。 https://github.com/TimDettmers/bitsandbytes/issues/659  

请按以下步骤安装`bitsandbytes`。   

### 使用0.35.0  

以下是PowerShell的例子。在命令提示符中,请使用copy代替cp。   

```powershell    
cd sd-scripts
.\venv\Scripts\activate
pip install bitsandbytes==0.35.0  

cp .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\  
cp .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
cp .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py
```  

### 使用0.41.1  

请从[此处](https://github.com/jllllll/bitsandbytes-windows-webui)或其他地方安装jllllll发布的Windows whl文件。   

```powershell   
python -m pip install bitsandbytes==0.41.1 --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui 
```


### 可选:使用Lion8bit 

如果要使用Lion8bit,需要将`bitsandbytes`升级到0.38.0以上。首先卸载`bitsandbytes`,然后在Windows中安装适合Windows的whl文件,例如[这里的](https://github.com/jllllll/bitsandbytes-windows-webui)。例如:

```powershell
pip install https://github.com/jllllll/bitsandbytes-windows-webui/raw/main/bitsandbytes-0.38.1-py3-none-any.whl
```

升级时用`pip install .`更新这个仓库,并视情况升级其他包。

### 可选:使用PagedAdamW8bit和PagedLion8bit

如果要使用PagedAdamW8bit和PagedLion8bit,需要将`bitsandbytes`升级到0.39.0以上。首先卸载`bitsandbytes`,然后在Windows中安装适合Windows的whl文件,例如[这里的](https://github.com/jllllll/bitsandbytes-windows-webui)。例如:

```powershell
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.39.1-py3-none-win_amd64.whl  
```

升级时用`pip install .`更新这个仓库,并视情况升级其他包。

## 升级

如果有新版本,可以用以下命令更新:

```powershell
cd sd-scripts
git pull
.\venv\Scripts\activate  
pip install --use-pep517 --upgrade -r requirements.txt
```

如果命令成功,就可以使用新版本了。

## 致谢

LoRA实现基于[cloneofsimo的仓库](https://github.com/cloneofsimo/lora)。表示感谢。

将Conv2d 3x3扩展到所有层起初由 [cloneofsimo](https://github.com/cloneofsimo/lora) 发布, [KohakuBlueleaf](https://github.com/KohakuBlueleaf/LoCon) 证明了其有效性。深深感谢 KohakuBlueleaf。

## 许可

脚本遵循 ASL 2.0 许可,但包含其他许可的代码部分(Diffusers和cloneofsimo的仓库)。

[Memory Efficient Attention Pytorch](https://github.com/lucidrains/memory-efficient-attention-pytorch): MIT 

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes): MIT

[BLIP](https://github.com/salesforce/BLIP): BSD-3-Clause