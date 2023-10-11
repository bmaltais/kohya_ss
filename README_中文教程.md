嗨!我把日语 README 文件的主要内容翻译成中文如下:

## 关于这个仓库

这个是用于Stable Diffusion模型训练、图像生成和其他脚本的仓库。 

[英文版 README](./README.md) <-- 更新信息在这里

GUI和PowerShell脚本等使其更易用的功能在[bmaltais的仓库](https://github.com/bmaltais/kohya_ss)(英语)中提供,一并参考。感谢bmaltais。

包含以下脚本:

* 支持DreamBooth、U-Net和文本编码器的训练
* fine-tuning的支持
* 图像生成
* 模型转换(Stable Diffusion ckpt/safetensors 和 Diffusers之间的相互转换)

## 使用方法 (中国用户只需要按照这个安装教程操作）
- 进入kohya_ss文件夹根目录下，点击 setup.bat 启动安装程序 *（需要科学上网）
- 根据界面上给出的英文选项：
Kohya_ss GUI setup menu:

1. Install kohya_ss gui
2. (Optional) Install cudann files (avoid unless you really need it)
3. (Optional) Install specific bitsandbytes versions
4. (Optional) Manually configure accelerate
5. (Optional) Start Kohya_ss GUI in browser
6. Quit

Enter your choice: 1

1. Torch 1 (legacy, no longer supported. Will be removed in v21.9.x)
2. Torch 2 (recommended)
3. Cancel

Enter your choice: 2

开始安装环境依赖，接着再出来的选项，按照下列选项操作：
```txt
- This machine
- No distributed training
- NO
- NO
- NO
- all
- bf16
```
--------------------------------------------------------------------
这里都选择完毕，即可关闭终端窗口，直接点击 gui.bat或者 kohya中文启动器.bat 即可运行kohya


当仓库内和note.com有相关文章,请参考那里。(未来可能全部移到这里)

* [关于训练,通用篇](./docs/train_README-ja.md): 数据准备和选项等
    * [数据集设置](./docs/config_README-ja.md)
* [DreamBooth训练指南](./docs/train_db_README-ja.md) 
* [fine-tuning指南](./docs/fine_tune_README_ja.md)
* [LoRA训练指南](./docs/train_network_README-ja.md)
* [文本反转训练指南](./docs/train_ti_README-ja.md)
* [图像生成脚本](./docs/gen_img_README-ja.md)
* note.com [模型转换脚本](https://note.com/kohya_ss/n/n374f316fe4ad)

## Windows环境所需程序

需要Python 3.10.6和Git。

- Python 3.10.6: https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe
- git: https://git-scm.com/download/win

如果要在PowerShell中使用venv,需要按以下步骤更改安全设置:
(不仅仅是venv,使脚本可以执行。请注意。)

- 以管理员身份打开PowerShell
- 输入"Set-ExecutionPolicy Unrestricted",选择Y
- 关闭管理员PowerShell

## 在Windows环境下安装

下例中安装的是PyTorch 1.12.1/CUDA 11.6版。如果要使用CUDA 11.3或PyTorch 1.13,请适当修改。

(如果只显示"python",请将下例中的"python"改为"py")  

在普通(非管理员)PowerShell中依次执行以下命令:

```powershell
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

python -m venv venv
.\venv\Scripts\activate

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install --upgrade -r requirements.txt
pip install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl

cp .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
cp .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
cp .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py

accelerate config
```

在命令提示符中:

```bat
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

python -m venv venv 
.\venv\Scripts\activate

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install --upgrade -r requirements.txt
pip install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl

copy /y .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
copy /y .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
copy /y .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py

accelerate config
```

accelerate config的问题请按以下回答:
(如果要用bf16训练,最后一个问题选择bf16)

```
- 此计算机
- 不进行分布式训练  
- 否
- 否 
- 否
- 所有
- fp16
```

### PyTorch和xformers版本注意事项

在其他版本中训练可能失败。如果没有特殊原因,请使用指定版本。


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