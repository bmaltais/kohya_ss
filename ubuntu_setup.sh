#!/bin/bash
echo installing tk
sudo apt install python3-tk
python3 -m venv venv
source venv/bin/activate
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install --use-pep517 --upgrade -r requirements.txt
pip install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/linux/xformers-0.0.14.dev0-cp310-cp310-linux_x86_64.whl

accelerate config

echo -e "setup finished! run \e[0;92m./gui.sh\e[0m to start"
