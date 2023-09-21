import os, yaml
from fastapi import FastAPI
from PIL import Image


class TrainModelResult:
    def __init__(self, user_info: dict, model_save_path, model_file=None):
        self.user_info = user_info
        self.model_save_path = model_save_path
        self.model_file = model_file


class TrainModel:
    def __init__(self):
        with open("config.yaml", 'r') as f:  # 用with读取文件更好
            configs = yaml.load(f, Loader=yaml.FullLoader)  # 按字典格式读取并返回
        train_configs = configs["train_config"]
        self.max_train_steps = train_configs["max_train_steps"]

        print(configs)

    def __save_train_picture_to_dir__(self, train_photo_list):
        for index, pic in enumerate(train_photo_list):
            pic.save(f"train_pictures/100_train/{index}.jpg")

    def train(self, user_info, main_photo: Image, train_photo_list: [Image, Image], progress_dir='',
              return_model=False, max_step=5000):

        self.__save_train_picture_to_dir__(train_photo_list)
        # blip 打标签
        bcmd = ('accelerate launch "finetune/api_make_captions.py" '
                '--batch_size="1" --num_beams="1" '
                '--top_p="0.9" '
                '--max_length="75" '
                '--min_length="10"  '
                '--beam_search    '
                '--caption_extension=".txt" '
                '"train_pictures/100_train" '
                '--caption_weights="https://storage.googleapis.com/sfr-vision-language'
                '-research/BLIP/models/model_large_caption.pth" '
                f'--user_id {user_info["user_id"]}')

        os.system(bcmd)

        tcmd = (r'accelerate launch  --num_cpu_threads_per_process=16 "./train_network.py" '
                r'--enable_bucket --min_bucket_reso=256 --max_bucket_reso=2048 '
                r'--pretrained_model_name_or_path="../stable-diffusion-webui/models/Stable-diffusion/epicrealism_pureEvolutionV3.safetensors" '
                r'--train_data_dir="train_pictures" '
                r'--resolution="512,512" '
                r'--output_dir="../stable-diffusion-webui/models/Lora" '
                r'--logging_dir="logs" '
                r'--network_alpha="128" '
                r'--save_model_as=safetensors '
                r'--network_module=networks.lora '
                r'--text_encoder_lr=0.0001 '
                r'--unet_lr=0.0001 '
                r'--network_dim=128 '
                r'--gradient_accumulation_steps=4 '
                r'--output_name="hsw" '
                r'--lr_scheduler_num_cycles="10" '
                r'--scale_weight_norms="1" '
                r'--no_half_vae -'
                r'-learning_rate="0.0001" '
                r'--lr_scheduler="cosine_with_restarts" '
                r'--lr_warmup_steps="675" '
                r'--train_batch_size="1" '
                f'--max_train_steps="{max_step}" '
                r'--save_every_n_epochs="1" '
                r'--mixed_precision="fp16" '
                r'--save_precision="fp16" '
                r'--caption_extension=".none-use-foldername" '
                r'--cache_latents '
                r'--cache_latents_to_disk '
                r'--optimizer_type="AdamW" '
                r'--max_data_loader_n_workers="0" '
                r'--bucket_reso_steps=1 '
                r'--min_snr_gamma=10 --xformers '
                r'--bucket_no_upscale '
                r'--multires_noise_iterations="8" '
                r'--multires_noise_discount="0.2"')
        os.system(tcmd)
        # 假设模型训练完成后，得到了模型的本地保存路径 model_save_path
        model_save_path = 'path_to_saved_model'

        # 如果选择了返回模型的选项，加载模型文件
        model_file = None
        if return_model:
            # 假设加载模型的逻辑
            model_file = 'Lora_path'

        # 保存训练进度到本地文件（格式待定）
        if progress_dir:
            # 假设保存训练进度的逻辑
            progress_file_path = progress_dir

        # 创建 TrainModelResult 对象，包括用户信息、模型保存地址和模型文件（如果返回模型）
        result = TrainModelResult(user_info, model_save_path, model_file)

        # 返回训练结果
        return result
