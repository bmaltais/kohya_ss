import os.path

from PIL import Image
import webuiapi
from webuiapi import ControlNetUnit
import yaml
import hashlib
import shutil


class CreatePhotoResult:
    def __init__(self, generated_photos):
        self.generated_photos = generated_photos


class CreatePhoto:
    def __init__(self):
        with open("config.yaml", 'r') as f:  # 用with读取文件更好
            configs = yaml.load(f, Loader=yaml.FullLoader)  # 按字典格式读取并返回
        create_configs = configs['create_config']
        print(create_configs)
        self.port = create_configs['port']
        self.Extrax = create_configs['Extrax']
        self.SStep = create_configs['SStep']
        self.CFG = create_configs['CFG']
        self.DS = create_configs['DS']
        self.create_width = create_configs['create_width']
        self.create_height = create_configs['create_height']
        ### controlnet
        self.Controlnet = create_configs['Controlnet']
        self.con_module = create_configs['con_module']
        self.guidance_start = create_configs['guidance_start']
        self.guidance_end = create_configs['guidance_end']
        self.NPrompt = create_configs['NPrompt']
        self.api = webuiapi.WebUIApi(
            host="127.0.0.1",
            port=self.port
        )

        self.loras = []
        options = {}
        options['sd_model_checkpoint'] = 'epicrealism_pureEvolutionV3.safetensors'
        res = self.api.set_options(options)

    # create_list = [
    #     {'style_photo': path_to_style_photo1, "background_image": background_image, 'promote': 'style1',
    #      'quantity': 5},
    #     {'style_photo': path_to_style_photo1, "background_image": background_image, 'promote': 'style2',
    #      'quantity': 3}
    # ]

    def generate_photos(self, user_info, main_photo,
                        create_list: [{}, {}, {}],
                        progress_temp_dir='',
                        result_save_dir='', lora_path=''):

        lora_path_model_name = os.path.basename(lora_path)
        if self.__get_lora__(lora_path) == -1:
            return None

        TargetPhotoList = []
        BackGroundList = []
        controlnets = []

        controlnet = ControlNetUnit(input_image=main_photo,
                                    module=self.con_module,
                                    model=self.Controlnet,
                                    weight=1.0,
                                    resize_mode="Resize and Fill",
                                    lowvram=False,
                                    processor_res=512,
                                    threshold_a=64,
                                    threshold_b=64,
                                    guidance=1.0,
                                    guidance_start=self.guidance_start,
                                    guidance_end=self.guidance_end,
                                    control_mode=0,
                                    pixel_perfect=True,
                                    guessmode=None,  # deprecated: use control_mode
                                    )
        controlnets.append(controlnet)
        for style in create_list:
            BackGroundList.append(style["background_image"])
            pic_count = style["quantity"]
            image_style_prompt = style["promote"]
            prompt = user_info["promote"] + image_style_prompt + f"<lora:{user_info['user_id']}:0.6>"
            negative_prompt = self.NPrompt
            print(negative_prompt)
            print(prompt)
            generate_photos_list = []

            for i in range(pic_count):
                result = self.api.txt2img(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    sampler_index="DPM++ 2M SDE K arras",
                    steps=self.SStep,
                    cfg_scale=self.CFG,
                    width=self.create_width,
                    height=self.create_height,
                    denoising_strength=self.DS,
                    controlnet_units=controlnets
                )
                result_image = result.image

                if self.Extrax != None:
                    width, height = result_image.size
                    # 计算宽度和高度的比例
                    aspect_ratio = width / height
                    if (width > height):
                        new_width = self.Extrax
                        new_height = int(self.Extrax / aspect_ratio)
                    else:
                        new_height = self.Extrax
                        new_width = int(self.Extrax * aspect_ratio)

                    extra_result = self.api.extra_single_image(result_image,  # PIL Image
                                                               resize_mode=1,
                                                               show_extras_results=True,
                                                               gfpgan_visibility=0,
                                                               codeformer_visibility=0,
                                                               codeformer_weight=0,
                                                               upscaling_resize=2,
                                                               upscaling_resize_w=new_width,
                                                               upscaling_resize_h=new_height,
                                                               upscaling_crop=True,
                                                               upscaler_1="R-ESRGAN 4x+",
                                                               upscaler_2="R-ESRGAN 4x+",
                                                               extras_upscaler_2_visibility=0,
                                                               upscale_first=False,
                                                               use_async=False,
                                                               )
                    extra_result = extra_result.image
                else:
                    extra_result = result_image

                generate_photos_list.append(extra_result)
            TargetPhotoList.append(generate_photos_list)

        print(TargetPhotoList)

        """
        此处调用照片后处理接口 将MainPhoto融合到TargetPhotoList上
        """
        ResultList = self.post_process_photos(main_photo, TargetPhotoList)

        """"
        此处调用照片后处理接口 将BackGroundList[index]融合到ResultList上
        """
        final_result_list = self.merge_photos_with_background(BackGroundList, ResultList)

        # 如果指定了生成进度临时目录，保存生成进度（格式待定）
        if progress_temp_dir:
            # 假设保存生成进度的逻辑
            progress_file_path = 'path_to_progress_file'

        # 如果指定了生成结果保存目录，保存生成结果
        if result_save_dir:
            # 假设保存生成结果的逻辑
            result_save_dir = 'path_to_result_save_dir'

        # 创建 CreatePhotoResult 对象，包含最终的生成结果
        result = CreatePhotoResult(final_result_list)

        # 返回生成结果
        return result

    def post_process_photos(self, main_photo, generated_photos):
        # 在这里进行照片后处理的逻辑
        pass
        # 假设返回处理后的照片列表
        return generated_photos

    def merge_photos_with_background(self, backgrounds, TargetPhotoList):
        # 在这里进行照片背景融合的操作
        pass
        # 假设返回最终的生成结果列表
        return TargetPhotoList

    def __refresh_loras__(self):
        if self.api.refresh_loras() is None:
            loras = self.api.get_loras()
            self.loras = []
            for lora in loras:
                self.loras.append(lora['name'] + ".safetensors")

    def __get_lora__(self, path):
        try:
            with open(path, 'rb') as fp:
                data = fp.read()
                remote_lora_md5 = hashlib.md5(data).hexdigest()
        except:
            return -1

        self.__refresh_loras__()
        if os.path.basename(path) not in self.loras:
            shutil.copy(path, "../stable-diffusion-webui/models/Lora")
        else:
            lora_path = os.path.join("../stable-diffusion-webui/models/Lora", os.path.basename(path))
            with open(lora_path, 'rb') as fp:
                data = fp.read()
                my_lora_md5 = hashlib.md5(data).hexdigest()

            if remote_lora_md5 != my_lora_md5:
                shutil.copy(path, "../stable-diffusion-webui/models/Lora")

        self.__refresh_loras__()
        print(os.path.basename(path))
        if os.path.basename(path) not in self.loras:
            return -1
        return 1
