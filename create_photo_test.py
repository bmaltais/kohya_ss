from CreatePhoto import CreatePhoto
from PIL import Image

# 示例用法
if __name__ == '__main__':

    # 准备生成所需的参数
    user_info = {'user_id': "AnneH728-000005",
                 'promote': ' Generate an image of a person with a broad smile, eyes closed in joy"eyes closed, AnneHathaway,a woman in a white dress posing for a picture with her hands on her hips and her hands on her hips'}

    main_photo = Image.open('lllll/TfwVAbyX_400x400.jpg')

    style_photo1 = Image.open("lllll/TfwVAbyX_400x400.jpg")

    style_photo2 = Image.open("lllll/TfwVAbyX_400x400.jpg")

    background_image1 = Image.open("lllll/00012-0-1P616194618-4.png")
    background_image2 = Image.open("lllll/00012-0-1P616194618-4.png")

    create_list = [
        {'style_photo': style_photo1, "background_image": background_image1,
         'promote': 'black suit, white T-shirt,red tie,',
         'quantity': 5},
        {'style_photo': style_photo2, "background_image": background_image2, 'promote': 'white suit,blue tie',
         'quantity': 3}
    ]

    progress_temp_dir = 'path_to_progress_temp_directory'
    result_save_dir = 'generate_photos'

    # 创建 CreatePhoto 对象
    photo_creator = CreatePhoto()

    # 调用 generate_photos 方法进行图片生成
    generation_result = photo_creator.generate_photos(user_info, main_photo, create_list, progress_temp_dir,
                                                      result_save_dir,
                                                      lora_path="/home/tt/Downloads/LoRA/model/AnneH728-000005.safetensors")

    # 处理生成结果
    generated_photos = generation_result.generated_photos
    for idx, photos in enumerate(generated_photos):
        print(f"Generated Photo {idx}: {photos}")
        for index, photo in enumerate(photos):
            photo.save(f"{result_save_dir}/{idx}_{index}.png")
