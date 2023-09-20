import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import time
import os

start = time.time()
image_face_fusion = pipeline(Tasks.image_face_fusion,
                             model='damo/cv_unet-image-face-fusion_damo')
# names = []
# template_paths = []
# directory = '/home/tt/stable-diffusion-webui/outputs/txt2img-images/test'
# for i in range(1, 10):
#     test_path = os.path.join(directory, f'test{i}')
#     name = [path for path in os.listdir(test_path) if path.endswith('.png')]
#     template_path = [os.path.join(test_path, path) for path in os.listdir(test_path)]
#     template_paths.append(template_path)
#     names.append(name)
# print(names)
# print(template_paths)
# for i in range(1, 10):
#     os.mkdir(f"test{i}")
#     template_path = template_paths[i - 1]
#     name = names[i - 1]
#     user_path = f'/home/tt/stable-diffusion-webui/outputs/txt2img-images/test/{i}.png'
#     for index, path in enumerate(template_path):
#         result = image_face_fusion(dict(template=path, user=user_path))
#         cv2.imwrite(f"test{i}/{name[index]}.png", result[OutputKeys.OUTPUT_IMG])
# template_path = [os.path.join(directory, path) for path in os.listdir(directory)]
# user_path = '/home/tt/stable-diffusion-webui/outputs/txt2img-images/test'
#
# for index in range(1, 10):
#     print(index)
#     # result = image_face_fusion(dict(template=path, user=user_path))
#     #
#     # cv2.imwrite(f'result_{index}.png', result[OutputKeys.OUTPUT_IMG])
# print('finished!')
# end_time = time.time()
# print(end_time - start)

def swapface(image_face_fusion:pipeline,source:str,tenmle:[str,str,...],dir):
    for path in tenmle:
        result = image_face_fusion(dict(template=path, user=source))
        file_name = os.path.basename(path)
        print(dir+"/"+file_name)
        cv2.imwrite(dir+"/"+file_name, result[OutputKeys.OUTPUT_IMG])


if __name__ == '__main__':
    image_face_fusion = pipeline(Tasks.image_face_fusion,
                                 model='damo/cv_unet-image-face-fusion_damo')

    for i in range(1, 10):
        os.mkdir(f"test{i}")
        source = f'/home/tt/stable-diffusion-webui/outputs/txt2img-images/test/test{i}/{i}.png'
        dir = f"/home/tt/stable-diffusion-webui/outputs/txt2img-images/test/test{i}"
    # source = "/home/tt/stable-diffusion-webui/outputs/txt2img-images/test/test1/1.png"
    # dir = "/home/tt/stable-diffusion-webui/outputs/txt2img-images/test/test1"
        tenmle = [os.path.join(dir,path) for path in os.listdir(dir)]
        swapface(image_face_fusion,source,tenmle,dir= f'test{i}')