import os
from PIL import Image
from TrainModel import *

# 示例用法
if __name__ == '__main__':
    # 创建 TrainModel 对象
    trainer = TrainModel()
    # 准备训练所需的参数
    user_info = {'user_id': 123456, 'promote': 'feature_value'}


    main_photo_path = 'lllll/TfwVAbyX_400x400.jpg'
    main_photo = Image.open(main_photo_path)
    train_photo_list = []

    for path in os.listdir("2222"):
        picpath = os.path.join("2222", path)
        train_photo_list.append(Image.open(picpath))

    progress_dir = 'path_to_progress_directory'
    return_model = True  # 如果需要返回模型文件

    # 调用 train 方法进行模型训练
    print(type(train_photo_list[0]))

    training_result = trainer.train(user_info, main_photo, train_photo_list, progress_dir, return_model)

    # 打印训练结果
    print("User Info:", training_result.user_info)
    print("Model Save Path:", training_result.model_save_path)
    if training_result.model_file:
        print("Model File:", training_result.model_file)

