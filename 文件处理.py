import os
from PIL import Image  
import shutil

src_dir = 'O:\data1016\starkei'
dst_dir = 'O:\data1016\starkei_512'
size = (512, 512) # 目标图像大小

for filename in os.listdir(src_dir):
    name, ext = os.path.splitext(filename)
    if ext.lower() in ['.jpg','.png']:
        img_file = os.path.join(src_dir, filename) 
        img = Image.open(img_file)
        # img_resized = img.thumbnail(size, Image.LANCZOS)
        img.thumbnail(size,Image.LANCZOS)
        img.save(os.path.join(dst_dir, filename))
        # img_resized.save(os.path.join(dst_dir, filename))
        
    if ext.lower() == '.txt':
        shutil.copy(os.path.join(src_dir, filename), dst_dir)