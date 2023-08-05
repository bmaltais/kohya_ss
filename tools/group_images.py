import argparse
import shutil
from PIL import Image, ImageOps
import os
import numpy as np

from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()

class ImageProcessor:

    def __init__(self, input_folder, output_folder, group_size, include_subfolders, do_not_copy_other_files, pad, caption, caption_ext):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.group_size = group_size
        self.include_subfolders = include_subfolders
        self.do_not_copy_other_files = do_not_copy_other_files
        self.pad = pad
        self.caption = caption
        self.caption_ext = caption_ext
        self.image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.tiff')

    def get_image_paths(self):
        images = []
        if self.include_subfolders:
            for dirpath, dirnames, filenames in os.walk(self.input_folder):
                for filename in filenames:
                    if filename.endswith(self.image_extensions):
                        images.append(os.path.join(dirpath, filename))
        else:
            images = [os.path.join(self.input_folder, f) for f in os.listdir(self.input_folder) if f.endswith(self.image_extensions)]
        return images

    def group_images(self, images):
        sorted_images = sorted(images, key=lambda path: Image.open(path).size[0] / Image.open(path).size[1])
        groups = [sorted_images[i:i+self.group_size] for i in range(0, len(sorted_images), self.group_size)]
        return groups

    def process_group(self, group, group_index):
        if len(group) > 0:
            aspect_ratios = self.get_aspect_ratios(group)
            avg_aspect_ratio = np.mean(aspect_ratios)
            if self.pad:
                padded_images = self.pad_images(group, avg_aspect_ratio)
                self.resize_and_save_images(padded_images, group_index, group)
            else:
                cropped_images = self.crop_images(group, avg_aspect_ratio)
                self.resize_and_save_images(cropped_images, group_index, group)
            if not self.do_not_copy_other_files:
                self.copy_other_files(group, group_index)

    def get_aspect_ratios(self, group):
        aspect_ratios = []
        for path in group:
            with Image.open(path) as img:
                width, height = img.size
                aspect_ratios.append(width / height)
        return aspect_ratios

    def crop_images(self, group, avg_aspect_ratio):
        cropped_images = []
        for j, path in enumerate(group):
            with Image.open(path) as img:
                log.info(f"  Processing image {j+1}: {path}")
                img = self.crop_image(img, avg_aspect_ratio)
                cropped_images.append(img)
        return cropped_images

    def crop_image(self, img, avg_aspect_ratio):
        img_aspect_ratio = img.width / img.height
        if img_aspect_ratio > avg_aspect_ratio:
            # Too wide, reduce width
            new_width = avg_aspect_ratio * img.height
            left = (img.width - new_width) / 2
            right = left + new_width
            img = img.crop((left, 0, right, img.height))
        else:
            # Too tall, reduce height
            new_height = img.width / avg_aspect_ratio
            top = (img.height - new_height) / 2
            bottom = top + new_height
            img = img.crop((0, top, img.width, bottom))
        return img

    def resize_and_save_images(self, cropped_images, group_index, source_paths):
        max_width = max(img.width for img in cropped_images)
        max_height = max(img.height for img in cropped_images)
        for j, img in enumerate(cropped_images):
            img = img.resize((max_width, max_height))
            os.makedirs(self.output_folder, exist_ok=True)
            original_filename = os.path.basename(source_paths[j])
            filename_without_ext = os.path.splitext(original_filename)[0]
            final_file_name = f"group-{group_index+1}-{j+1}-{filename_without_ext}"
            output_path = os.path.join(self.output_folder, f"{final_file_name}.jpg")
            log.info(f"  Saving processed image to {output_path}")
            img.convert('RGB').save(output_path, quality=70)
            
            if self.caption:
                self.create_caption_file(source_paths[j], group_index, final_file_name)

    def create_caption_file(self, source_path, group_index, caption_filename):
        dirpath = os.path.dirname(source_path)
        caption = os.path.basename(dirpath).split('_')[-1]
        caption_filename = caption_filename + self.caption_ext
        caption_path = os.path.join(self.output_folder, caption_filename)
        with open(caption_path, 'w') as f:
            f.write(caption)


    def copy_other_files(self, group, group_index):
        for j, path in enumerate(group):
            dirpath, original_filename = os.path.split(path)
            original_basename, original_ext = os.path.splitext(original_filename)
            for filename in os.listdir(dirpath):
                if filename.endswith('.npz'):  # Skip .npz
                    continue
                basename, ext = os.path.splitext(filename)
                if basename == original_basename and ext != original_ext:
                    shutil.copy2(os.path.join(dirpath, filename), os.path.join(self.output_folder, f"group-{group_index+1}-{j+1}-{filename}"))

    def process_images(self):
        images = self.get_image_paths()
        groups = self.group_images(images)
        for i, group in enumerate(groups):
            log.info(f"Processing group {i+1} with {len(group)} images...")
            self.process_group(group, i)
            
    def process_group(self, group, group_index):
        if len(group) > 0:
            aspect_ratios = self.get_aspect_ratios(group)
            avg_aspect_ratio = np.mean(aspect_ratios)
            if self.pad:
                padded_images = self.pad_images(group, avg_aspect_ratio)
                self.resize_and_save_images(padded_images, group_index, group)
            else:
                cropped_images = self.crop_images(group, avg_aspect_ratio)
                self.resize_and_save_images(cropped_images, group_index, group)
            if not self.do_not_copy_other_files:
                self.copy_other_files(group, group_index)

    def pad_images(self, group, avg_aspect_ratio):
        padded_images = []
        for j, path in enumerate(group):
            with Image.open(path) as img:
                log.info(f"  Processing image {j+1}: {path}")
                img = self.pad_image(img, avg_aspect_ratio)
                padded_images.append(img)
        return padded_images

    def pad_image(self, img, avg_aspect_ratio):
        img_aspect_ratio = img.width / img.height
        if img_aspect_ratio < avg_aspect_ratio:
            # Too tall, increase width
            new_width = avg_aspect_ratio * img.height
            pad_width = int((new_width - img.width) / 2)
            img = ImageOps.expand(img, border=(pad_width, 0), fill='black')
        else:
            # Too wide, increase height
            new_height = img.width / avg_aspect_ratio
            pad_height = int((new_height - img.height) / 2)
            img = ImageOps.expand(img, border=(0, pad_height), fill='black')
        return img

def main():
    parser = argparse.ArgumentParser(description='Process groups of images.')
    parser.add_argument('input_folder', type=str, help='Input folder containing images')
    parser.add_argument('output_folder', type=str, help='Output folder to store processed images')
    parser.add_argument('group_size', type=int, help='Number of images in each group')
    parser.add_argument('--include_subfolders', action='store_true', help='Include subfolders in search for images')
    parser.add_argument('--do_not_copy_other_files', '--no_copy', dest='do_not_copy_other_files', action='store_true', help='Do not copy other files with the same name as images')
    parser.add_argument('--pad', action='store_true', help='Pad images instead of cropping them')
    parser.add_argument('--caption', action='store_true', help='Create a caption file for each image')
    parser.add_argument('--caption_ext', type=str, default='.txt', help='Extension for the caption file')

    args = parser.parse_args()

    processor = ImageProcessor(args.input_folder, args.output_folder, args.group_size, args.include_subfolders, args.do_not_copy_other_files, args.pad, args.caption, args.caption_ext)
    processor.process_images()

if __name__ == "__main__":
    main()
