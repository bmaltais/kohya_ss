import os
import cv2
import argparse
import shutil
import math

def resize_images(src_img_folder, dst_img_folder, max_resolution="512x512", divisible_by=2):
    # Calculate max_pixels from max_resolution string
    max_pixels = int(max_resolution.split("x")[0]) * int(max_resolution.split("x")[1])
    
    # Create destination folder if it does not exist
    if not os.path.exists(dst_img_folder):
        os.makedirs(dst_img_folder)
    
    # Iterate through all files in src_img_folder
    for filename in os.listdir(src_img_folder):
        # Check if the image is png, jpg or webp
        if not filename.endswith(('.png', '.jpg', '.webp')):
            # Copy the file to the destination folder if not png, jpg or webp
            shutil.copy(os.path.join(src_img_folder, filename), os.path.join(dst_img_folder, filename))
            continue

        # Load image
        img = cv2.imread(os.path.join(src_img_folder, filename))
        
        # Calculate current number of pixels
        current_pixels = img.shape[0] * img.shape[1]
        
        # Check if the image needs resizing
        if current_pixels > max_pixels:
            # Calculate scaling factor
            scale_factor = max_pixels / current_pixels
            
            # Calculate new dimensions
            new_height = int(img.shape[0] * math.sqrt(scale_factor))
            new_width = int(img.shape[1] * math.sqrt(scale_factor))
            
            # Resize image
            img = cv2.resize(img, (new_width, new_height))
            
            # Calculate the new height and width that are divisible by divisible_by
            new_height = new_height if new_height % divisible_by == 0 else new_height - new_height % divisible_by
            new_width = new_width if new_width % divisible_by == 0 else new_width - new_width % divisible_by
            
            # Center crop the image to the calculated dimensions
            y = int((img.shape[0] - new_height) / 2)
            x = int((img.shape[1] - new_width) / 2)
            img = img[y:y + new_height, x:x + new_width]
        
        # Save resized image in dst_img_folder
        cv2.imwrite(os.path.join(dst_img_folder, filename), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
        print(f"Resized image: {filename} with size {img.shape[0]}x{img.shape[1]}")
        
        
def main():
    parser = argparse.ArgumentParser(description='Resize images in a folder to a specified max resolution')
    parser.add_argument('src_img_folder', type=str, help='Source folder containing the images')
    parser.add_argument('dst_img_folder', type=str, help='Destination folder to save the resized images')
    parser.add_argument('--max_resolution', type=str, help='Maximum resolution in the format "512x512"', default="512x512")
    parser.add_argument('--divisible_by', type=int, help='Ensure new dimensions are divisible by this value', default=2)
    args = parser.parse_args()
    resize_images(args.src_img_folder, args.dst_img_folder, args.max_resolution)

if __name__ == '__main__':
    main()