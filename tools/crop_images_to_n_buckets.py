# This code sorts a collection of images in a given directory by their aspect ratio, groups 
# them into batches of a given size, crops each image in a batch to the average aspect ratio 
# of that batch, and saves the cropped images in a specified directory. The user provides 
# the paths to the input directory and the output directory, as well as the desired batch 
# size. The program drops any images that do not fit exactly into the batches.

import os
import cv2
import argparse

def aspect_ratio(img_path):
    """Return aspect ratio of an image"""
    image = cv2.imread(img_path)
    height, width = image.shape[:2]
    aspect_ratio = float(width) / float(height)
    return aspect_ratio

def sort_images_by_aspect_ratio(path):
    """Sort all images in a folder by aspect ratio"""
    images = []
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            img_path = os.path.join(path, filename)
            images.append((img_path, aspect_ratio(img_path)))
    # sort the list of tuples based on the aspect ratio
    sorted_images = sorted(images, key=lambda x: x[1])
    return sorted_images

def create_groups(sorted_images, n_groups):
    """Create n groups from sorted list of images"""
    n = len(sorted_images)
    size = n // n_groups
    groups = [sorted_images[i * size : (i + 1) * size] for i in range(n_groups - 1)]
    groups.append(sorted_images[(n_groups - 1) * size:])
    return groups

def average_aspect_ratio(group):
    """Calculate average aspect ratio for a group"""
    aspect_ratios = [aspect_ratio for _, aspect_ratio in group]
    avg_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)
    return avg_aspect_ratio

def center_crop_image(image, target_aspect_ratio):
    height, width = image.shape[:2]
    current_aspect_ratio = float(width) / float(height)

    if current_aspect_ratio == target_aspect_ratio:
        return image

    if current_aspect_ratio > target_aspect_ratio:
        new_width = int(target_aspect_ratio * height)
        x_start = (width - new_width) // 2
        cropped_image = image[:, x_start:x_start+new_width]
    else:
        new_height = int(width / target_aspect_ratio)
        y_start = (height - new_height) // 2
        cropped_image = image[y_start:y_start+new_height, :]

    return cropped_image

def save_cropped_images(group, folder_name, group_number, avg_aspect_ratio):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # get the smallest size of the images
    small_height = 0
    small_width = 0
    smallest_res = 100000000
    for i, image in enumerate(group):
        img_path, aspect_ratio = image
        image = cv2.imread(img_path)
        cropped_image = center_crop_image(image, avg_aspect_ratio)
        height, width = cropped_image.shape[:2]
        if smallest_res > height * width:
            small_height = height
            small_width = width
            smallest_res = height * width

    # resize all images to the smallest resolution of the images in the group
    for i, image in enumerate(group):
        img_path, aspect_ratio = image
        image = cv2.imread(img_path)
        cropped_image = center_crop_image(image, avg_aspect_ratio)
        resized_image = cv2.resize(cropped_image, (small_width, small_height))
        save_path = os.path.join(folder_name, "group_{}_{}.jpg".format(group_number, i))
        cv2.imwrite(save_path, resized_image)
        

def main():
    parser = argparse.ArgumentParser(description='Sort images and crop them based on aspect ratio')
    parser.add_argument('--path', type=str, help='Path to the directory containing images', required=True)
    parser.add_argument('--dst_path', type=str, help='Path to the directory to save the cropped images', required=True)
    parser.add_argument('--batch_size', type=int, help='Size of the batches to create', required=True)

    args = parser.parse_args()

    sorted_images = sort_images_by_aspect_ratio(args.path)
    total_images = len(sorted_images)
    print(f'Total images: {total_images}')
    
    group_size = total_images // args.batch_size
    
    print(f'Train batch size: {args.batch_size}, image group size: {group_size}')
    remainder = total_images % args.batch_size

    if remainder != 0:
        print(f'Dropping {remainder} images that do not fit in groups...')
        sorted_images = sorted_images[:-remainder]
        total_images = len(sorted_images)
        group_size = total_images // args.batch_size

    print('Creating groups...')
    groups = create_groups(sorted_images, group_size)
    
    print('Saving cropped and resize images...')
    for i, group in enumerate(groups):
        avg_aspect_ratio = average_aspect_ratio(group)
        save_cropped_images(group, args.dst_path, i+1, avg_aspect_ratio)

if __name__ == '__main__':
    main()