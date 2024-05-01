# This code sorts a collection of images in a given directory by their aspect ratio, groups 
# them into batches of a given size, crops each image in a batch to the average aspect ratio 
# of that batch, and saves the cropped images in a specified directory. The user provides 
# the paths to the input directory and the output directory, as well as the desired batch 
# size. The program drops any images that do not fit exactly into the batches.

import os
import cv2
import argparse
import shutil

def aspect_ratio(img_path):
    """
    Calculate and return the aspect ratio of an image.
    
    Parameters:
    img_path: A string representing the path to the input image.
    
    Returns:
    float: Aspect ratio of the input image, defined as width / height.
           Returns None if the image cannot be read.
    """
    try:
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError("Image not found or could not be read.")
        height, width = image.shape[:2]
        return float(width) / float(height)
    except Exception as e:
        print(f"Error: {e}")
        return None

def sort_images_by_aspect_ratio(path):
    """Sort all images in a folder by aspect ratio"""
    images = []
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".webp"):
            print(filename)
            img_path = os.path.join(path, filename)
            images.append((img_path, aspect_ratio(img_path)))
    # sort the list of tuples based on the aspect ratio
    sorted_images = sorted(images, key=lambda x: x[1])
    return sorted_images

def create_groups(sorted_images, n_groups):
    """
    Create groups of images from a sorted list of images.

    This function takes a sorted list of images and a group size as input, and returns a list of groups,
    where each group contains a specified number of images.

    Parameters:
    sorted_images (list of tuples): A list of tuples, where each tuple contains the path to an image and its aspect ratio.
    n_groups (int): The number of images to include in each group.

    Returns:
    list of lists: A list of groups, where each group is a list of tuples representing the images in the group.

    Raises:
    ValueError: If the group size is not a positive integer or if the group size is greater than the number of images.
    """
    if not isinstance(n_groups, int) or n_groups <= 0:
        raise ValueError("Error: n_groups must be a positive integer.")
    if n_groups > len(sorted_images):
        raise ValueError("Error: n_groups must be less than or equal to the number of images.")
    n = len(sorted_images)
    size = n // n_groups
    groups = [sorted_images[i * size : (i + 1) * size] for i in range(n_groups - 1)]
    groups.append(sorted_images[(n_groups - 1) * size:])
    return groups

def average_aspect_ratio(group):
    """
    Calculate the average aspect ratio for a given group of images.

    Parameters:
    group (list of tuples):, A list of tuples, where each tuple contains the path to an image and its aspect ratio.

    Returns:
    float: The average aspect ratio of the images in the group.
    """
    if not group:
        print("Error: The group is empty")
        return None
    
    try:
        aspect_ratios = [aspect_ratio for _, aspect_ratio in group]
        avg_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)
        print(f"Average aspect ratio for group: {avg_aspect_ratio}")
        return avg_aspect_ratio
    except TypeError:
        print("Error: Check the structure of the input group elements. They should be tuples of (image_path, aspect_ratio).")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def center_crop_image(image, target_aspect_ratio):
    """Crop the input image to the target aspect ratio.

    The function calculates the crop region for the input image based on its current aspect ratio and the target aspect ratio.

    Args:
        image: A numpy array representing the input image.
        target_aspect_ratio: A float representing the target aspect ratio.

    Returns:
        A numpy array representing the cropped image.
        
    Raises:
        ValueError: If the input image is not a valid numpy array with at least two dimensions or if the calculated new width or height is zero.

    """
    # Check if the input image is a valid numpy array with at least two dimensions
    if not isinstance(image, np.ndarray) or image.ndim < 2:
        raise ValueError("Input image must be a valid numpy array with at least two dimensions.")

    height, width = image.shape[:2]
    current_aspect_ratio = float(width) / float(height)

    # If the current aspect ratio is already equal to the target aspect ratio, return the image as is
    if current_aspect_ratio == target_aspect_ratio:
        return image

    # Calculate the new width and height based on the target aspect ratio
    if current_aspect_ratio > target_aspect_ratio:
        new_width = int(target_aspect_ratio * height)
        if new_width == 0:
            raise ValueError("Calculated new width is zero. Please check the input image and target aspect ratio.")
        x_start = (width - new_width) // 2
        cropped_image = image[:, x_start:x_start+new_width]
    else:
        new_height = int(width / target_aspect_ratio)
        if new_height == 0:
            raise ValueError("Calculated new height is zero. Please check the input image and target aspect ratio.")
        y_start = (height - new_height) // 2
        cropped_image = image[y_start:y_start+new_height, :]

    return cropped_image

def copy_related_files(img_path, save_path):
    """
    Copy all files in the same directory as the input image that have the same base name as the input image to the
    output directory with the corresponding new filename.

    Args:
        img_path (str): Path to the input image file.
        save_path: Path to the output directory where the files should be copied with a new name.
    """
    # Get the base filename and directory
    img_dir, img_basename = os.path.split(img_path)
    img_base, img_ext = os.path.splitext(img_basename)

    save_dir, save_basename = os.path.split(save_path)
    save_base, save_ext = os.path.splitext(save_basename)

    # Create the output directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Loop over all files in the same directory as the input image
    try:
        for filename in os.listdir(img_dir):
            # Skip files with the same name as the input image
            if filename == img_basename:
                continue

            # Check if the file has the same base name as the input image
            file_base, file_ext = os.path.splitext(filename)
            if file_base == img_base:
                # Build the new filename and copy the file
                new_filename = os.path.join(save_dir, f"{save_base}{file_ext}")
                shutil.copy2(os.path.join(img_dir, filename), new_filename)
    except OSError as e:
        print(f"Error: {e}")  # Handle errors from os.listdir()

def save_resized_cropped_images(group, folder_name, group_number, avg_aspect_ratio, use_original_name=False):
    """Crop and resize all images in the input group to the smallest resolution, and save them to a folder.

    Args:
        group: A list of tuples, where each tuple contains the path to an image and its aspect ratio.
        folder_name: A string representing the name of the folder to save the images to.
        group_number: An integer representing the group number.
        avg_aspect_ratio: A float representing the average aspect ratio of the images in the group.
        use_original_name: A boolean indicating whether to save the images with their original file names.

    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # get the smallest size of the images
    smallest_res = float("inf")
    for img_path, _ in group:
        image = cv2.imread(img_path)
        cropped_image = center_crop_image(image, avg_aspect_ratio)
        height, width = cropped_image.shape[:2]
        image_res = height * width
        if image_res < smallest_res:
            smallest_res = image_res
            small_height, small_width = height, width

    # resize all images to the smallest resolution of the images in the group
    for i, (img_path, aspect_ratio) in enumerate(group):
        image = cv2.imread(img_path)
        cropped_image = center_crop_image(image, avg_aspect_ratio)
        # resized_image = cv2.resize(cropped_image, (small_width, small_height))
        if use_original_name:
            save_name = os.path.basename(img_path)
        else:
            save_name = f"group_{group_number}_{i}.jpg"
        save_path = os.path.join(folder_name, save_name)
        cv2.imwrite(save_path, cropped_image)
        
        # Copy matching files named the same as img_path to
        copy_related_files(img_path, save_path)
        
        print(f"Saved {save_name} to {folder_name}")
        

def main():
    parser = argparse.ArgumentParser(description='Sort images and crop them based on aspect ratio')
    parser.add_argument('input_dir', type=str, help='Path to the directory containing images')
    parser.add_argument('output_dir', type=str, help='Path to the directory to save the cropped images')
    parser.add_argument('batch_size', type=int, help='Size of the batches to create')
    parser.add_argument('--use_original_name', action='store_true', help='Whether to use original file names for the saved images')

    args = parser.parse_args()

    print(f"Sorting images by aspect ratio in {args.input_dir}...")
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return

    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except OSError:
            print(f"Error: Failed to create output directory: {args.output_dir}")
            return

    sorted_images = sort_images_by_aspect_ratio(args.input_dir)
    total_images = len(sorted_images)
    print(f'Total images: {total_images}')

    if args.batch_size <= 0:
        print("Error: Batch size must be greater than 0")
        return
    
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
    print(f"Created {len(groups)} groups")

    print('Saving cropped and resize images...')
    for i, group in enumerate(groups):
        avg_aspect_ratio = average_aspect_ratio(group)
        print(f"Processing group {i+1} with {len(group)} images...")
        try:
            save_resized_cropped_images(group, args.output_dir, i+1, avg_aspect_ratio, args.use_original_name)
        except Exception as e:
            print(f"Error: Failed to save images in group {i+1}: {e}")

    print('Done')

if __name__ == '__main__':
    main()