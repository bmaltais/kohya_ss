import argparse
from PIL import Image
import os
import numpy as np
from collections import defaultdict
import math

def process_images(input_folder, output_folder, group_size):
    # Step 1: Get all image paths and their aspect ratios
    images = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))]

    # Step 2: Sort images by aspect ratio
    sorted_images = sorted(images, key=lambda path: Image.open(path).size[0] / Image.open(path).size[1])

    # Step 3: Group images
    groups = [sorted_images[i:i+group_size] for i in range(0, len(sorted_images), group_size)]

    # Step 4: Process each group
    for i, group in enumerate(groups):
        print(f"Processing group {i+1} with {len(group)} images...")

        if len(group) > 0:
            aspect_ratios = []
            for path in group:
                with Image.open(path) as img:
                    width, height = img.size
                    aspect_ratios.append(width / height)

            # Average aspect ratio
            avg_aspect_ratio = np.mean(aspect_ratios)

            # Crop all images to have the average aspect ratio
            cropped_images = []
            for j, path in enumerate(group):
                with Image.open(path) as img:
                    print(f"  Processing image {j+1}: {path}")

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

                    cropped_images.append(img)

            # Find the largest dimensions among the cropped images
            max_width = max(img.width for img in cropped_images)
            max_height = max(img.height for img in cropped_images)

            # Resize all images to match these dimensions
            for j, img in enumerate(cropped_images):
                img = img.resize((max_width, max_height))

                # Ensure the output directory exists
                os.makedirs(output_folder, exist_ok=True)

                # Save the image with the new filename
                output_path = os.path.join(output_folder, f"group-{i+1}-image-{j+1}.jpg")
                print(f"  Saving processed image to {output_path}")
                img.save(output_path)

def main():
    parser = argparse.ArgumentParser(description='Process groups of images.')
    parser.add_argument('input_folder', type=str, help='Input folder containing images')
    parser.add_argument('output_folder', type=str, help='Output folder to store processed images')
    parser.add_argument('group_size', type=int, help='Number of images in each group')

    args = parser.parse_args()
    process_images(args.input_folder, args.output_folder, args.group_size)

if __name__ == "__main__":
    main()