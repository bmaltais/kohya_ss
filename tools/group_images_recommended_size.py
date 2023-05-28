import argparse
from PIL import Image
import os
import numpy as np
import itertools

class ImageProcessor:

    def __init__(self, input_folder, min_group, max_group, include_subfolders, pad):
        self.input_folder = input_folder
        self.min_group = min_group
        self.max_group = max_group
        self.include_subfolders = include_subfolders
        self.pad = pad
        self.image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.webp')
        self.losses = []  # List to store loss values for each image

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

    def group_images(self, images, group_size):
        sorted_images = sorted(images, key=lambda path: Image.open(path).size[0] / Image.open(path).size[1])
        groups = [sorted_images[i:i+group_size] for i in range(0, len(sorted_images), group_size)]
        return groups

    def process_group(self, group):
        if len(group) > 0:
            aspect_ratios = self.get_aspect_ratios(group)
            avg_aspect_ratio = np.mean(aspect_ratios)
            self.calculate_losses(group, avg_aspect_ratio)

    def get_aspect_ratios(self, group):
        aspect_ratios = []
        for path in group:
            with Image.open(path) as img:
                width, height = img.size
                aspect_ratios.append(width / height)
        return aspect_ratios

    def calculate_losses(self, group, avg_aspect_ratio):
        for j, path in enumerate(group):
            with Image.open(path) as img:
                loss = self.calculate_loss(img, avg_aspect_ratio)
                self.losses.append((path, loss))  # Add (path, loss) tuple to the list

    def calculate_loss(self, img, avg_aspect_ratio):
        img_aspect_ratio = img.width / img.height
        if img_aspect_ratio > avg_aspect_ratio:
            # Too wide, reduce width
            new_width = avg_aspect_ratio * img.height
            loss = abs(img.width - new_width) / img.width  # Calculate loss value
        else:
            # Too tall, reduce height
            new_height = img.width / avg_aspect_ratio
            loss = abs(img.height - new_height) / img.height  # Calculate loss value
        return loss

    def monte_carlo_optimization(self, groups):
        best_groups = groups.copy()
        best_loss = np.inf
        best_removed_images = []

        for group in groups:
            num_images = len(group)
            all_combinations = []
            # Generate all possible combinations of images to remove
            for r in range(1, num_images + 1):
                combinations = list(itertools.combinations(group, r))
                all_combinations.extend(combinations)

            for combination in all_combinations:
                self.losses = []  # Reset losses for each combination
                remaining_images = list(set(group) - set(combination))
                self.process_group(remaining_images)
                avg_loss = np.mean(self.losses)

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_groups[best_groups.index(group)] = remaining_images
                    best_removed_images = combination

        return best_groups, best_loss, best_removed_images

    def process_images(self):
        images = self.get_image_paths()
        num_images = len(images)
        results = []

        for group_size in range(self.min_group, self.max_group + 1):
            groups = self.group_images(images, group_size)
            optimized_groups, avg_loss, removed_images = self.monte_carlo_optimization(groups)
            num_remaining = num_images % group_size

            results.append((group_size, avg_loss, num_remaining, optimized_groups, removed_images))

        # Sort results based on average crop loss in ascending order
        sorted_results = sorted(results, key=lambda x: x[1])

        for group_size, avg_loss, num_remaining, optimized_groups, removed_images in sorted_results:
            print(f"Group size: {group_size}, Average crop loss: {avg_loss}, Number of images remaining: {num_remaining}")
            print(f"Optimized Groups: {optimized_groups}")
            print(f"Removed Images: {removed_images}")


def main():
    parser = argparse.ArgumentParser(description='Process groups of images.')
    parser.add_argument('input_folder', type=str, help='Input folder containing images')
    parser.add_argument('min_group', type=int, help='Minimum group size')
    parser.add_argument('max_group', type=int, help='Maximum group size')
    parser.add_argument('--include_subfolders', action='store_true', help='Include subfolders in search for images')
    parser.add_argument('--pad', action='store_true', help='Pad images instead of cropping them')

    args = parser.parse_args()

    processor = ImageProcessor(args.input_folder, args.min_group, args.max_group, args.include_subfolders, args.pad)
    processor.process_images()


if __name__ == "__main__":
    main()
