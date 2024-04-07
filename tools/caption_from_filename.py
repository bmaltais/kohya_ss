# Proposed by https://github.com/kainatquaderee
import os
import argparse

def main(image_directory, output_directory, image_extension, text_extension):
    # Ensure the output directory exists, create it if necessary
    os.makedirs(output_directory, exist_ok=True)

    # Initialize a counter for the number of text files created
    text_files_created = 0

    # Iterate through files in the directory
    for image_filename in os.listdir(image_directory):
        # Check if the file is an image
        if any(image_filename.lower().endswith(ext) for ext in image_extension):
            # Extract prompt from filename
            prompt = os.path.splitext(image_filename)[0]

            # Construct path for the output text file
            text_file_path = os.path.join(output_directory, prompt + text_extension)

            # Write prompt to text file
            with open(text_file_path, 'w') as text_file:
                text_file.write(prompt)

            print(f"Text file saved: {text_file_path}")

            # Increment the counter
            text_files_created += 1

    # Report if no text files were created
    if text_files_created == 0:
        print("No image matching extensions were found in the specified directory. No caption files were created.")
    else:
        print(f"{text_files_created} text files created successfully.")

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Generate caption files from image filenames.')

    # Add arguments for the image directory, output directory, and file extension
    parser.add_argument('image_directory', help='Directory containing the image files')
    parser.add_argument('output_directory', help='Output directory where text files will be saved')
    parser.add_argument('--image_extension', nargs='+', default=['.jpg', '.jpeg', '.png', '.webp', '.bmp'], help='Extension for the image files')
    parser.add_argument('--text_extension', default='.txt', help='Extension for the output text files')

    # Parse the command-line arguments
    args = parser.parse_args()

    main(args.image_directory, args.output_directory, args.image_extension, args.text_extension)
