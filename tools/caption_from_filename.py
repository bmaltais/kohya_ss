# Proposed by https://github.com/kainatquaderee
import os
import argparse

def main(image_directory, output_directory):
    # Ensure the output directory exists, create it if necessary
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through files in the directory
    for image_filename in os.listdir(image_directory):
        # Extract prompt from filename
        prompt = os.path.splitext(image_filename)[0]

        # Construct path for the output text file
        text_file_path = os.path.join(output_directory, prompt + '.txt')

        # Write prompt to text file
        with open(text_file_path, 'w') as text_file:
            text_file.write(prompt)

        print(f"Text file saved: {text_file_path}")

    print("All text files saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate text files from image filenames.')
    parser.add_argument('image_directory', help='Directory containing the image files')
    parser.add_argument('output_directory', help='Output directory where text files will be saved')

    args = parser.parse_args()

    main(args.image_directory, args.output_directory)