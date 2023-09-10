import os
import argparse

def main(folder_path):
    # Validate if the folder exists
    if not os.path.exists(folder_path):
        print("The specified folder does not exist.")
        return
    
    # Loop through all files in the directory
    for filename in os.listdir(folder_path):
        # Check if the file is an image file (webp, jpg, png)
        if filename.lower().endswith(('.webp', '.jpg', '.png')):
            # Remove the file extension from the filename
            name_without_extension = os.path.splitext(filename)[0]
            
            # Construct the name of the txt file
            txt_filename = f"{name_without_extension}.txt"
            
            # Extract the content before the underscore
            content = name_without_extension.split("_")[0]
            
            # Write the content to the txt file
            with open(os.path.join(folder_path, txt_filename), "w") as txt_file:
                txt_file.write(content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a folder.')
    parser.add_argument('folder_path', type=str, help='Path to the folder to process')
    
    args = parser.parse_args()
    main(args.folder_path)
