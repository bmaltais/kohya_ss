import json
import argparse
import glob


def remove_items_with_keywords(json_file_path):
    keywords = [
        "caption_metadata_filename",
        "dir",
        "image_folder",
        "latent_metadata_filename",
        "logging_dir",
        "model_list",
        "output_dir",
        "output_name",
        "pretrained_model_name_or_path",
        "resume",
        "save_model_as",
        "save_state",
        "sample_",
        "train_dir",
        "wandb_api_key",
    ]

    with open(json_file_path) as file:
        data = json.load(file)

    for key in list(data.keys()):
        for keyword in keywords:
            if keyword in key:
                del data[key]
                break

    sorted_data = {k: data[k] for k in sorted(data)}

    with open(json_file_path, "w") as file:
        json.dump(sorted_data, file, indent=4)

    print(
        "Items with keywords have been removed from the JSON file and the list has been sorted alphabetically:",
        json_file_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove items from JSON files based on keywords in the keys"
    )
    parser.add_argument(
        "json_files", type=str, nargs="+", help="Path(s) to the JSON file(s)"
    )
    args = parser.parse_args()

    json_files = args.json_files
    for file_pattern in json_files:
        for json_file_path in glob.glob(file_pattern):
            remove_items_with_keywords(json_file_path)
