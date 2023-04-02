class CommonUtilities:
    file_filters = {
        "all": [("All files", "*.*")],
        "video": [("Video files", "*.mp4;*.avi;*.mkv;*.mov;*.flv;*.wmv")],
        "images": [("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.tiff")],
        "json": [("JSON files", "*.json")],
        "lora": [("LoRa files", "*.ckpt;*.pt;*.safetensors")],
        "directory": [],
    }

    def is_valid_config(self, data):
        # Check if the data is a dictionary
        if not isinstance(data, dict):
            return False

        # Add checks for expected keys and valid values
        # For example, check if 'use_8bit_adam' is a boolean
        if "use_8bit_adam" in data and not isinstance(data["use_8bit_adam"], bool):
            return False

        # Add more checks for other keys as needed

        # If all checks pass, return True
        return True
