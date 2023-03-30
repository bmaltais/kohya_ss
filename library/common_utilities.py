def is_valid_config(data):
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
