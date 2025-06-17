import toml
from .common_gui import scriptdir
from .custom_logging import setup_logging

# Set up logging
log = setup_logging()


class KohyaSSGUIConfig:
    """
    A class to handle the configuration for the Kohya SS GUI.
    """

    def __init__(self, config_file_path: str = "./config.toml"):
        """
        Initialize the KohyaSSGUIConfig class.
        """
        self.config = self.load_config(config_file_path=config_file_path)

    def load_config(self, config_file_path: str = "./config.toml") -> dict:
        """
        Loads the Kohya SS GUI configuration from a TOML file.

        Returns:
        dict: The configuration data loaded from the TOML file.
        """
        try:
            # Attempt to load the TOML configuration file from the specified directory.
            config = toml.load(f"{config_file_path}")
            log.debug(f"Loaded configuration from {config_file_path}")
        except FileNotFoundError:
            # If the config file is not found, initialize `config` as an empty dictionary to handle missing configurations gracefully.
            config = {}
            log.debug(
                f"No configuration file found at {config_file_path}. Initializing empty configuration."
            )

        return config

    def save_config(self, config: dict, config_file_path: str = "./config.toml"):
        """
        Saves the Kohya SS GUI configuration to a TOML file.

        Parameters:
        - config (dict): The configuration data to save.
        """
        # Write the configuration data to the TOML file
        with open(f"{config_file_path}", "w", encoding="utf-8") as f:
            toml.dump(config, f)

    def get(self, key: str, default=None):
        """
        Retrieves the value of a specified key from the configuration data.

        Parameters:
        - key (str): The key to retrieve the value for.
        - default: The default value to return if the key is not found.

        Returns:
        The value associated with the key, or the default value if the key is not found.
        """
        # Split the key into a list of keys if it contains a dot (.)
        keys = key.split(".")
        # Initialize `data` with the entire configuration data
        data = self.config

        # Iterate over the keys to access nested values
        for k in keys:
            log.debug(k)
            # If the key is not found in the current data, return the default value
            if k not in data:
                log.debug(
                    f"Key '{key}' not found in configuration. Returning default value."
                )
                return default

            # Update `data` to the value associated with the current key
            data = data.get(k)

        # Return the final value
        log.debug(f"Returned {data}")
        return data

    def is_config_loaded(self) -> bool:
        """
        Checks if the configuration was loaded from a file.

        Returns:
        bool: True if the configuration was loaded from a file, False otherwise.
        """
        is_loaded = self.config != {}
        log.debug(f"Configuration was loaded from file: {is_loaded}")
        return is_loaded
