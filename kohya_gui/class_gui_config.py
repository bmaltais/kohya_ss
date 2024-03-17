import toml
from .common_gui import scriptdir
from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

class KohyaSSGUIConfig:
    """
    A class to handle the configuration for the Kohya SS GUI.
    """

    def __init__(self):
        """
        Initialize the KohyaSSGUIConfig class.
        """
        self.config = self.load_config()

    def load_config(self) -> dict:
        """
        Loads the Kohya SS GUI configuration from a TOML file.

        Returns:
        dict: The configuration data loaded from the TOML file.
        """
        try:
            # Attempt to load the TOML configuration file from the specified directory.
            config = toml.load(fr"{scriptdir}/config.toml")
        except FileNotFoundError:
            # If the config file is not found, initialize `config` as an empty dictionary to handle missing configurations gracefully.
            config = {}

        return config

    def get(self, key, default=None):
        """
        Retrieves the value of a specified key from the configuration data.

        Parameters:
        - key (str): The key to retrieve the value for.
        - default: The default value to return if the key is not found.

        Returns:
        The value associated with the key, or the default value if the key is not found.
        """
        if key not in self.config:
            log.debug(f"Key '{key}' not found in configuration. Returning default value.")
        return self.config.get(key, default)
