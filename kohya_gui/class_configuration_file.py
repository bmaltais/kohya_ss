import gradio as gr
import os
from .common_gui import list_files, scriptdir, create_refresh_button
from .custom_logging import setup_logging

# Set up logging
log = setup_logging()


class ConfigurationFile:
    """
    A class to handle configuration file operations in the GUI.
    """

    def __init__(
        self, headless: bool = False, config_dir: str = None, config: dict = {}
    ):
        """
        Initialize the ConfigurationFile class.

        Parameters:
        - headless (bool): Whether to run in headless mode.
        - config_dir (str): The directory for configuration files.
        """

        self.headless = headless

        self.config = config

        # Sets the directory for storing configuration files, defaults to a 'presets' folder within the script directory.
        self.current_config_dir = self.config.get(
            "config_dir", os.path.join(scriptdir, "presets")
        )

        # Initialize the GUI components for configuration.
        self.create_config_gui()

    def list_config_dir(self, path: str) -> list:
        """
        List directories in the data directory.

        Parameters:
        - path (str): The path to list directories from.

        Returns:
        - list: A list of directories.
        """
        self.current_config_dir = path if not path == "" else "."
        # Lists all .json files in the current configuration directory, used for populating dropdown choices.
        return list(list_files(self.current_config_dir, exts=[".json"], all=True))

    def create_config_gui(self) -> None:
        """
        Create the GUI for configuration file operations.
        """
        # Starts a new group in the GUI for better layout organization.
        with gr.Group():
            # Creates a row within the group to align elements horizontally.
            with gr.Row():
                # Dropdown for selecting or entering the name of a configuration file.
                self.config_file_name = gr.Dropdown(
                    label="Load/Save Config file",
                    choices=[self.config.get("config_dir", "")] + self.list_config_dir(self.current_config_dir),
                    value=self.config.get("config_dir", ""),
                    interactive=True,
                    allow_custom_value=True,
                )

                # Button to refresh the list of configuration files in the dropdown.
                create_refresh_button(
                    self.config_file_name,
                    lambda: None,  # Placeholder for potential future functionality.
                    lambda: {
                        "choices": [""] + self.list_config_dir(self.current_config_dir)
                    },
                    "open_folder_small",
                )

                # Buttons for opening, saving, and loading configuration files, displayed conditionally based on headless mode.
                self.button_open_config = gr.Button(
                    "📂",
                    elem_id="open_folder_small",
                    elem_classes=["tool"],
                    visible=(not self.headless),
                )
                self.button_save_config = gr.Button(
                    "💾",
                    elem_id="open_folder_small",
                    elem_classes=["tool"],
                )
                self.button_load_config = gr.Button(
                    "↩️ ",
                    elem_id="open_folder_small",
                    elem_classes=["tool"],
                )

            # Handler for change events on the configuration file dropdown, allowing dynamic update of choices.
            self.config_file_name.change(
                fn=lambda path: gr.Dropdown(choices=[""] + self.list_config_dir(path)),
                inputs=self.config_file_name,
                outputs=self.config_file_name,
                show_progress=False,
            )
