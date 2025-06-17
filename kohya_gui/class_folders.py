import gradio as gr
import os
from .common_gui import get_folder_path, scriptdir, list_dirs, create_refresh_button


class Folders:
    """
    A class to handle folder operations in the GUI.
    """

    def __init__(
        self, finetune: bool = False, headless: bool = False, config: dict = {}
    ):
        """
        Initialize the Folders class.

        Parameters:
        - finetune (bool): Whether to finetune the model.
        - headless (bool): Whether to run in headless mode.
        """
        self.headless = headless
        self.finetune = finetune

        # Load kohya_ss GUI configs from config.toml if it exist
        self.config = config

        # Set default directories if not provided
        self.current_output_dir = self.config.get(
            "output_dir", os.path.join(scriptdir, "outputs")
        )
        self.current_logging_dir = self.config.get(
            "logging_dir", os.path.join(scriptdir, "logs")
        )
        self.current_reg_data_dir = self.config.get(
            "reg_data_dir", os.path.join(scriptdir, "reg")
        )

        # Create directories if they don't exist
        self.create_directory_if_not_exists(self.current_output_dir)
        self.create_directory_if_not_exists(self.current_logging_dir)

        # Create the GUI for folder selection
        self.create_folders_gui()

    def create_directory_if_not_exists(self, directory: str) -> None:
        """
        Create a directory if it does not exist.

        Parameters:
        - directory (str): The directory to create.
        """
        if (
            directory is not None
            and directory.strip() != ""
            and not os.path.exists(directory)
        ):
            os.makedirs(directory, exist_ok=True)

    def list_output_dirs(self, path: str) -> list:
        """
        List directories in the output directory.

        Parameters:
        - path (str): The path to list directories from.

        Returns:
        - list: A list of directories.
        """
        self.current_output_dir = path if not path == "" else "."
        return list(list_dirs(path))

    def list_logging_dirs(self, path: str) -> list:
        """
        List directories in the logging directory.

        Parameters:
        - path (str): The path to list directories from.

        Returns:
        - list: A list of directories.
        """
        self.current_logging_dir = path if not path == "" else "."
        return list(list_dirs(path))

    def list_reg_data_dirs(self, path: str) -> list:
        """
        List directories in the regularization data directory.

        Parameters:
        - path (str): The path to list directories from.

        Returns:
        - list: A list of directories.
        """
        self.current_reg_data_dir = path if not path == "" else "."
        return list(list_dirs(path))

    def create_folders_gui(self) -> None:
        """
        Create the GUI for folder selection.
        """
        with gr.Row():
            # Output directory dropdown
            self.output_dir = gr.Dropdown(
                label="Output directory for trained model",
                choices=[self.config.get("folders.output_dir", "")] + self.list_output_dirs(self.current_output_dir),
                value=self.config.get("folders.output_dir", ""),
                interactive=True,
                allow_custom_value=True,
            )
            # Refresh button for output directory
            create_refresh_button(
                self.output_dir,
                lambda: None,
                lambda: {
                    "choices": [""] + self.list_output_dirs(self.current_output_dir)
                },
                "open_folder_small",
            )
            # Output directory button
            self.output_dir_folder = gr.Button(
                "📂",
                elem_id="open_folder_small",
                elem_classes=["tool"],
                visible=(not self.headless),
            )
            # Output directory button click event
            self.output_dir_folder.click(
                get_folder_path,
                outputs=self.output_dir,
                show_progress=False,
            )

            # Regularisation directory dropdown
            self.reg_data_dir = gr.Dropdown(
                label=(
                    "Regularisation directory (Optional. containing regularisation images)"
                    if not self.finetune
                    else "Train config directory (Optional. where config files will be saved)"
                ),
                choices=[self.config.get("folders.reg_data_dir", "")] + self.list_reg_data_dirs(self.current_reg_data_dir),
                value=self.config.get("folders.reg_data_dir", ""),
                interactive=True,
                allow_custom_value=True,
            )
            # Refresh button for regularisation directory
            create_refresh_button(
                self.reg_data_dir,
                lambda: None,
                lambda: {
                    "choices": [""] + self.list_reg_data_dirs(self.current_reg_data_dir)
                },
                "open_folder_small",
            )
            # Regularisation directory button
            self.reg_data_dir_folder = gr.Button(
                "📂",
                elem_id="open_folder_small",
                elem_classes=["tool"],
                visible=(not self.headless),
            )
            # Regularisation directory button click event
            self.reg_data_dir_folder.click(
                get_folder_path,
                outputs=self.reg_data_dir,
                show_progress=False,
            )
        with gr.Row():
            # Logging directory dropdown
            self.logging_dir = gr.Dropdown(
                label="Logging directory (Optional. to enable logging and output Tensorboard log)",
                choices=[self.config.get("folders.logging_dir", "")] + self.list_logging_dirs(self.current_logging_dir),
                value=self.config.get("folders.logging_dir", ""),
                interactive=True,
                allow_custom_value=True,
            )
            # Refresh button for logging directory
            create_refresh_button(
                self.logging_dir,
                lambda: None,
                lambda: {
                    "choices": [""] + self.list_logging_dirs(self.current_logging_dir)
                },
                "open_folder_small",
            )
            # Logging directory button
            self.logging_dir_folder = gr.Button(
                "📂",
                elem_id="open_folder_small",
                elem_classes=["tool"],
                visible=(not self.headless),
            )
            # Logging directory button click event
            self.logging_dir_folder.click(
                get_folder_path,
                outputs=self.logging_dir,
                show_progress=False,
            )

            # Change event for output directory dropdown
            self.output_dir.change(
                fn=lambda path: gr.Dropdown(choices=[""] + self.list_output_dirs(path)),
                inputs=self.output_dir,
                outputs=self.output_dir,
                show_progress=False,
            )
            # Change event for regularisation directory dropdown
            self.reg_data_dir.change(
                fn=lambda path: gr.Dropdown(
                    choices=[""] + self.list_reg_data_dirs(path)
                ),
                inputs=self.reg_data_dir,
                outputs=self.reg_data_dir,
                show_progress=False,
            )
            # Change event for logging directory dropdown
            self.logging_dir.change(
                fn=lambda path: gr.Dropdown(
                    choices=[""] + self.list_logging_dirs(path)
                ),
                inputs=self.logging_dir,
                outputs=self.logging_dir,
                show_progress=False,
            )
