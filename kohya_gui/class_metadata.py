import gradio as gr

from .class_gui_config import KohyaSSGUIConfig


class MetaData:
    def __init__(
        self,
        config: KohyaSSGUIConfig = {},
    ) -> None:
        self.config = config

        with gr.Row():
            self.metadata_title = gr.Textbox(
                label="Metadata title",
                placeholder="(optional) title for model metadata (default is output_name)",
                interactive=True,
                value=self.config.get("metadata.title", ""),
            )
            self.metadata_author = gr.Textbox(
                label="Metadata author",
                placeholder="(optional) author name for model metadata",
                interactive=True,
                value=self.config.get("metadata.author", ""),
            )
        self.metadata_description = gr.Textbox(
            label="Metadata description",
            placeholder="(optional) description for model metadata",
            interactive=True,
            value=self.config.get("metadata.description", ""),
        )
        with gr.Row():
            self.metadata_license = gr.Textbox(
                label="Metadata license",
                placeholder="(optional) license for model metadata",
                interactive=True,
                value=self.config.get("metadata.license", ""),
            )
            self.metadata_tags = gr.Textbox(
                label="Metadata tags",
                placeholder="(optional) tags for model metadata, separated by comma",
                interactive=True,
                value=self.config.get("metadata.tags", ""),
            )

    def run_cmd(run_cmd: list, **kwargs):
        if "metadata_title" in kwargs and kwargs.get("metadata_title") != "":
            run_cmd.append("--metadata_title")
            run_cmd.append(kwargs["metadata_title"])

        if "metadata_author" in kwargs and kwargs.get("metadata_author") != "":
            run_cmd.append("--metadata_author")
            run_cmd.append(kwargs["metadata_author"])

        if "metadata_description" in kwargs and kwargs.get("metadata_description") != "":
            run_cmd.append("--metadata_description")
            run_cmd.append(kwargs["metadata_description"])

        if "metadata_license" in kwargs and kwargs.get("metadata_license") != "":
            run_cmd.append("--metadata_license")
            run_cmd.append(kwargs["metadata_license"])

        if "metadata_tags" in kwargs and kwargs.get("metadata_tags") != "":
            run_cmd.append("--metadata_tags")
            run_cmd.append(kwargs["metadata_tags"])

        return run_cmd
