import gradio as gr
import toml
from .class_gui_config import KohyaSSGUIConfig

class HuggingFace:
    def __init__(
        self,
        config: KohyaSSGUIConfig,
    ) -> None:
        self.config = config

        # Initialize the UI components
        self.initialize_ui_components()

    def initialize_ui_components(self) -> None:
        # --huggingface_repo_id HUGGINGFACE_REPO_ID
        #                         huggingface repo name to upload / huggingfaceにアップロードするリポジトリ名
        # --huggingface_repo_type HUGGINGFACE_REPO_TYPE
        #                         huggingface repo type to upload / huggingfaceにアップロードするリポジトリの種類
        # --huggingface_path_in_repo HUGGINGFACE_PATH_IN_REPO
        #                         huggingface model path to upload files / huggingfaceにアップロードするファイルのパス
        # --huggingface_token HUGGINGFACE_TOKEN
        #                         huggingface token / huggingfaceのトークン
        # --huggingface_repo_visibility HUGGINGFACE_REPO_VISIBILITY
        #                         huggingface repository visibility ('public' for public, 'private' or None for private) / huggingfaceにアップロードするリポジトリの公開設定（'public'で公開、'private'またはNoneで非公開）
        # --save_state_to_huggingface
        #                         save state to huggingface / huggingfaceにstateを保存する
        # --resume_from_huggingface
        #                         resume from huggingface (ex: --resume {repo_id}/{path_in_repo}:{revision}:{repo_type}) / huggingfaceから学習を再開する(例: --resume {repo_id}/{path_in_repo}:{revision}:{repo_type})
        # --async_upload        upload to huggingface asynchronously / huggingfaceに非同期でアップロードする
        with gr.Row():
            self.huggingface_repo_id = gr.Textbox(
                label="Huggingface repo id",
                placeholder="huggingface repo id",
                value=self.config.get("huggingface.repo_id", ""),
            )

            self.huggingface_token = gr.Textbox(
                label="Huggingface token",
                placeholder="huggingface token",
                value=self.config.get("huggingface.token", ""),
            )

        with gr.Row():
            # Repository settings
            self.huggingface_repo_type = gr.Textbox(
                label="Huggingface repo type",
                placeholder="huggingface repo type",
                value=self.config.get("huggingface.repo_type", ""),
            )

            self.huggingface_repo_visibility = gr.Textbox(
                label="Huggingface repo visibility",
                placeholder="huggingface repo visibility",
                value=self.config.get("huggingface.repo_visibility", ""),
            )

        with gr.Row():
            # File location in the repository
            self.huggingface_path_in_repo = gr.Textbox(
                label="Huggingface path in repo",
                placeholder="huggingface path in repo",
                value=self.config.get("huggingface.path_in_repo", ""),
            )

        with gr.Row():
            # Functions
            self.save_state_to_huggingface = gr.Checkbox(
                label="Save state to huggingface",
                value=self.config.get("huggingface.save_state_to_huggingface", False),
            )

            self.resume_from_huggingface = gr.Textbox(
                label="Resume from huggingface",
                placeholder="resume from huggingface",
                value=self.config.get("huggingface.resume_from_huggingface", ""),
            )

            self.async_upload = gr.Checkbox(
                label="Async upload",
                value=self.config.get("huggingface.async_upload", False),
            )