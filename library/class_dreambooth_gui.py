import gradio as gr
import json
from .class_configuration_file import ConfigurationFile
from .class_source_model import SourceModel
from .class_folders import Folders
from .class_basic_training import BasicTraining
from .class_advanced_training import AdvancedTraining
from .class_sample_images import SampleImages
from library.dreambooth_folder_creation_gui import (
    gradio_dreambooth_folder_creation_tab,
)
from .common_gui import color_aug_changed

class Dreambooth:
    def __init__(
        self,
        headless: bool = False,
    ):
        self.headless = headless
        self.dummy_db_true = gr.Label(value=True, visible=False)
        self.dummy_db_false = gr.Label(value=False, visible=False)
        self.dummy_headless = gr.Label(value=headless, visible=False)

        gr.Markdown('Train a custom model using kohya dreambooth python code...')
    
        # Setup Configuration Files Gradio
        self.config = ConfigurationFile(headless)
        
        self.source_model = SourceModel(headless=headless)
        
        with gr.Tab('Folders'):
            self.folders = Folders(headless=headless)
        with gr.Tab('Parameters'):
            self.basic_training = BasicTraining(
                learning_rate_value='1e-5',
                lr_scheduler_value='cosine',
                lr_warmup_value='10',
            )
            self.full_bf16 = gr.Checkbox(
                label='Full bf16', value = False
            )
            with gr.Accordion('Advanced Configuration', open=False):
                self.advanced_training = AdvancedTraining(headless=headless)
                self.advanced_training.color_aug.change(
                    color_aug_changed,
                    inputs=[self.advanced_training.color_aug],
                    outputs=[self.basic_training.cache_latents],
                )

            self.sample = SampleImages()

        with gr.Tab('Tools'):
            gr.Markdown(
                'This section provide Dreambooth tools to help setup your dataset...'
            )
            gradio_dreambooth_folder_creation_tab(
                train_data_dir_input=self.folders.train_data_dir,
                reg_data_dir_input=self.folders.reg_data_dir,
                output_dir_input=self.folders.output_dir,
                logging_dir_input=self.folders.logging_dir,
                headless=headless,
            )
            
    def save_to_json(self, filepath):
        def serialize(obj):
            if isinstance(obj, gr.inputs.Input):
                return obj.get()
            if isinstance(obj, (bool, int, float, str)):
                return obj
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            if hasattr(obj, "__dict__"):
                return serialize(vars(obj))
            return str(obj)  # Fallback for objects that can't be serialized

        try:
            with open(filepath, 'w') as outfile:
                print(serialize(vars(self)))
                json.dump(serialize(vars(self)), outfile)
        except Exception as e:
            print(f"Error saving to JSON: {str(e)}")

    def load_from_json(self, filepath):
        def deserialize(key, value):
            if hasattr(self, key):
                attr = getattr(self, key)
                if isinstance(attr, gr.inputs.Input):
                    attr.set(value)
                elif hasattr(attr, "__dict__"):
                    for k, v in value.items():
                        deserialize(k, v)
                else:
                    setattr(self, key, value)
            else:
                print(f"Warning: {key} not found in the object's attributes.")

        try:
            with open(filepath) as json_file:
                data = json.load(json_file)
                for key, value in data.items():
                    deserialize(key, value)
        except FileNotFoundError:
            print(f"Error: The file {filepath} was not found.")
        except json.JSONDecodeError:
            print(f"Error: The file {filepath} could not be decoded as JSON.")
        except Exception as e:
            print(f"Error loading from JSON: {str(e)}")