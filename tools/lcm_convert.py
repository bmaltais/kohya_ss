import argparse
import torch
import logging
from library.utils import setup_logging
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, LCMScheduler
from library.sdxl_model_util import convert_diffusers_unet_state_dict_to_sdxl, sdxl_original_unet, save_stable_diffusion_checkpoint, _load_state_dict_on_device as load_state_dict_on_device
from accelerate import init_empty_weights

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)


def parse_command_line_arguments():
    argument_parser = argparse.ArgumentParser("lcm_convert")
    argument_parser.add_argument("--name", help="Name of the new LCM model", required=True, type=str)
    argument_parser.add_argument("--model", help="A model to convert", required=True, type=str)
    argument_parser.add_argument("--lora-scale", default=1.0, help="Strength of the LCM", type=float)
    argument_parser.add_argument("--sdxl", action="store_true", help="Use SDXL models")
    argument_parser.add_argument("--ssd-1b", action="store_true", help="Use SSD-1B models")
    return argument_parser.parse_args()

def load_diffusion_pipeline(command_line_args):
    if command_line_args.sdxl or command_line_args.ssd_1b:
        return StableDiffusionXLPipeline.from_single_file(command_line_args.model)
    else:
        return StableDiffusionPipeline.from_single_file(command_line_args.model)

def convert_and_save_diffusion_model(diffusion_pipeline, command_line_args):
    diffusion_pipeline.scheduler = LCMScheduler.from_config(diffusion_pipeline.scheduler.config)
    lora_weight_file_path = "latent-consistency/lcm-lora-" + ("sdxl" if command_line_args.sdxl else "ssd-1b" if command_line_args.ssd_1b else "sdv1-5")
    diffusion_pipeline.load_lora_weights(lora_weight_file_path)
    diffusion_pipeline.fuse_lora(lora_scale=command_line_args.lora_scale)

    diffusion_pipeline = diffusion_pipeline.to(dtype=torch.float16)
    logger.info("Saving file...")

    text_encoder_primary = diffusion_pipeline.text_encoder
    text_encoder_secondary = diffusion_pipeline.text_encoder_2
    variational_autoencoder = diffusion_pipeline.vae
    unet_network = diffusion_pipeline.unet

    del diffusion_pipeline

    state_dict = convert_diffusers_unet_state_dict_to_sdxl(unet_network.state_dict())
    with init_empty_weights():
        unet_network = sdxl_original_unet.SdxlUNet2DConditionModel()
    
    load_state_dict_on_device(unet_network, state_dict, device="cuda", dtype=torch.float16)

    save_stable_diffusion_checkpoint(
        command_line_args.name,
        text_encoder_primary,
        text_encoder_secondary,
        unet_network,
        None,
        None,
        None,
        variational_autoencoder,
        None,
        None,
        torch.float16,
    )

    logger.info("...done saving")

def main():
    command_line_args = parse_command_line_arguments()
    try:
        diffusion_pipeline = load_diffusion_pipeline(command_line_args)
        convert_and_save_diffusion_model(diffusion_pipeline, command_line_args)
    except Exception as error:
        logger.error(f"An error occurred: {error}")

if __name__ == "__main__":
    main()
