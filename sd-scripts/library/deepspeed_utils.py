import os
import argparse
import torch
from accelerate import DeepSpeedPlugin, Accelerator

from .utils import setup_logging

from .device_utils import get_preferred_device

setup_logging()
import logging

logger = logging.getLogger(__name__)


def add_deepspeed_arguments(parser: argparse.ArgumentParser):
    # DeepSpeed Arguments. https://huggingface.co/docs/accelerate/usage_guides/deepspeed
    parser.add_argument("--deepspeed", action="store_true", help="enable deepspeed training")
    parser.add_argument("--zero_stage", type=int, default=2, choices=[0, 1, 2, 3], help="Possible options are 0,1,2,3.")
    parser.add_argument(
        "--offload_optimizer_device",
        type=str,
        default=None,
        choices=[None, "cpu", "nvme"],
        help="Possible options are none|cpu|nvme. Only applicable with ZeRO Stages 2 and 3.",
    )
    parser.add_argument(
        "--offload_optimizer_nvme_path",
        type=str,
        default=None,
        help="Possible options are /nvme|/local_nvme. Only applicable with ZeRO Stage 3.",
    )
    parser.add_argument(
        "--offload_param_device",
        type=str,
        default=None,
        choices=[None, "cpu", "nvme"],
        help="Possible options are none|cpu|nvme. Only applicable with ZeRO Stage 3.",
    )
    parser.add_argument(
        "--offload_param_nvme_path",
        type=str,
        default=None,
        help="Possible options are /nvme|/local_nvme. Only applicable with ZeRO Stage 3.",
    )
    parser.add_argument(
        "--zero3_init_flag",
        action="store_true",
        help="Flag to indicate whether to enable `deepspeed.zero.Init` for constructing massive models."
        "Only applicable with ZeRO Stage-3.",
    )
    parser.add_argument(
        "--zero3_save_16bit_model",
        action="store_true",
        help="Flag to indicate whether to save 16-bit model. Only applicable with ZeRO Stage-3.",
    )
    parser.add_argument(
        "--fp16_master_weights_and_gradients",
        action="store_true",
        help="fp16_master_and_gradients requires optimizer to support keeping fp16 master and gradients while keeping the optimizer states in fp32.",
    )


def prepare_deepspeed_args(args: argparse.Namespace):
    if not args.deepspeed:
        return

    # To avoid RuntimeError: DataLoader worker exited unexpectedly with exit code 1.
    args.max_data_loader_n_workers = 1


def prepare_deepspeed_plugin(args: argparse.Namespace):
    if not args.deepspeed:
        return None

    try:
        import deepspeed
    except ImportError as e:
        logger.error(
            "deepspeed is not installed. please install deepspeed in your environment with following command. DS_BUILD_OPS=0 pip install deepspeed"
        )
        exit(1)

    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=args.zero_stage,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clipping=args.max_grad_norm,
        offload_optimizer_device=args.offload_optimizer_device,
        offload_optimizer_nvme_path=args.offload_optimizer_nvme_path,
        offload_param_device=args.offload_param_device,
        offload_param_nvme_path=args.offload_param_nvme_path,
        zero3_init_flag=args.zero3_init_flag,
        zero3_save_16bit_model=args.zero3_save_16bit_model,
    )
    deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_batch_size
    deepspeed_plugin.deepspeed_config["train_batch_size"] = (
        args.train_batch_size * args.gradient_accumulation_steps * int(os.environ["WORLD_SIZE"])
    )
    
    deepspeed_plugin.set_mixed_precision(args.mixed_precision)
    if args.mixed_precision.lower() == "fp16":
        deepspeed_plugin.deepspeed_config["fp16"]["initial_scale_power"] = 0  # preventing overflow.
    if args.full_fp16 or args.fp16_master_weights_and_gradients:
        if args.offload_optimizer_device == "cpu" and args.zero_stage == 2:
            deepspeed_plugin.deepspeed_config["fp16"]["fp16_master_weights_and_grads"] = True
            logger.info("[DeepSpeed] full fp16 enable.")
        else:
            logger.info(
                "[DeepSpeed]full fp16, fp16_master_weights_and_grads currently only supported using ZeRO-Offload with DeepSpeedCPUAdam on ZeRO-2 stage."
            )

    if args.offload_optimizer_device is not None:
        logger.info("[DeepSpeed] start to manually build cpu_adam.")
        deepspeed.ops.op_builder.CPUAdamBuilder().load()
        logger.info("[DeepSpeed] building cpu_adam done.")

    return deepspeed_plugin


# Accelerate library does not support multiple models for deepspeed. So, we need to wrap multiple models into a single model.
def prepare_deepspeed_model(args: argparse.Namespace, **models):
    # remove None from models
    models = {k: v for k, v in models.items() if v is not None}

    class DeepSpeedWrapper(torch.nn.Module):
        def __init__(self, **kw_models) -> None:
            super().__init__()
            
            self.models = torch.nn.ModuleDict()
            
            wrap_model_forward_with_torch_autocast = args.mixed_precision is not "no"

            for key, model in kw_models.items():
                if isinstance(model, list):
                    model = torch.nn.ModuleList(model)
                                            
                if wrap_model_forward_with_torch_autocast:
                    model = self.__wrap_model_with_torch_autocast(model)  
                
                assert isinstance(
                    model, torch.nn.Module
                ), f"model must be an instance of torch.nn.Module, but got {key} is {type(model)}"

                self.models.update(torch.nn.ModuleDict({key: model}))

        def __wrap_model_with_torch_autocast(self, model):
            if isinstance(model, torch.nn.ModuleList):
                model = torch.nn.ModuleList([self.__wrap_model_forward_with_torch_autocast(m) for m in model])
            else:
                model = self.__wrap_model_forward_with_torch_autocast(model)
            return model

        def __wrap_model_forward_with_torch_autocast(self, model):
            
            assert hasattr(model, "forward"), f"model must have a forward method."

            forward_fn = model.forward

            def forward(*args, **kwargs):
                try:
                    device_type = model.device.type
                except AttributeError:
                    logger.warning(
                            "[DeepSpeed] model.device is not available. Using get_preferred_device() "
                            "to determine the device_type for torch.autocast()."
                    )                    
                    device_type = get_preferred_device().type

                with torch.autocast(device_type = device_type):
                    return forward_fn(*args, **kwargs)

            model.forward = forward
            return model
        
        def get_models(self):
            return self.models
        

    ds_model = DeepSpeedWrapper(**models)
    return ds_model
