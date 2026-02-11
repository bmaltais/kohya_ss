import os
import re
from typing import Dict, List, Optional, Union
import torch
from tqdm import tqdm
from library.device_utils import synchronize_device
from library.fp8_optimization_utils import load_safetensors_with_fp8_optimization
from library.safetensors_utils import MemoryEfficientSafeOpen, TensorWeightAdapter, WeightTransformHooks, get_split_weight_filenames
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def filter_lora_state_dict(
    weights_sd: Dict[str, torch.Tensor],
    include_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    # apply include/exclude patterns
    original_key_count = len(weights_sd.keys())
    if include_pattern is not None:
        regex_include = re.compile(include_pattern)
        weights_sd = {k: v for k, v in weights_sd.items() if regex_include.search(k)}
        logger.info(f"Filtered keys with include pattern {include_pattern}: {original_key_count} -> {len(weights_sd.keys())}")

    if exclude_pattern is not None:
        original_key_count_ex = len(weights_sd.keys())
        regex_exclude = re.compile(exclude_pattern)
        weights_sd = {k: v for k, v in weights_sd.items() if not regex_exclude.search(k)}
        logger.info(f"Filtered keys with exclude pattern {exclude_pattern}: {original_key_count_ex} -> {len(weights_sd.keys())}")

    if len(weights_sd) != original_key_count:
        remaining_keys = list(set([k.split(".", 1)[0] for k in weights_sd.keys()]))
        remaining_keys.sort()
        logger.info(f"Remaining LoRA modules after filtering: {remaining_keys}")
        if len(weights_sd) == 0:
            logger.warning("No keys left after filtering.")

    return weights_sd


def load_safetensors_with_lora_and_fp8(
    model_files: Union[str, List[str]],
    lora_weights_list: Optional[List[Dict[str, torch.Tensor]]],
    lora_multipliers: Optional[List[float]],
    fp8_optimization: bool,
    calc_device: torch.device,
    move_to_device: bool = False,
    dit_weight_dtype: Optional[torch.dtype] = None,
    target_keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
    disable_numpy_memmap: bool = False,
    weight_transform_hooks: Optional[WeightTransformHooks] = None,
) -> dict[str, torch.Tensor]:
    """
    Merge LoRA weights into the state dict of a model with fp8 optimization if needed.

    Args:
        model_files (Union[str, List[str]]): Path to the model file or list of paths. If the path matches a pattern like `00001-of-00004`, it will load all files with the same prefix.
        lora_weights_list (Optional[List[Dict[str, torch.Tensor]]]): List of dictionaries of LoRA weight tensors to load.
        lora_multipliers (Optional[List[float]]): List of multipliers for LoRA weights.
        fp8_optimization (bool): Whether to apply FP8 optimization.
        calc_device (torch.device): Device to calculate on.
        move_to_device (bool): Whether to move tensors to the calculation device after loading.
        target_keys (Optional[List[str]]): Keys to target for optimization.
        exclude_keys (Optional[List[str]]): Keys to exclude from optimization.
        disable_numpy_memmap (bool): Whether to disable numpy memmap when loading safetensors.
        weight_transform_hooks (Optional[WeightTransformHooks]): Hooks for transforming weights during loading.
    """

    # if the file name ends with 00001-of-00004 etc, we need to load the files with the same prefix
    if isinstance(model_files, str):
        model_files = [model_files]

    extended_model_files = []
    for model_file in model_files:
        split_filenames = get_split_weight_filenames(model_file)
        if split_filenames is not None:
            extended_model_files.extend(split_filenames)
        else:
            extended_model_files.append(model_file)
    model_files = extended_model_files
    logger.info(f"Loading model files: {model_files}")

    # load LoRA weights
    weight_hook = None
    if lora_weights_list is None or len(lora_weights_list) == 0:
        lora_weights_list = []
        lora_multipliers = []
        list_of_lora_weight_keys = []
    else:
        list_of_lora_weight_keys = []
        for lora_sd in lora_weights_list:
            lora_weight_keys = set(lora_sd.keys())
            list_of_lora_weight_keys.append(lora_weight_keys)

        if lora_multipliers is None:
            lora_multipliers = [1.0] * len(lora_weights_list)
        while len(lora_multipliers) < len(lora_weights_list):
            lora_multipliers.append(1.0)
        if len(lora_multipliers) > len(lora_weights_list):
            lora_multipliers = lora_multipliers[: len(lora_weights_list)]

        # Merge LoRA weights into the state dict
        logger.info(f"Merging LoRA weights into state dict. multipliers: {lora_multipliers}")

        # make hook for LoRA merging
        def weight_hook_func(model_weight_key, model_weight: torch.Tensor, keep_on_calc_device=False):
            nonlocal list_of_lora_weight_keys, lora_weights_list, lora_multipliers, calc_device

            if not model_weight_key.endswith(".weight"):
                return model_weight

            original_device = model_weight.device
            if original_device != calc_device:
                model_weight = model_weight.to(calc_device)  # to make calculation faster

            for lora_weight_keys, lora_sd, multiplier in zip(list_of_lora_weight_keys, lora_weights_list, lora_multipliers):
                # check if this weight has LoRA weights
                lora_name = model_weight_key.rsplit(".", 1)[0]  # remove trailing ".weight"
                lora_name = "lora_unet_" + lora_name.replace(".", "_")
                down_key = lora_name + ".lora_down.weight"
                up_key = lora_name + ".lora_up.weight"
                alpha_key = lora_name + ".alpha"
                if down_key not in lora_weight_keys or up_key not in lora_weight_keys:
                    continue

                # get LoRA weights
                down_weight = lora_sd[down_key]
                up_weight = lora_sd[up_key]

                dim = down_weight.size()[0]
                alpha = lora_sd.get(alpha_key, dim)
                scale = alpha / dim

                down_weight = down_weight.to(calc_device)
                up_weight = up_weight.to(calc_device)

                original_dtype = model_weight.dtype
                if original_dtype.itemsize == 1:  # fp8
                    # temporarily convert to float16 for calculation
                    model_weight = model_weight.to(torch.float16)
                    down_weight = down_weight.to(torch.float16)
                    up_weight = up_weight.to(torch.float16)

                # W <- W + U * D
                if len(model_weight.size()) == 2:
                    # linear
                    if len(up_weight.size()) == 4:  # use linear projection mismatch
                        up_weight = up_weight.squeeze(3).squeeze(2)
                        down_weight = down_weight.squeeze(3).squeeze(2)
                    model_weight = model_weight + multiplier * (up_weight @ down_weight) * scale
                elif down_weight.size()[2:4] == (1, 1):
                    # conv2d 1x1
                    model_weight = (
                        model_weight
                        + multiplier
                        * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                        * scale
                    )
                else:
                    # conv2d 3x3
                    conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                    # logger.info(conved.size(), weight.size(), module.stride, module.padding)
                    model_weight = model_weight + multiplier * conved * scale

                if original_dtype.itemsize == 1:  # fp8
                    model_weight = model_weight.to(original_dtype)  # convert back to original dtype

                # remove LoRA keys from set
                lora_weight_keys.remove(down_key)
                lora_weight_keys.remove(up_key)
                if alpha_key in lora_weight_keys:
                    lora_weight_keys.remove(alpha_key)

            if not keep_on_calc_device and original_device != calc_device:
                model_weight = model_weight.to(original_device)  # move back to original device
            return model_weight

        weight_hook = weight_hook_func

    state_dict = load_safetensors_with_fp8_optimization_and_hook(
        model_files,
        fp8_optimization,
        calc_device,
        move_to_device,
        dit_weight_dtype,
        target_keys,
        exclude_keys,
        weight_hook=weight_hook,
        disable_numpy_memmap=disable_numpy_memmap,
        weight_transform_hooks=weight_transform_hooks,
    )

    for lora_weight_keys in list_of_lora_weight_keys:
        # check if all LoRA keys are used
        if len(lora_weight_keys) > 0:
            # if there are still LoRA keys left, it means they are not used in the model
            # this is a warning, not an error
            logger.warning(f"Warning: not all LoRA keys are used: {', '.join(lora_weight_keys)}")

    return state_dict


def load_safetensors_with_fp8_optimization_and_hook(
    model_files: list[str],
    fp8_optimization: bool,
    calc_device: torch.device,
    move_to_device: bool = False,
    dit_weight_dtype: Optional[torch.dtype] = None,
    target_keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
    weight_hook: callable = None,
    disable_numpy_memmap: bool = False,
    weight_transform_hooks: Optional[WeightTransformHooks] = None,
) -> dict[str, torch.Tensor]:
    """
    Load state dict from safetensors files and merge LoRA weights into the state dict with fp8 optimization if needed.
    """
    if fp8_optimization:
        logger.info(
            f"Loading state dict with FP8 optimization. Dtype of weight: {dit_weight_dtype}, hook enabled: {weight_hook is not None}"
        )
        # dit_weight_dtype is not used because we use fp8 optimization
        state_dict = load_safetensors_with_fp8_optimization(
            model_files,
            calc_device,
            target_keys,
            exclude_keys,
            move_to_device=move_to_device,
            weight_hook=weight_hook,
            disable_numpy_memmap=disable_numpy_memmap,
            weight_transform_hooks=weight_transform_hooks,
        )
    else:
        logger.info(
            f"Loading state dict without FP8 optimization. Dtype of weight: {dit_weight_dtype}, hook enabled: {weight_hook is not None}"
        )
        state_dict = {}
        for model_file in model_files:
            with MemoryEfficientSafeOpen(model_file, disable_numpy_memmap=disable_numpy_memmap) as original_f:
                f = TensorWeightAdapter(weight_transform_hooks, original_f) if weight_transform_hooks is not None else original_f
                for key in tqdm(f.keys(), desc=f"Loading {os.path.basename(model_file)}", leave=False):
                    if weight_hook is None and move_to_device:
                        value = f.get_tensor(key, device=calc_device, dtype=dit_weight_dtype)
                    else:
                        value = f.get_tensor(key)  # we cannot directly load to device because get_tensor does non-blocking transfer
                        if weight_hook is not None:
                            value = weight_hook(key, value, keep_on_calc_device=move_to_device)
                        if move_to_device:
                            value = value.to(calc_device, dtype=dit_weight_dtype, non_blocking=True)
                        elif dit_weight_dtype is not None:
                            value = value.to(dit_weight_dtype)

                    state_dict[key] = value
        if move_to_device:
            synchronize_device(calc_device)

    return state_dict
