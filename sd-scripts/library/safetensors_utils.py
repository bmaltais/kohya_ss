from dataclasses import dataclass
import os
import re
import numpy as np
import torch
import json
import struct
from typing import Dict, Any, Union, Optional

from safetensors.torch import load_file

from library.device_utils import synchronize_device


def mem_eff_save_file(tensors: Dict[str, torch.Tensor], filename: str, metadata: Dict[str, Any] = None):
    """
    memory efficient save file
    """

    _TYPES = {
        torch.float64: "F64",
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.int16: "I16",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.bool: "BOOL",
        getattr(torch, "float8_e5m2", None): "F8_E5M2",
        getattr(torch, "float8_e4m3fn", None): "F8_E4M3",
    }
    _ALIGN = 256

    def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
        validated = {}
        for key, value in metadata.items():
            if not isinstance(key, str):
                raise ValueError(f"Metadata key must be a string, got {type(key)}")
            if not isinstance(value, str):
                print(f"Warning: Metadata value for key '{key}' is not a string. Converting to string.")
                validated[key] = str(value)
            else:
                validated[key] = value
        return validated

    # print(f"Using memory efficient save file: {filename}")

    header = {}
    offset = 0
    if metadata:
        header["__metadata__"] = validate_metadata(metadata)
    for k, v in tensors.items():
        if v.numel() == 0:  # empty tensor
            header[k] = {"dtype": _TYPES[v.dtype], "shape": list(v.shape), "data_offsets": [offset, offset]}
        else:
            size = v.numel() * v.element_size()
            header[k] = {"dtype": _TYPES[v.dtype], "shape": list(v.shape), "data_offsets": [offset, offset + size]}
            offset += size

    hjson = json.dumps(header).encode("utf-8")
    hjson += b" " * (-(len(hjson) + 8) % _ALIGN)

    with open(filename, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)

        for k, v in tensors.items():
            if v.numel() == 0:
                continue
            if v.is_cuda:
                # Direct GPU to disk save
                with torch.cuda.device(v.device):
                    if v.dim() == 0:  # if scalar, need to add a dimension to work with view
                        v = v.unsqueeze(0)
                    tensor_bytes = v.contiguous().view(torch.uint8)
                    tensor_bytes.cpu().numpy().tofile(f)
            else:
                # CPU tensor save
                if v.dim() == 0:  # if scalar, need to add a dimension to work with view
                    v = v.unsqueeze(0)
                v.contiguous().view(torch.uint8).numpy().tofile(f)


class MemoryEfficientSafeOpen:
    """Memory-efficient reader for safetensors files.

    This class provides a memory-efficient way to read tensors from safetensors files
    by using memory mapping for large tensors and avoiding unnecessary copies.
    """

    def __init__(self, filename, disable_numpy_memmap=False):
        """Initialize the SafeTensor reader.

        Args:
            filename (str): Path to the safetensors file to read.
            disable_numpy_memmap (bool): If True, disable numpy memory mapping for large tensors, using standard file read instead.
        """
        self.filename = filename
        self.file = open(filename, "rb")
        self.header, self.header_size = self._read_header()
        self.disable_numpy_memmap = disable_numpy_memmap

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and close file."""
        self.file.close()

    def keys(self):
        """Get all tensor keys in the file.

        Returns:
            list: List of tensor names (excludes metadata).
        """
        return [k for k in self.header.keys() if k != "__metadata__"]

    def metadata(self) -> Dict[str, str]:
        """Get metadata from the file.

        Returns:
            Dict[str, str]: Metadata dictionary.
        """
        return self.header.get("__metadata__", {})

    def _read_header(self):
        """Read and parse the header from the safetensors file.

        Returns:
            tuple: (header_dict, header_size) containing parsed header and its size.
        """
        # Read header size (8 bytes, little-endian unsigned long long)
        header_size = struct.unpack("<Q", self.file.read(8))[0]
        # Read and decode header JSON
        header_json = self.file.read(header_size).decode("utf-8")
        return json.loads(header_json), header_size

    def get_tensor(self, key: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """Load a tensor from the file with memory-efficient strategies.

        **Note:**
        If device is 'cuda' , the transfer to GPU is done efficiently using pinned memory and non-blocking transfer.
        So you must ensure that the transfer is completed before using the tensor (e.g., by `torch.cuda.synchronize()`).

        If the tensor is large (>10MB) and the target device is CUDA, memory mapping with numpy.memmap is used to avoid intermediate copies.

        Args:
            key (str): Name of the tensor to load.
            device (Optional[torch.device]): Target device for the tensor.
            dtype (Optional[torch.dtype]): Target dtype for the tensor.

        Returns:
            torch.Tensor: The loaded tensor.

        Raises:
            KeyError: If the tensor key is not found in the file.
        """
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")

        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]
        num_bytes = offset_end - offset_start

        original_dtype = self._get_torch_dtype(metadata["dtype"])
        target_dtype = dtype if dtype is not None else original_dtype

        # Handle empty tensors
        if num_bytes == 0:
            return torch.empty(metadata["shape"], dtype=target_dtype, device=device)

        # Determine if we should use pinned memory for GPU transfer
        non_blocking = device is not None and device.type == "cuda"

        # Calculate absolute file offset
        tensor_offset = self.header_size + 8 + offset_start  # adjust offset by header size

        # Memory mapping strategy for large tensors to GPU
        # Use memmap for large tensors to avoid intermediate copies.
        # If device is cpu, tensor is not copied to gpu, so using memmap locks the file, which is not desired.
        # So we only use memmap if device is not cpu.
        # If disable_numpy_memmap is True, skip numpy memory mapping to load with standard file read.
        if not self.disable_numpy_memmap and num_bytes > 10 * 1024 * 1024 and device is not None and device.type != "cpu":
            # Create memory map for zero-copy reading
            mm = np.memmap(self.filename, mode="c", dtype=np.uint8, offset=tensor_offset, shape=(num_bytes,))
            byte_tensor = torch.from_numpy(mm)  # zero copy
            del mm

            # Deserialize tensor (view and reshape)
            cpu_tensor = self._deserialize_tensor(byte_tensor, metadata)  # view and reshape
            del byte_tensor

            # Transfer to target device and dtype
            gpu_tensor = cpu_tensor.to(device=device, dtype=target_dtype, non_blocking=non_blocking)
            del cpu_tensor
            return gpu_tensor

        # Standard file reading strategy for smaller tensors or CPU target
        # seek to the specified position
        self.file.seek(tensor_offset)

        # read directly into a numpy array by numpy.fromfile without intermediate copy
        numpy_array = np.fromfile(self.file, dtype=np.uint8, count=num_bytes)
        byte_tensor = torch.from_numpy(numpy_array)
        del numpy_array

        # deserialize (view and reshape)
        deserialized_tensor = self._deserialize_tensor(byte_tensor, metadata)
        del byte_tensor

        # cast to target dtype and move to device
        return deserialized_tensor.to(device=device, dtype=target_dtype, non_blocking=non_blocking)

    def _deserialize_tensor(self, byte_tensor: torch.Tensor, metadata: Dict):
        """Deserialize byte tensor to the correct shape and dtype.

        Args:
            byte_tensor (torch.Tensor): Raw byte tensor from file.
            metadata (Dict): Tensor metadata containing dtype and shape info.

        Returns:
            torch.Tensor: Deserialized tensor with correct shape and dtype.
        """
        dtype = self._get_torch_dtype(metadata["dtype"])
        shape = metadata["shape"]

        # Handle special float8 types
        if metadata["dtype"] in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, metadata["dtype"], shape)

        # Standard conversion: view as target dtype and reshape
        return byte_tensor.view(dtype).reshape(shape)

    @staticmethod
    def _get_torch_dtype(dtype_str):
        """Convert string dtype to PyTorch dtype.

        Args:
            dtype_str (str): String representation of the dtype.

        Returns:
            torch.dtype: Corresponding PyTorch dtype.
        """
        # Standard dtype mappings
        dtype_map = {
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        # Add float8 types if available in PyTorch version
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn
        return dtype_map.get(dtype_str)

    @staticmethod
    def _convert_float8(byte_tensor, dtype_str, shape):
        """Convert byte tensor to float8 format if supported.

        Args:
            byte_tensor (torch.Tensor): Raw byte tensor.
            dtype_str (str): Float8 dtype string ("F8_E5M2" or "F8_E4M3").
            shape (tuple): Target tensor shape.

        Returns:
            torch.Tensor: Tensor with float8 dtype.

        Raises:
            ValueError: If float8 type is not supported in current PyTorch version.
        """
        # Convert to specific float8 types if available
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            # Float8 not supported in this PyTorch version
            raise ValueError(f"Unsupported float8 type: {dtype_str} (upgrade PyTorch to support float8 types)")


def load_safetensors(
    path: str,
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    dtype: Optional[torch.dtype] = None,
    disable_numpy_memmap: bool = False,
) -> dict[str, torch.Tensor]:
    if disable_mmap:
        # return safetensors.torch.load(open(path, "rb").read())
        # use experimental loader
        # logger.info(f"Loading without mmap (experimental)")
        state_dict = {}
        device = torch.device(device) if device is not None else None
        with MemoryEfficientSafeOpen(path, disable_numpy_memmap=disable_numpy_memmap) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key, device=device, dtype=dtype)
        synchronize_device(device)
        return state_dict
    else:
        try:
            state_dict = load_file(path, device=device)
        except:
            state_dict = load_file(path)  # prevent device invalid Error
        if dtype is not None:
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(dtype=dtype)
        return state_dict


def get_split_weight_filenames(file_path: str) -> Optional[list[str]]:
    """
    Get the list of split weight filenames (full paths) if the file name ends with 00001-of-00004 etc.
    Returns None if the file is not split.
    """
    basename = os.path.basename(file_path)
    match = re.match(r"^(.*?)(\d+)-of-(\d+)\.safetensors$", basename)
    if match:
        prefix = basename[: match.start(2)]
        count = int(match.group(3))
        filenames = []
        for i in range(count):
            filename = f"{prefix}{i + 1:05d}-of-{count:05d}.safetensors"
            filepath = os.path.join(os.path.dirname(file_path), filename)
            if os.path.exists(filepath):
                filenames.append(filepath)
            else:
                raise FileNotFoundError(f"File {filepath} not found")
        return filenames
    else:
        return None


def load_split_weights(
    file_path: str, device: Union[str, torch.device] = "cpu", disable_mmap: bool = False, dtype: Optional[torch.dtype] = None
) -> Dict[str, torch.Tensor]:
    """
    Load split weights from a file. If the file name ends with 00001-of-00004 etc, it will load all files with the same prefix.
    dtype is as is, no conversion is done.
    """
    device = torch.device(device)

    # if the file name ends with 00001-of-00004 etc, we need to load the files with the same prefix
    split_filenames = get_split_weight_filenames(file_path)
    if split_filenames is not None:
        state_dict = {}
        for filename in split_filenames:
            state_dict.update(load_safetensors(filename, device=device, disable_mmap=disable_mmap, dtype=dtype))
    else:
        state_dict = load_safetensors(file_path, device=device, disable_mmap=disable_mmap, dtype=dtype)
    return state_dict


def find_key(safetensors_file: str, starts_with: Optional[str] = None, ends_with: Optional[str] = None) -> Optional[str]:
    """
    Find a key in a safetensors file that starts with `starts_with` and ends with `ends_with`.
    If `starts_with` is None, it will match any key.
    If `ends_with` is None, it will match any key.
    Returns the first matching key or None if no key matches.
    """
    with MemoryEfficientSafeOpen(safetensors_file) as f:
        for key in f.keys():
            if (starts_with is None or key.startswith(starts_with)) and (ends_with is None or key.endswith(ends_with)):
                return key
    return None


@dataclass
class WeightTransformHooks:
    split_hook: Optional[callable] = None
    concat_hook: Optional[callable] = None
    rename_hook: Optional[callable] = None


class TensorWeightAdapter:
    """
    A wrapper for weight conversion hooks (split and concat) to be used with MemoryEfficientSafeOpen.
    This wrapper adapts the original MemoryEfficientSafeOpen to apply the provided split and concat hooks
    when loading tensors.

    split_hook: A callable that takes (original_key: str, original_tensor: torch.Tensor) and returns (new_keys: list[str], new_tensors: list[torch.Tensor]).
    concat_hook: A callable that takes (original_key: str, tensors: dict[str, torch.Tensor]) and returns (new_key: str,  concatenated_tensor: torch.Tensor).
    rename_hook: A callable that takes (original_key: str) and returns (new_key: str).

    If tensors is None, the hook should return only the new keys (for split) or new key (for concat), without tensors.

    No need to implement __enter__ and __exit__ methods, as they are handled by the original MemoryEfficientSafeOpen.
    Do not use this wrapper as a context manager directly, like `with WeightConvertHookWrapper(...) as f:`.

    **concat_hook is not tested yet.**
    """

    def __init__(self, weight_convert_hook: WeightTransformHooks, original_f: MemoryEfficientSafeOpen):
        self.original_f = original_f
        self.new_key_to_original_key_map: dict[str, Union[str, list[str]]] = (
            {}
        )  # for split: new_key -> original_key; for concat: new_key -> list of original_keys; for direct mapping: new_key -> original_key
        self.concat_key_set = set()  # set of concatenated keys
        self.split_key_set = set()  # set of split keys
        self.new_keys = []
        self.tensor_cache = {}  # cache for split tensors
        self.split_hook = weight_convert_hook.split_hook
        self.concat_hook = weight_convert_hook.concat_hook
        self.rename_hook = weight_convert_hook.rename_hook

        for key in self.original_f.keys():
            if self.split_hook is not None:
                converted_keys, _ = self.split_hook(key, None)  # get new keys only
                if converted_keys is not None:
                    for converted_key in converted_keys:
                        self.new_key_to_original_key_map[converted_key] = key
                        self.split_key_set.add(converted_key)
                    self.new_keys.extend(converted_keys)
                    continue  # skip concat_hook if split_hook is applied

            if self.concat_hook is not None:
                converted_key, _ = self.concat_hook(key, None)  # get new key only
                if converted_key is not None:
                    if converted_key not in self.concat_key_set:  # first time seeing this concatenated key
                        self.concat_key_set.add(converted_key)
                        self.new_key_to_original_key_map[converted_key] = []
                        self.new_keys.append(converted_key)

                    # multiple original keys map to the same concatenated key
                    self.new_key_to_original_key_map[converted_key].append(key)
                    continue  # skip to next key

            # direct mapping
            if self.rename_hook is not None:
                new_key = self.rename_hook(key)
                self.new_key_to_original_key_map[new_key] = key
            else:
                new_key = key

            self.new_keys.append(new_key)

    def keys(self) -> list[str]:
        return self.new_keys

    def get_tensor(self, new_key: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        # load tensor by new_key, applying split or concat hooks as needed
        if new_key not in self.new_key_to_original_key_map:
            # direct mapping
            return self.original_f.get_tensor(new_key, device=device, dtype=dtype)

        elif new_key in self.split_key_set:
            # split hook: split key is requested multiple times, so we cache the result
            original_key = self.new_key_to_original_key_map[new_key]
            if original_key not in self.tensor_cache:  # not yet split
                original_tensor = self.original_f.get_tensor(original_key, device=device, dtype=dtype)
                new_keys, new_tensors = self.split_hook(original_key, original_tensor)  # apply split hook
                for k, t in zip(new_keys, new_tensors):
                    self.tensor_cache[k] = t
            return self.tensor_cache.pop(new_key)  # return and remove from cache

        elif new_key in self.concat_key_set:
            # concat hook: concatenated key is requested only once, so we do not cache the result
            tensors = {}
            for original_key in self.new_key_to_original_key_map[new_key]:
                tensor = self.original_f.get_tensor(original_key, device=device, dtype=dtype)
                tensors[original_key] = tensor
            _, concatenated_tensors = self.concat_hook(self.new_key_to_original_key_map[new_key][0], tensors)  # apply concat hook
            return concatenated_tensors

        else:
            # direct mapping
            original_key = self.new_key_to_original_key_map[new_key]
            return self.original_f.get_tensor(original_key, device=device, dtype=dtype)
