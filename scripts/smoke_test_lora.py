"""Move 9 smoke test: build a real v2 LoRA run config and print the
resulting TOML/command, for manual/automated launch. Not part of the
shipped package -- a one-off validation script for this checkpoint.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from kohya_gui_v2.config_io import build_run_config
from kohya_gui_v2.tabs.lora_fields import LORA_REGISTRY, derive

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "smoke_test", "lora")
os.makedirs(OUTPUT_DIR, exist_ok=True)

arch_key = "sd15"
raw = {spec.name: spec.default for spec in LORA_REGISTRY}
raw.update(
    {
        "architecture": arch_key,
        "training_type": "lora",
        "pretrained_model_name_or_path": os.path.join(
            PROJECT_ROOT, "models", "v1-5-pruned-emaonly.safetensors"
        ),
        "train_data_dir": os.path.join(PROJECT_ROOT, "test", "img"),
        "output_dir": OUTPUT_DIR,
        "output_name": "smoke_lora",
        "resolution": "512,512",
        "train_batch_size": 1,
        "max_train_steps": 20,
        "save_every_n_epochs": 0,
        "network_dim": 4,
        "network_alpha": 1.0,
        "network_module": "networks.lora",
        "learning_rate": 0.0001,
        "optimizer": "AdamW8bit",
        "lr_scheduler": "constant",
        "mixed_precision": "fp16",
        "cache_latents": True,
        "seed": 42,
        "max_data_loader_n_workers": 0,
        "xformers": "sdpa",
        "caption_extension": ".txt",
        "dynamo_backend": "no",
    }
)
raw.update(derive(raw, arch_key))

run_config = build_run_config(
    LORA_REGISTRY, raw, arch_key=arch_key, training_type="lora"
)

toml_path = os.path.join(OUTPUT_DIR, "smoke_lora_config.toml")
import toml as toml_lib

with open(toml_path, "w", encoding="utf-8") as f:
    toml_lib.dump(run_config, f)

script = os.path.join(PROJECT_ROOT, "sd-scripts", "train_network.py")
print("TOML written to:", toml_path)
print()
print("Launch command:")
print(
    f'uv run accelerate launch --num_processes 1 --num_machines 1 --mixed_precision fp16 "{script}" --config_file "{toml_path}"'
)
