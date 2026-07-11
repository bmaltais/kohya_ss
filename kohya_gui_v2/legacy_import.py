"""Legacy JSON preset importer (Move 8).

The six `kohya_gui/*_gui.py` tabs persist presets as JSON in GUI-parameter
space (`save_configuration`'s positional signature), not the flat TOML the
v2 registries save/load. `import_json` bridges a legacy JSON preset into a
v2 FieldRegistry's TOML-key-space values dict, using the same rename
tables and architecture-detection heuristics validated by the Move 7
equivalence harnesses (tests/test_v2_equivalence_*.py) -- this module is
the single source of truth for that mapping; the batch converter
(tools/convert_presets_to_toml.py) and every tab's Open-dialog JSON path
both route through it rather than keeping their own copies.

`detect_training_type` lets a generic caller (the batch converter) figure
out which of the six registries a JSON file belongs to from its keys alone,
using markers confirmed by direct `inspect.signature` introspection against
each `train_model()` (not guessed -- see the docstring on the function).
A per-tab Open button already knows its own training type from context, so
it should pass `training_type=` explicitly rather than relying on
detection (a mismatch there is a user error worth flagging, not silently
overriding the tab the user is in).
"""

import inspect
import json
import re
from dataclasses import dataclass, field

from .fields import FieldRegistry
from .tabs import (
    anima_lllite_derivations,
    dreambooth_derivations,
    finetune_derivations,
    leco_derivations,
    lora_derivations,
    textual_inversion_derivations,
)
from .tabs.anima_lllite_fields import ANIMA_LLLITE_REGISTRY
from .tabs.anima_lllite_fields import derive as _derive_anima_lllite
from .tabs.dreambooth_fields import DREAMBOOTH_REGISTRY
from .tabs.dreambooth_fields import derive as _derive_dreambooth
from .tabs.finetune_fields import FINETUNE_REGISTRY
from .tabs.finetune_fields import derive as _derive_finetune
from .tabs.leco_fields import LECO_REGISTRY
from .tabs.leco_fields import derive as _derive_leco
from .tabs.lora_fields import LORA_REGISTRY
from .tabs.lora_fields import derive as _derive_lora
from .tabs.textual_inversion_fields import TEXTUAL_INVERSION_REGISTRY
from .tabs.textual_inversion_fields import derive as _derive_textual_inversion

REGISTRIES = {
    "lora": LORA_REGISTRY,
    "dreambooth": DREAMBOOTH_REGISTRY,
    "finetune": FINETUNE_REGISTRY,
    "textual_inversion": TEXTUAL_INVERSION_REGISTRY,
    "leco": LECO_REGISTRY,
    "anima_lllite": ANIMA_LLLITE_REGISTRY,
}

DERIVE_FUNCS = {
    "lora": _derive_lora,
    "dreambooth": _derive_dreambooth,
    "finetune": _derive_finetune,
    "textual_inversion": _derive_textual_inversion,
    "leco": _derive_leco,
    "anima_lllite": _derive_anima_lllite,
}

# GUI param name -> v2 FieldSpec (TOML key) name, per training type. Mirrors
# the RENAME_MAP tables validated by each tests/test_v2_equivalence_*.py
# harness (Move 7) -- this module is the canonical copy going forward.
RENAME_MAPS = {
    "lora": {
        "optimizer": "optimizer_type",
        "max_resolution": "resolution",
    },
    "dreambooth": {
        "optimizer": "optimizer_type",
        "max_resolution": "resolution",
        "sd3_text_encoder_batch_size": "text_encoder_batch_size",
    },
    "finetune": {
        "optimizer": "optimizer_type",
        "max_resolution": "resolution",
        "sd3_text_encoder_batch_size": "text_encoder_batch_size",
    },
    "textual_inversion": {
        "optimizer": "optimizer_type",
        "max_resolution": "resolution",
    },
    "leco": {
        "optimizer": "optimizer_type",
    },
    "anima_lllite": {
        "optimizer": "optimizer_type",
        "anima_qwen3": "qwen3",
        "anima_vae": "vae",
        "anima_llm_adapter_path": "llm_adapter_path",
        "anima_t5_tokenizer_path": "t5_tokenizer_path",
        "anima_discrete_flow_shift": "discrete_flow_shift",
        "anima_timestep_sampling": "timestep_sampling",
        "anima_sigmoid_scale": "sigmoid_scale",
        "anima_qwen3_max_token_length": "qwen3_max_token_length",
        "anima_t5_max_token_length": "t5_max_token_length",
        "anima_attn_mode": "attn_mode",
        "anima_split_attn": "split_attn",
        "anima_vae_chunk_size": "vae_chunk_size",
        "anima_vae_disable_cache": "vae_disable_cache",
        "anima_qwen_image_vae_2d": "qwen_image_vae_2d",
        "anima_compile": "compile",
        "anima_torch_compile": "torch_compile",
        "anima_compile_backend": "compile_backend",
        "anima_compile_mode": "compile_mode",
        "anima_compile_dynamic": "compile_dynamic",
        "anima_compile_fullgraph": "compile_fullgraph",
        "anima_compile_cache_size_limit": "compile_cache_size_limit",
    },
}


def _architecture_for_lora(values: dict) -> str:
    if values.get("sdxl"):
        return "sdxl"
    if values.get("flux1_checkbox"):
        return "flux1"
    if values.get("sd3_checkbox"):
        return "sd3"
    if values.get("hunyuan_image_checkbox"):
        return "hunyuan_image"
    if values.get("anima_checkbox"):
        return "anima"
    if values.get("lumina_checkbox"):
        return "lumina"
    return "sd15"


def _architecture_for_dreambooth(values: dict) -> str:
    if values.get("sdxl"):
        return "sdxl"
    if values.get("sd3_checkbox"):
        return "sd3"
    if values.get("flux1_checkbox"):
        return "flux1"
    return "sd15"


def _architecture_for_finetune(values: dict) -> str:
    if values.get("sdxl_checkbox"):
        return "sdxl"
    if values.get("sd3_checkbox"):
        return "sd3"
    if values.get("flux1_checkbox"):
        return "flux1"
    if values.get("anima_checkbox"):
        return "anima"
    if values.get("lumina_checkbox"):
        return "lumina"
    return "base"


def _architecture_for_textual_inversion(values: dict) -> str:
    return "sdxl" if values.get("sdxl") else "sd_v1v2"


def _architecture_for_leco(values: dict) -> str:
    return "sdxl" if values.get("sdxl") else "sd15"


def _architecture_for_anima_lllite(values: dict) -> str:
    return "anima_lllite"


ARCH_DETECTORS = {
    "lora": _architecture_for_lora,
    "dreambooth": _architecture_for_dreambooth,
    "finetune": _architecture_for_finetune,
    "textual_inversion": _architecture_for_textual_inversion,
    "leco": _architecture_for_leco,
    "anima_lllite": _architecture_for_anima_lllite,
}

DERIVATION_MODULES = {
    "lora": lora_derivations,
    "dreambooth": dreambooth_derivations,
    "finetune": finetune_derivations,
    "textual_inversion": textual_inversion_derivations,
    "leco": leco_derivations,
    "anima_lllite": anima_lllite_derivations,
}

_SOURCE_KEY_PATTERN = re.compile(r'values(?:\.get)?\(?\[?"([a-zA-Z0-9_]+)"')

# A handful of derivations build the source key dynamically
# (f"{arch_key}_cache_text_encoder_outputs") rather than as a string
# literal, so the regex above can't see them. Enumerated here per the
# per-arch prefixes those two derivations actually use (dreambooth_
# derivations.py / finetune_derivations.py).
_DYNAMIC_SOURCE_KEY_SUFFIXES = (
    "_cache_text_encoder_outputs",
    "_cache_text_encoder_outputs_to_disk",
)
_DYNAMIC_SOURCE_KEY_PREFIXES = ("sdxl", "sd3", "flux1", "anima", "lumina")
EXTRA_SOURCE_KEYS = {
    "dreambooth": {
        f"{prefix}{suffix}"
        for prefix in _DYNAMIC_SOURCE_KEY_PREFIXES
        for suffix in _DYNAMIC_SOURCE_KEY_SUFFIXES
    },
    "finetune": {
        f"{prefix}{suffix}"
        for prefix in _DYNAMIC_SOURCE_KEY_PREFIXES
        for suffix in _DYNAMIC_SOURCE_KEY_SUFFIXES
    },
}


def _composite_source_keys_for(training_type: str) -> set:
    """Every raw widget name a training type's derive()/architecture
    detector reads from the `values` dict, extracted directly from source
    (via regex over `values.get("name")` / `values["name"]` call sites)
    rather than hand-maintained -- so this list can't silently drift out of
    sync with the derivation code the way a manually-copied one would.
    """
    keys = set()
    for fn in (DERIVATION_MODULES[training_type].derive, ARCH_DETECTORS[training_type]):
        keys.update(_SOURCE_KEY_PATTERN.findall(inspect.getsource(fn)))
    keys |= EXTRA_SOURCE_KEYS.get(training_type, set())
    return keys


def detect_training_type(cfg: dict) -> str:
    """Detect which of the six training types a legacy JSON preset belongs
    to, using keys unique to that type's `train_model()` signature
    (confirmed via direct `inspect.signature` introspection -- see the
    Move 8 recon note in wargame/2026-07-11-musubi-style-gui-v2.md).

    Checks run most-distinctive-first: `token_string`/`init_word` only
    exist on Textual Inversion; `prompts_file` + `network_module` only
    co-occur on LeCo (LoRA never has a literal `network_module` param --
    its network module is derived from a LoRA_type dropdown); `lllite_mlp_dim`
    only exists on Anima LLLite; `use_latent_files`/`generate_caption_database`/
    `dataset_repeats` only exist on Finetune; `network_dim` narrows the
    remainder to LoRA; anything left over is DreamBooth (no single
    DreamBooth-unique key exists, so it is the fallback by elimination).
    """
    keys = set(cfg.keys())
    if "token_string" in keys and "init_word" in keys:
        return "textual_inversion"
    if "prompts_file" in keys and "network_module" in keys:
        return "leco"
    if "lllite_mlp_dim" in keys:
        return "anima_lllite"
    if keys & {"use_latent_files", "generate_caption_database", "dataset_repeats"}:
        return "finetune"
    if "network_dim" in keys:
        return "lora"
    return "dreambooth"


@dataclass
class ImportResult:
    training_type: str
    architecture: str
    values: dict
    unrecognized_keys: list = field(default_factory=list)


def import_json(path: str, training_type: str = None) -> ImportResult:
    """Import a legacy JSON preset into a v2 values dict.

    `training_type`, if given, is trusted (the calling tab already knows
    which registry it owns); otherwise it is detected from the file's keys.
    Keys present in the JSON that neither the rename map nor the target
    registry recognizes are collected into `unrecognized_keys` instead of
    being silently dropped, so callers (the Open-dialog wiring) can warn
    the user (per the wargame plan's counter-move for this Move).
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if training_type is None:
        training_type = detect_training_type(cfg)

    registry: FieldRegistry = REGISTRIES[training_type]
    rename_map = RENAME_MAPS[training_type]

    renamed = dict(cfg)
    for gui_name, v2_name in rename_map.items():
        if gui_name in cfg:
            renamed[v2_name] = cfg[gui_name]

    values = {}
    for spec in registry:
        if spec.gui_only:
            continue
        values[spec.name] = renamed.get(spec.name, spec.default)

    arch_key = ARCH_DETECTORS[training_type](renamed)
    values.update(DERIVE_FUNCS[training_type](renamed, arch_key))

    # A raw JSON key is "recognized" if it's a rename-map source name, a
    # FieldSpec's own name (both consumed directly above), or a composite
    # source widget the derive()/architecture-detector functions read (e.g.
    # `sdxl_checkbox`, `sdxl_no_half_vae`) -- anything else genuinely wasn't
    # consulted anywhere in this import (older GUI version, user-edited
    # file, or a real gap-analysis miss).
    recognized = (
        set(rename_map.keys())
        | {spec.name for spec in registry}
        | _composite_source_keys_for(training_type)
    )
    unrecognized = sorted(k for k in cfg.keys() if k not in recognized)

    return ImportResult(
        training_type=training_type,
        architecture=arch_key,
        values=values,
        unrecognized_keys=unrecognized,
    )
