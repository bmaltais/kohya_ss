"""Extract FieldLayout map from legacy kohya_gui/class_*.py widgets.

Emits kohya_gui_v2/layout_map.py — plain data used by generators (group/label/info)
and by tab_builder (section order, rows, accordion titles).

Run: uv run python scripts/extract_layout_map.py
"""

from __future__ import annotations

import ast
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
GUI_DIR = ROOT / "kohya_gui"
OUT_PATH = ROOT / "kohya_gui_v2" / "layout_map.py"

# class file stem -> section key
CLASS_SECTION = {
    "class_accelerate_launch": "accelerate_launch",
    "class_source_model": "model",
    "class_folders": "folders",
    "class_metadata": "metadata",
    "class_basic_training": "basic",
    "class_advanced_training": "advanced",
    "class_sample_images": "samples",
    "class_huggingface": "huggingface",
    # Arch-specific parameter panels → advanced (legacy nests them under Parameters)
    "class_flux1": "advanced",
    "class_sd3": "advanced",
    "class_sdxl_parameters": "advanced",
    "class_anima": "advanced",
    "class_hunyuan_image": "advanced",
    "class_lumina": "advanced",
    "class_tensorboard": "advanced",
}

# Legacy attr name → argparse dest / FieldSpec name
NAME_ALIASES = {
    "optimizer": "optimizer_type",
    "learning_rate_te": "text_encoder_lr",
    "learning_rate_te1": "text_encoder_lr",
    "learning_rate_te2": "text_encoder_lr",
    "epoch": "max_train_epochs",  # closest; max_train_epochs already extracted too
    "lr_warmup": "lr_warmup_steps",
    "max_resolution": "resolution",
    "logging_dir": "logging_dir",
    "no_token_padding": "no_token_padding",
    "lowvram": "lowram",
    "flux1_cache_text_encoder_outputs": "cache_text_encoder_outputs",
    "flux1_cache_text_encoder_outputs_to_disk": "cache_text_encoder_outputs_to_disk",
}

# Network / LoRA-structure fields (section "network")
NETWORK_FIELDS = frozenset(
    {
        "network_dim",
        "network_alpha",
        "network_module",
        "network_args",
        "network_dropout",
        "network_train_unet_only",
        "network_train_text_encoder_only",
        "network_weights",
        "dim_from_weights",
        "scale_weight_norms",
        "base_weights",
        "base_weights_multiplier",
        "conv_dim",
        "conv_alpha",
        "rank_dropout",
        "module_dropout",
        "unit",
        "block_dims",
        "block_alphas",
        "conv_block_dims",
        "conv_block_alphas",
        "down_lr_weight",
        "mid_lr_weight",
        "up_lr_weight",
        "block_lr_zero_threshold",
        "factor",
        "bypass_mode",
        "dora_wd",
        "use_tucker",
        "use_scalar",
        "use_cp",
        "decompose_both",
        "train_norm",
        "train_on_input",
        "rescaled",
        "constrain",
        "loraplus_lr_ratio",
        "loraplus_unet_lr_ratio",
        "loraplus_text_encoder_lr_ratio",
    }
)

# Model / path-ish names that may only appear in generated specs
MODEL_FIELDS = frozenset(
    {
        "pretrained_model_name_or_path",
        "train_data_dir",
        "dataset_config",
        "output_name",
        "training_comment",
        "save_model_as",
        "save_precision",
        "v2",
        "v_parameterization",
        "clip_l",
        "clip_g",
        "t5xxl",
        "ae",
        "vae",
        "checkpoint_file",
    }
)

FOLDERS_FIELDS = frozenset(
    {
        "output_dir",
        "reg_data_dir",
        "logging_dir",
    }
)

SAMPLES_FIELDS = frozenset(
    {
        "sample_prompts",
        "sample_every_n_steps",
        "sample_every_n_epochs",
        "sample_sampler",
        "sample_at_first",
    }
)

HUGGINGFACE_FIELDS = frozenset(
    {
        "huggingface_repo_id",
        "huggingface_token",
        "huggingface_repo_type",
        "huggingface_repo_visibility",
        "huggingface_path_in_repo",
        "save_state_to_huggingface",
        "resume_from_huggingface",
        "async_upload",
    }
)

METADATA_FIELDS = frozenset(
    {
        "metadata_title",
        "metadata_author",
        "metadata_description",
        "metadata_license",
        "metadata_tags",
        "metadata_is_negative_embedding",
        "metadata_merged_from",
        "metadata_preprocessor",
        "metadata_thumbnail",
        "metadata_trigger_phrase",
        "metadata_usage_hint",
    }
)

ACCELERATE_FIELDS = frozenset(
    {
        "mixed_precision",
        "num_cpu_threads_per_process",
        "num_processes",
        "num_machines",
        "multi_gpu",
        "gpu_ids",
        "main_process_port",
        "dynamo_backend",
        "dynamo_mode",
        "dynamo_use_fullgraph",
        "dynamo_use_dynamic",
        "extra_accelerate_launch_args",
    }
)

SECTION_TITLES = {
    "model": "Model",
    "folders": "Folders",
    "dataset": "Dataset",
    "basic": "Basic",
    "network": "Network",
    "advanced": "Advanced",
    "samples": "Samples",
    "huggingface": "HuggingFace",
    "metadata": "Metadata",
    "accelerate_launch": "Accelerate launch",
}

SECTION_ORDER = [
    "model",
    "folders",
    "dataset",
    "basic",
    "network",
    "advanced",
    "samples",
    "huggingface",
    "metadata",
    "accelerate_launch",
]

OPEN_BY_DEFAULT = frozenset({"model", "folders", "basic"})

# Curated dropdown choices (legacy GUI free-text→dropdown); allow_custom noted in builder
CURATED_CHOICES = {
    "optimizer_type": [
        "AdamW",
        "AdamWScheduleFree",
        "AdamW8bit",
        "Adafactor",
        "bitsandbytes.optim.AdEMAMix8bit",
        "bitsandbytes.optim.PagedAdEMAMix8bit",
        "DAdaptation",
        "DAdaptAdaGrad",
        "DAdaptAdam",
        "DAdaptAdan",
        "DAdaptAdanIP",
        "DAdaptAdamPreprint",
        "DAdaptLion",
        "DAdaptSGD",
        "Lion",
        "Lion8bit",
        "PagedAdamW8bit",
        "PagedAdamW32bit",
        "PagedLion8bit",
        "Prodigy",
        "prodigyplus.ProdigyPlusScheduleFree",
        "pytorch_optimizer.CAME",
        "RAdamScheduleFree",
        "SGDNesterov",
        "SGDNesterov8bit",
        "SGDScheduleFree",
    ],
    "lr_scheduler": [
        "adafactor",
        "constant",
        "constant_with_warmup",
        "cosine",
        "cosine_with_restarts",
        "linear",
        "piecewise_constant",
        "polynomial",
        "cosine_with_min_lr",
        "inverse_sqrt",
        "warmup_stable_decay",
    ],
    "lr_scheduler_type": [
        "",
        "CosineAnnealingLR",
    ],
    "caption_extension": [
        "",
        ".cap",
        ".caption",
        ".txt",
    ],
    # Align with kohya_gui/class_accelerate_launch.py (not train_util's
    # incomplete list which omits "no" and misspells tensorrt as "tensort").
    # Legacy JSON presets almost always store dynamo_backend="no".
    "dynamo_backend": [
        "no",
        "eager",
        "aot_eager",
        "inductor",
        "aot_ts_nvfuser",
        "nvprims_nvfuser",
        "cudagraphs",
        "ofi",
        "fx2trt",
        "onnxrt",
        "tensorrt",
        "ipex",
        "tvm",
    ],
    # Align with kohya_gui/class_advanced_training.py. Argparse only allows
    # [None, 150, 225] (None → 75 token default), but the legacy GUI and every
    # preset stores the explicit 75 choice; without it Gradio rejects open/save.
    "max_token_length": [
        75,
        150,
        225,
    ],
}

# Defaults that must win over argparse introspection (same reason as CURATED_CHOICES).
CURATED_DEFAULTS = {
    "dynamo_backend": "no",
    "max_token_length": 75,
}

# Hand-curated high-traffic rows (used when AST leaves row=None or to force clusters)
HAND_CURATED_ROWS = {
    "learning_rate": "basic.lr_family",
    "unet_lr": "basic.lr_family",
    "text_encoder_lr": "basic.lr_family",
    "lr_warmup_steps": "basic.lr_family",
    "train_batch_size": "basic.batch_epoch",
    "max_train_epochs": "basic.batch_epoch",
    "max_train_steps": "basic.batch_epoch",
    "save_every_n_epochs": "basic.batch_epoch",
    "seed": "basic.seed_cache",
    "cache_latents": "basic.seed_cache",
    "cache_latents_to_disk": "basic.seed_cache",
    "lr_scheduler": "basic.opt_sched",
    "lr_scheduler_type": "basic.opt_sched",
    "optimizer_type": "basic.opt_sched",
    "max_grad_norm": "basic.grad_args",
    "lr_scheduler_args": "basic.grad_args",
    "optimizer_args": "basic.grad_args",
    "output_dir": "folders.paths",
    "reg_data_dir": "folders.paths",
    "logging_dir": "folders.logging",
    "pretrained_model_name_or_path": "model.source",
    "output_name": "model.source",
    "train_data_dir": "model.data",
    "dataset_config": "model.data",
    "network_dim": "network.dim_alpha",
    "network_alpha": "network.dim_alpha",
    "network_dropout": "network.dim_alpha",
    "sample_every_n_steps": "samples.timing",
    "sample_every_n_epochs": "samples.timing",
    "sample_sampler": "samples.timing",
}

# Force section (and optional label) when AST misses or name is registry-only
HAND_SECTION_AND_LABEL: dict[str, tuple[str, str | None]] = {
    "learning_rate": ("basic", "Learning rate"),
    "unet_lr": ("basic", "U-Net learning rate"),
    "text_encoder_lr": ("basic", "Text encoder learning rate"),
    "lr_warmup_steps": ("basic", "LR warmup steps"),
    "train_batch_size": ("basic", "Train batch size"),
    "max_train_epochs": ("basic", "Max train epochs"),
    "max_train_steps": ("basic", "Max train steps"),
    "save_every_n_epochs": ("basic", "Save every N epochs"),
    "save_every_n_steps": ("basic", "Save every N steps"),
    "seed": ("basic", "Seed"),
    "cache_latents": ("basic", "Cache latents"),
    "cache_latents_to_disk": ("basic", "Cache latents to disk"),
    "optimizer_type": ("basic", "Optimizer"),
    "optimizer_args": ("basic", "Optimizer extra arguments"),
    "lr_scheduler": ("basic", "LR Scheduler"),
    "lr_scheduler_type": ("basic", "LR Scheduler type"),
    "lr_scheduler_args": ("basic", "LR scheduler extra arguments"),
    "lr_scheduler_num_cycles": ("basic", "LR scheduler num cycles"),
    "lr_scheduler_power": ("basic", "LR scheduler power"),
    "max_grad_norm": ("basic", "Max grad norm"),
    "gradient_checkpointing": ("basic", "Gradient checkpointing"),
    "gradient_accumulation_steps": ("basic", "Gradient accumulation steps"),
    "caption_extension": ("basic", "Caption file extension"),
    "resolution": ("basic", "Resolution"),
    "enable_bucket": ("basic", "Enable buckets"),
    "min_bucket_reso": ("basic", "Min bucket resolution"),
    "max_bucket_reso": ("basic", "Max bucket resolution"),
    "network_dim": ("network", "Network Rank (Dimension)"),
    "network_alpha": ("network", "Network Alpha"),
    "network_module": ("network", "Network module"),
    "network_args": ("network", "Network args"),
    "network_dropout": ("network", "Network dropout"),
    "network_weights": ("network", "Network weights"),
    "network_train_unet_only": ("network", "Train U-Net only"),
    "network_train_text_encoder_only": ("network", "Train text encoder only"),
    "sample_prompts": ("samples", "Sample prompts"),
    "mixed_precision": ("accelerate_launch", "Mixed precision"),
    "huggingface_repo_id": ("huggingface", "HuggingFace repo id"),
}

# Prefix / substring rules for gap fields → keep Advanced from swallowing everything
# Applied only when field would otherwise be fallback advanced.
PREFIX_SECTION_RULES: list[tuple[str, str]] = [
    ("sample_", "samples"),
    ("huggingface_", "huggingface"),
    ("metadata_", "metadata"),
    ("network_", "network"),
    ("caption_", "dataset"),
    ("dataset_", "dataset"),
    ("bucket_", "dataset"),
    ("keep_tokens", "dataset"),
    ("shuffle_caption", "dataset"),
    ("token_", "dataset"),
    ("color_aug", "dataset"),
    ("flip_aug", "dataset"),
    ("random_crop", "dataset"),
    ("face_crop", "dataset"),
    ("debug_dataset", "dataset"),
    ("resolution", "basic"),
    ("learning_rate", "basic"),
    ("lr_", "basic"),
    ("optimizer", "basic"),
    ("save_", "basic"),
    ("max_train", "basic"),
    ("train_batch", "basic"),
    ("seed", "basic"),
    ("cache_latent", "basic"),
    ("gradient_", "basic"),
    ("mixed_precision", "accelerate_launch"),
    ("dynamo_", "accelerate_launch"),
    ("log_", "advanced"),
    ("wandb", "advanced"),
    ("logging", "folders"),
]


@dataclass
class Extracted:
    name: str
    section: str
    row: Optional[str]
    label: Optional[str]
    info: Optional[str]
    source: str


def _const_str(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    # label=("a" if x else "b") — skip non-constants
    return None


class ClassExtractor(ast.NodeVisitor):
    WIDGETS = frozenset(
        {"Textbox", "Number", "Checkbox", "Dropdown", "Slider", "Radio"}
    )

    def __init__(self, section: str, file_stem: str):
        self.section = section
        self.file_stem = file_stem
        self.row_stack: list[str] = []
        self.row_counter = 0
        self.fields: list[Extracted] = []

    def visit_With(self, node: ast.With):
        is_row = False
        for item in node.items:
            ctx = item.context_expr
            if isinstance(ctx, ast.Call):
                f = ctx.func
                if isinstance(f, ast.Attribute) and f.attr == "Row":
                    is_row = True
                elif isinstance(f, ast.Name) and f.id == "Row":
                    is_row = True
        if is_row:
            self.row_counter += 1
            rid = f"{self.section}.{self.file_stem}.row_{self.row_counter}"
            self.row_stack.append(rid)
            self.generic_visit(node)
            self.row_stack.pop()
        else:
            self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) != 1:
            self.generic_visit(node)
            return
        t = node.targets[0]
        if not (
            isinstance(t, ast.Attribute)
            and isinstance(t.value, ast.Name)
            and t.value.id == "self"
        ):
            self.generic_visit(node)
            return
        name = t.attr
        if not isinstance(node.value, ast.Call):
            self.generic_visit(node)
            return
        call = node.value
        f = call.func
        widget = None
        if isinstance(f, ast.Attribute):
            widget = f.attr
        elif isinstance(f, ast.Name):
            widget = f.id
        if widget not in self.WIDGETS:
            self.generic_visit(node)
            return
        label = info = None
        for kw in call.keywords:
            if kw.arg == "label":
                label = _const_str(kw.value)
            elif kw.arg == "info":
                info = _const_str(kw.value)
        row = self.row_stack[-1] if self.row_stack else None
        self.fields.append(
            Extracted(
                name=name,
                section=self.section,
                row=row,
                label=label,
                info=info,
                source=self.file_stem,
            )
        )
        self.generic_visit(node)


# fn name (as called via button.click(fn, ..., outputs=self.X)) -> picker kind.
# get_file_path/get_any_file_path both open a file dialog; get_folder_path a
# folder dialog. Mirrors kohya_gui/common_gui.py's three dialog helpers.
PATH_FN_NAMES = {
    "get_folder_path": "folder",
    "get_file_path": "file",
    "get_any_file_path": "file",
}


def _call_fn_name(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Lambda) and isinstance(node.body, ast.Call):
        return _call_fn_name(node.body.func)
    return None


def extract_path_fields() -> dict[str, str]:
    """Scan every legacy class_*.py for ``<button>.click(get_folder_path|
    get_file_path|get_any_file_path, ..., outputs=self.<field>)`` wiring and
    return ``{field_name: "file"|"folder"}``.

    This is how the legacy GUI decides which fields get a native-dialog
    Browse button; v2 reuses the same decision instead of re-guessing it.
    """
    kinds: dict[str, str] = {}
    for path in sorted(GUI_DIR.glob("class_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "click"
            ):
                continue
            fn_node = node.args[0] if node.args else None
            for kw in node.keywords:
                if kw.arg == "fn":
                    fn_node = kw.value
            kind = PATH_FN_NAMES.get(_call_fn_name(fn_node))
            if not kind:
                continue
            outputs_node = node.args[1] if len(node.args) > 1 else None
            for kw in node.keywords:
                if kw.arg == "outputs":
                    outputs_node = kw.value
            if (
                isinstance(outputs_node, ast.Attribute)
                and isinstance(outputs_node.value, ast.Name)
                and outputs_node.value.id == "self"
            ):
                field_name = NAME_ALIASES.get(outputs_node.attr, outputs_node.attr)
                kinds.setdefault(field_name, kind)
    return kinds


# Conservative fallback for fields with no legacy widget at all (gap-analysis
# additions): only classify unambiguous folder-suffixed names. Deliberately
# does NOT guess "file" — misclassifying a folder field as file (or vice
# versa) opens the wrong dialog, which is worse than no button at all.
def path_kind_heuristic(name: str) -> Optional[str]:
    if name.endswith(("_dir", "_folder")):
        return "folder"
    return None


def extract_all() -> dict[str, Extracted]:
    """Return best Extracted per field name (first section wins; network override later)."""
    by_name: dict[str, Extracted] = {}
    for path in sorted(GUI_DIR.glob("class_*.py")):
        stem = path.stem
        section = CLASS_SECTION.get(stem)
        if not section:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        vis = ClassExtractor(section, stem)
        vis.visit(tree)
        for ext in vis.fields:
            canonical = NAME_ALIASES.get(ext.name, ext.name)
            # Prefer first hit; keep richer label/info if later overwrites empty
            if canonical not in by_name:
                by_name[canonical] = Extracted(
                    name=canonical,
                    section=ext.section,
                    row=ext.row,
                    label=ext.label,
                    info=ext.info,
                    source=ext.source,
                )
            else:
                cur = by_name[canonical]
                if cur.label is None and ext.label:
                    cur.label = ext.label
                if cur.info is None and ext.info:
                    cur.info = ext.info
                if cur.row is None and ext.row:
                    cur.row = ext.row
    return by_name


def apply_section_overrides(by_name: dict[str, Extracted]) -> None:
    overrides = [
        (NETWORK_FIELDS, "network"),
        (MODEL_FIELDS, "model"),
        (FOLDERS_FIELDS, "folders"),
        (SAMPLES_FIELDS, "samples"),
        (HUGGINGFACE_FIELDS, "huggingface"),
        (METADATA_FIELDS, "metadata"),
        (ACCELERATE_FIELDS, "accelerate_launch"),
    ]
    for names, section in overrides:
        for n in names:
            if n in by_name:
                by_name[n].section = section
            else:
                # seed placeholder so LAYOUT has section even without legacy widget
                by_name[n] = Extracted(
                    name=n,
                    section=section,
                    row=None,
                    label=None,
                    info=None,
                    source="override",
                )


def apply_hand_rows(by_name: dict[str, Extracted]) -> None:
    for name, row in HAND_CURATED_ROWS.items():
        if name in by_name:
            by_name[name].row = row
        else:
            # infer section from row prefix
            section = row.split(".", 1)[0]
            by_name[name] = Extracted(
                name=name,
                section=section,
                row=row,
                label=None,
                info=None,
                source="hand_row",
            )


def apply_hand_section_labels(by_name: dict[str, Extracted]) -> None:
    for name, (section, label) in HAND_SECTION_AND_LABEL.items():
        if name in by_name:
            by_name[name].section = section
            if label:
                by_name[name].label = label
        else:
            by_name[name] = Extracted(
                name=name,
                section=section,
                row=HAND_CURATED_ROWS.get(name),
                label=label,
                info=None,
                source="hand_section",
            )


def section_from_name_heuristics(name: str) -> Optional[str]:
    for prefix, section in PREFIX_SECTION_RULES:
        if name == prefix.rstrip("_") or name.startswith(prefix):
            return section
    return None


def auto_label(name: str) -> str:
    return name.replace("_", " ").capitalize()


def collect_all_registry_names() -> set[str]:
    """Import all generated field lists without building Gradio."""
    import sys

    sys.path.insert(0, str(ROOT))
    names: set[str] = set()
    modules = [
        "kohya_gui_v2.tabs.lora_fields_generated",
        "kohya_gui_v2.tabs.dreambooth_fields_generated",
        "kohya_gui_v2.tabs.finetune_fields_generated",
        "kohya_gui_v2.tabs.textual_inversion_fields_generated",
        "kohya_gui_v2.tabs.anima_lllite_fields_generated",
        "kohya_gui_v2.tabs.leco_fields_generated",
    ]
    attr_names = [
        "LORA_FIELDS",
        "DREAMBOOTH_FIELDS",
        "FINETUNE_FIELDS",
        "TEXTUAL_INVERSION_FIELDS",
        "ANIMA_LLLITE_FIELDS",
        "LECO_FIELDS",
    ]
    import importlib

    for mod_name, attr in zip(modules, attr_names):
        mod = importlib.import_module(mod_name)
        for spec in getattr(mod, attr):
            names.add(spec.name)
    # gui-only
    names.update({"architecture", "training_type"})
    return names


def emit_layout_map(
    by_name: dict[str, Extracted],
    all_names: set[str],
    path_kinds: dict[str, str],
) -> str:
    # Ensure every registry name has an entry
    for n in sorted(all_names):
        if n not in by_name:
            section = "advanced"
            if n in NETWORK_FIELDS:
                section = "network"
            elif n in MODEL_FIELDS:
                section = "model"
            elif n in FOLDERS_FIELDS:
                section = "folders"
            elif n in SAMPLES_FIELDS:
                section = "samples"
            elif n in HUGGINGFACE_FIELDS:
                section = "huggingface"
            elif n in METADATA_FIELDS:
                section = "metadata"
            elif n in ACCELERATE_FIELDS:
                section = "accelerate_launch"
            elif n in ("architecture", "training_type"):
                section = "model"
            else:
                guessed = section_from_name_heuristics(n)
                if guessed:
                    section = guessed
            by_name[n] = Extracted(
                name=n,
                section=section,
                row=HAND_CURATED_ROWS.get(n),
                label=None,
                info=None,
                source="fallback",
            )
        else:
            # Re-home pure advanced fallbacks via heuristics when still advanced
            e = by_name[n]
            if e.source in ("fallback", "override") and e.section == "advanced":
                guessed = section_from_name_heuristics(n)
                if guessed:
                    e.section = guessed

    lines = [
        '"""Layout map for v2 GUI — section / row / label / info per field name.',
        "",
        "Generated by scripts/extract_layout_map.py from kohya_gui/class_*.py.",
        "Regenerate: uv run python scripts/extract_layout_map.py",
        "Hand edits to CURATED_CHOICES / HAND rows should be made in the extractor.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import NamedTuple, Optional",
        "",
        "",
        "class FieldLayout(NamedTuple):",
        "    section: str  # accordion group key",
        "    row: Optional[str]  # shared key packs fields into gr.Row; None = own line",
        "    label: Optional[str]",
        "    info: Optional[str]",
        "",
        "",
        "SECTION_ORDER = [",
    ]
    for s in SECTION_ORDER:
        lines.append(f"    {s!r},")
    lines.append("]")
    lines.append("")
    lines.append("SECTION_TITLES = {")
    for k, v in SECTION_TITLES.items():
        lines.append(f"    {k!r}: {v!r},")
    lines.append("}")
    lines.append("")
    lines.append("OPEN_BY_DEFAULT = frozenset({")
    for s in sorted(OPEN_BY_DEFAULT):
        lines.append(f"    {s!r},")
    lines.append("})")
    lines.append("")
    lines.append("CURATED_CHOICES: dict[str, list] = {")
    for k, choices in CURATED_CHOICES.items():
        lines.append(f"    {k!r}: [")
        for c in choices:
            lines.append(f"        {c!r},")
        lines.append("    ],")
    lines.append("}")
    lines.append("")
    lines.append(
        "# Defaults that must win over argparse introspection (same reason as CURATED_CHOICES)."
    )
    lines.append("CURATED_DEFAULTS: dict[str, object] = {")
    for k, v in CURATED_DEFAULTS.items():
        lines.append(f"    {k!r}: {v!r},")
    lines.append("}")
    lines.append("")
    lines.append(
        "# Fields that use CURATED_CHOICES and allow typing values not in the list"
    )
    lines.append(
        "CURATED_ALLOW_CUSTOM = frozenset("
        '{"optimizer_type", "lr_scheduler", "lr_scheduler_type", "caption_extension"})'
    )
    lines.append("")
    lines.append(
        '# "file" or "folder" — which native dialog the legacy GUI wires a Browse'
    )
    lines.append(
        "# button to for this field (extract_path_fields() in the generator script);"
    )
    lines.append("# absent means the legacy GUI has no picker for it either.")
    lines.append("PATH_FIELDS: dict[str, str] = {")
    for k in sorted(path_kinds):
        lines.append(f"    {k!r}: {path_kinds[k]!r},")
    lines.append("}")
    lines.append("")
    lines.append("")
    lines.append("def path_kind_for(name: str) -> Optional[str]:")
    lines.append('    """Return "file", "folder", or None for a field name."""')
    lines.append("    return PATH_FIELDS.get(name)")
    lines.append("")
    lines.append("LAYOUT: dict[str, FieldLayout] = {")
    for n in sorted(by_name.keys()):
        e = by_name[n]
        label = e.label
        info = e.info
        row = e.row
        # Prefer hand-curated row
        if n in HAND_CURATED_ROWS:
            row = HAND_CURATED_ROWS[n]
        lines.append(
            f"    {n!r}: FieldLayout("
            f"section={e.section!r}, row={row!r}, label={label!r}, info={info!r}),"
        )
    lines.append("}")
    lines.append("")
    lines.append("")
    lines.append("def layout_for(name: str) -> FieldLayout:")
    lines.append(
        '    """Return layout for a field; unknown names fall back to advanced."""'
    )
    lines.append("    if name in LAYOUT:")
    lines.append("        return LAYOUT[name]")
    lines.append(
        '    return FieldLayout(section="advanced", row=None, '
        "label=name.replace('_', ' ').capitalize(), info=None)"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    by_name = extract_all()
    apply_section_overrides(by_name)
    apply_hand_rows(by_name)
    apply_hand_section_labels(by_name)
    all_names = collect_all_registry_names()
    path_kinds = extract_path_fields()
    for n in all_names:
        if n not in path_kinds:
            guess = path_kind_heuristic(n)
            if guess:
                path_kinds[n] = guess
    text = emit_layout_map(by_name, all_names, path_kinds)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text, encoding="utf-8")

    # Coverage report
    unmapped_advanced = [
        n
        for n in sorted(all_names)
        if by_name.get(n) and by_name[n].source == "fallback"
    ]
    sections: dict[str, int] = {}
    for n in all_names:
        sec = by_name[n].section if n in by_name else "advanced"
        sections[sec] = sections.get(sec, 0) + 1

    print(f"Wrote {OUT_PATH}")
    print(f"LAYOUT entries: {len(by_name)}")
    print(f"Registry names: {len(all_names)}")
    print(f"Fallback (advanced auto) fields: {len(unmapped_advanced)}")
    print(
        f"PATH_FIELDS: {len(path_kinds)} "
        f"(file={sum(1 for v in path_kinds.values() if v == 'file')}, "
        f"folder={sum(1 for v in path_kinds.values() if v == 'folder')})"
    )
    print("Per-section counts (registry names):")
    for s in SECTION_ORDER:
        print(f"  {s}: {sections.get(s, 0)}")
    # Spot checks
    for check in (
        "learning_rate",
        "output_dir",
        "network_dim",
        "sample_prompts",
        "mixed_precision",
        "huggingface_repo_id",
        "optimizer_type",
    ):
        e = by_name.get(check)
        print(f"  spot {check}: {e}")


if __name__ == "__main__":
    main()
