"""Shared helpers for scripts/gen_*_fields.py generators.

Handles: argparse action → widget mapping (with None-choice normalization),
layout_map group/label/info, and curated free-text→dropdown choices.
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Hiragana, Katakana, CJK Unified Ideographs (+ extensions / compatibility)
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _normalize_choice(c):
    """Replace literal None choice with empty string for Gradio safety."""
    return "" if c is None else c


def english_only_help(text: str) -> str:
    """Keep the English half of sd-scripts bilingual help strings.

    Upstream help is often ``\"english description / 日本語…\"``. Prefer the
    first segment that contains no CJK; fall back to the first segment.

    Pure-English strings (including those with ``" / "`` in prose, e.g.
    dataset_config's multi-resolution help) pass through unchanged — only
    split when CJK is present.
    """
    text = " ".join(str(text).split())
    if not text:
        return text
    # No CJK at all → never split; " / " appears in legitimate English help.
    if not _CJK_RE.search(text):
        return text
    # Bilingual separator used by sd-scripts
    for sep in (" / ", "／"):
        if sep in text:
            parts = [p.strip() for p in text.split(sep) if p.strip()]
            non_cjk = [p for p in parts if not _CJK_RE.search(p)]
            if non_cjk:
                return non_cjk[0]
            return parts[0]
    # No separator: pure-CJK help is unusable as a tooltip
    if not re.search(r"[A-Za-z]{3,}", text):
        return ""
    # Mixed without separator — strip CJK runs as a last resort
    stripped = _CJK_RE.sub("", text)
    stripped = re.sub(r"\s{2,}", " ", stripped).strip(" /|")
    return stripped or text


def action_to_widget_and_default(action):
    """Map an argparse.Action to (Widget enum name, default repr, to_toml repr,
    choices repr or None).

    Normalizes None entries in choice lists to "" and aligns default None → "".
    """
    if action.nargs == 0 and isinstance(action.default, bool):
        return "CHECKBOX", repr(bool(action.default)), "None", None
    if action.choices:
        coercer = "None"
        if action.type in (int,):
            coercer = "_to_int"
        elif action.type in (float,):
            coercer = "_to_float"
        choices_list = [_normalize_choice(c) for c in action.choices]
        default = action.default
        if default is None:
            default = ""
        else:
            default = _normalize_choice(default)
        if default not in choices_list:
            choices_list = [default] + choices_list
        # de-dupe preserving order
        seen = set()
        deduped = []
        for c in choices_list:
            if c not in seen:
                seen.add(c)
                deduped.append(c)
        return "DROPDOWN", repr(default), coercer, repr(deduped)
    if action.nargs not in (None, 1) or isinstance(action.default, (list, tuple)):
        return (
            "TEXTBOX",
            repr(action.default if action.default is not None else ""),
            "None",
            None,
        )
    if action.type in (int,):
        return (
            "NUMBER",
            repr(action.default if action.default is not None else 0),
            "_to_int",
            None,
        )
    if action.type in (float,):
        return (
            "NUMBER",
            repr(action.default if action.default is not None else 0.0),
            "_to_float",
            None,
        )
    return (
        "TEXTBOX",
        repr(action.default if action.default is not None else ""),
        "None",
        None,
    )


def _safe_help(action) -> str | None:
    if action is None:
        return None
    help_text = getattr(action, "help", None)
    if not help_text or help_text == "===SUPPRESS===":
        return None
    cleaned = english_only_help(help_text)
    return cleaned or None


def layout_presentation(name: str, action=None):
    """Return (group, label_repr, info_repr, widget_override, choices_override).

    widget_override / choices_override are None unless name is in CURATED_CHOICES.
    """
    from kohya_gui_v2.layout_map import CURATED_CHOICES, layout_for

    lay = layout_for(name)
    group = lay.section
    label = lay.label
    info = lay.info or _safe_help(action)
    if info:
        info = english_only_help(info) or None
    if label is None:
        label = name.replace("_", " ").capitalize()

    widget_override = None
    choices_override = None
    if name in CURATED_CHOICES:
        widget_override = "DROPDOWN"
        choices = list(CURATED_CHOICES[name])
        # Gradio warns if value not in choices; empty string is a valid
        # "unset" sentinel for curated fields with allow_custom_value.
        if "" not in choices and name in (
            "optimizer_type",
            "lr_scheduler_type",
            "caption_extension",
        ):
            choices = [""] + choices
        choices_override = repr(choices)

    return (
        group,
        repr(label),
        repr(info) if info else "None",
        widget_override,
        choices_override,
    )


def emit_field_spec_line(
    name: str,
    widget: str,
    default_repr: str,
    coercer: str,
    choices_repr: str | None,
    archs_repr: str,
    training_type: str,
    action=None,
) -> str:
    """One FieldSpec(...) source line for a generated fields module."""
    group, label_repr, info_repr, widget_override, choices_override = (
        layout_presentation(name, action=action)
    )
    if widget_override:
        widget = widget_override
    if choices_override is not None:
        choices_repr = choices_override
        # Ensure default is in curated list or leave as-is (allow_custom at runtime)
        # If default was None/empty for optimizer etc., keep default_repr from argparse

    coerce_kwargs = ""
    if coercer != "None":
        coerce_kwargs = f", to_toml={coercer}, from_toml={coercer}"
    choices_kwargs = f", choices={choices_repr}" if choices_repr else ""
    info_kwargs = f", info={info_repr}" if info_repr != "None" else ""
    return (
        f"    FieldSpec(name={name!r}, widget=Widget.{widget}, default={default_repr}, "
        f"label={label_repr}{info_kwargs}, "
        f"archs={archs_repr}, training_types=frozenset({{{training_type!r}}}), "
        f"group={group!r}{coerce_kwargs}{choices_kwargs}),"
    )
