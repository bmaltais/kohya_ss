"""Gradio widget builder driven entirely by a FieldRegistry.

`build_components` is the single place that turns FieldSpecs into Gradio
components. The returned ordered dict IS the order used for save/open/train
event wiring elsewhere -- there is no second hand-maintained list (the
musubi-tuner-gui FIELD_NAMES/settings_list drift this design eliminates; see
wargame plan Move 3).
"""

from collections import OrderedDict
from typing import Optional

import gradio as gr

from .fields import FieldRegistry, Widget

try:
    from .layout_map import CURATED_ALLOW_CUSTOM
except ImportError:  # pragma: no cover - layout_map always present in package
    CURATED_ALLOW_CUSTOM = frozenset()

_WIDGET_FACTORIES = {
    Widget.TEXTBOX: gr.Textbox,
    Widget.NUMBER: gr.Number,
    Widget.CHECKBOX: gr.Checkbox,
    Widget.DROPDOWN: gr.Dropdown,
    Widget.SLIDER: gr.Slider,
    Widget.FILE: gr.Textbox,  # path textbox + browse button is wired by callers, same as existing GUI
    Widget.FOLDER: gr.Textbox,
}


def build_components(
    registry: FieldRegistry,
    config=None,
    *,
    elem_id_prefix: Optional[str] = None,
    elem_classes: Optional[list] = None,
) -> "OrderedDict[str, object]":
    """Instantiate one Gradio component per non-gui_only, visible-by-default
    FieldSpec, in registry order. `config` is a GUIConfig-like object
    exposing `.get(name, default)` for persisted GUI defaults
    (kohya_gui.class_gui_config.KohyaSSGUIConfig / musubi's GUIConfig share
    this interface).

    Optional `elem_id_prefix` (e.g. ``\"v2_lora\"``) yields
    ``elem_id=\"{prefix}_{name}\"``; `elem_classes` are applied to every widget.
    """
    components: "OrderedDict[str, object]" = OrderedDict()
    for spec in registry:
        factory = _WIDGET_FACTORIES[spec.widget]
        default = spec.default
        if config is not None:
            default = config.get(spec.name, spec.default)

        kwargs = {"value": default, "label": spec.label or spec.name}
        if spec.info:
            kwargs["info"] = spec.info
        if spec.widget in (Widget.DROPDOWN,) and spec.choices is not None:
            kwargs["choices"] = spec.choices
            if spec.name in CURATED_ALLOW_CUSTOM:
                kwargs["allow_custom_value"] = True
        if spec.widget is Widget.SLIDER:
            kwargs.setdefault("minimum", 0)
            kwargs.setdefault("maximum", 1)
        if elem_id_prefix:
            kwargs["elem_id"] = f"{elem_id_prefix}_{spec.name}"
        if elem_classes:
            kwargs["elem_classes"] = list(elem_classes)

        components[spec.name] = factory(**kwargs)

    return components
