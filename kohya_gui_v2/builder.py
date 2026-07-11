"""Gradio widget builder driven entirely by a FieldRegistry.

`build_components` is the single place that turns FieldSpecs into Gradio
components. The returned ordered dict IS the order used for save/open/train
event wiring elsewhere -- there is no second hand-maintained list (the
musubi-tuner-gui FIELD_NAMES/settings_list drift this design eliminates; see
wargame plan Move 3).
"""

from collections import OrderedDict

import gradio as gr

from .fields import FieldRegistry, Widget

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
    registry: FieldRegistry, config=None
) -> "OrderedDict[str, object]":
    """Instantiate one Gradio component per non-gui_only, visible-by-default
    FieldSpec, in registry order. `config` is a GUIConfig-like object
    exposing `.get(name, default)` for persisted GUI defaults
    (kohya_gui.class_gui_config.KohyaSSGUIConfig / musubi's GUIConfig share
    this interface).
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
        if spec.widget is Widget.SLIDER:
            kwargs.setdefault("minimum", 0)
            kwargs.setdefault("maximum", 1)

        components[spec.name] = factory(**kwargs)

    return components


def build_group_visibility_map(registry: FieldRegistry) -> dict:
    """Return {group_name: FieldSpec list} so callers can wrap each group in
    its own gr.Column/gr.Group and generate visibility-toggle wiring from
    the registry instead of a hand-written positional tuple (contrast with
    musubi-tuner-gui's apply_architecture(), which returns a positional
    20-tuple of gr.Column(visible=...) -- this map lets callers build that
    generically for any number of groups).
    """
    groups: dict = {}
    for spec in registry:
        groups.setdefault(spec.group, []).append(spec)
    return groups


def visible_groups_for(
    registry: FieldRegistry, arch_key: str, training_type: str
) -> set:
    """Which group names should be visible for a given architecture +
    training type selection, derived purely from FieldSpec.archs /
    training_types -- a group is visible if at least one of its fields
    supports the selection.
    """
    visible = set()
    for spec in registry:
        if spec.supports_arch(arch_key) and spec.supports_training_type(training_type):
            visible.add(spec.group)
    return visible
