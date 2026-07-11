"""Declarative field registry.

A single `FieldSpec` per trainer-facing value replaces musubi-tuner-gui's
hand-synced `FIELD_NAMES` list + `settings_list` pair (see
wargame/2026-07-11-musubi-style-gui-v2.md, Move 3). One ordered mapping of
`name -> FieldSpec` is the sole source of truth for: which Gradio component
gets built, the order component values are wired into event handlers, and
how a value round-trips through TOML.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class Widget(Enum):
    TEXTBOX = "textbox"
    NUMBER = "number"
    CHECKBOX = "checkbox"
    DROPDOWN = "dropdown"
    SLIDER = "slider"
    FILE = "file"
    FOLDER = "folder"


@dataclass(frozen=True)
class FieldSpec:
    """One trainer-facing (or GUI-only bookkeeping) value.

    `name` must equal the sd-scripts argparse `dest` (the TOML key) unless
    `gui_only=True`, in which case `name` is a v2-internal identifier that
    never reaches `--config_file` (e.g. "architecture", "training_type").

    `archs` / `training_types`: `None` means universal (all architectures /
    all training types support this field). A non-None set restricts when
    the field's widget group is visible and when it is eligible for
    inclusion in a saved/run config for a given selection.

    `keep_if_falsy`: by default, keys whose value is "", False, or None are
    dropped from the run config (mirrors the existing GUI's documented
    behavior). Set True for fields where a legitimate falsy value is
    meaningful and must survive (e.g. guidance_scale=0.0 for Chroma) --
    this replaces per-key hardcoded exceptions with a spec-level flag.
    """

    name: str
    widget: Widget
    default: Any = None
    label: Optional[str] = None
    info: Optional[str] = None
    choices: Optional[list] = None
    group: str = "general"
    archs: Optional[frozenset] = None
    training_types: Optional[frozenset] = None
    gui_only: bool = False
    keep_if_falsy: bool = False
    to_toml: Optional[Callable[[Any], Any]] = None
    from_toml: Optional[Callable[[Any], Any]] = None

    def supports_arch(self, arch_key: Optional[str]) -> bool:
        return self.archs is None or arch_key is None or arch_key in self.archs

    def supports_training_type(self, training_type: Optional[str]) -> bool:
        return (
            self.training_types is None
            or training_type is None
            or training_type in self.training_types
        )

    def coerce_to_toml(self, value: Any) -> Any:
        return self.to_toml(value) if self.to_toml else value

    def coerce_from_toml(self, value: Any) -> Any:
        return self.from_toml(value) if self.from_toml else value


class FieldRegistry:
    """Ordered collection of FieldSpecs with uniqueness enforced at construction.

    The insertion order of `specs` becomes the canonical order used for
    Gradio component construction and event-handler wiring -- there is no
    second list to keep in sync.
    """

    def __init__(self, specs: list):
        seen = set()
        for spec in specs:
            if spec.name in seen:
                raise ValueError(f"Duplicate FieldSpec name: {spec.name!r}")
            seen.add(spec.name)
        self._specs = list(specs)
        self._by_name = {s.name: s for s in specs}

    def __iter__(self):
        return iter(self._specs)

    def __len__(self):
        return len(self._specs)

    def __getitem__(self, name: str) -> FieldSpec:
        return self._by_name[name]

    def names(self) -> list:
        return [s.name for s in self._specs]

    def for_selection(
        self, arch_key: Optional[str] = None, training_type: Optional[str] = None
    ):
        """Yield specs visible/eligible for the given architecture + training type."""
        for spec in self._specs:
            if spec.supports_arch(arch_key) and spec.supports_training_type(
                training_type
            ):
                yield spec
