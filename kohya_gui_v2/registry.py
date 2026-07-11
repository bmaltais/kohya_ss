"""Architecture + backend registry.

`BackendSpec` is the seam that keeps this framework trainer-agnostic (see
wargame/2026-07-11-musubi-style-gui-v2.md, Move 3 decided constraint): it
owns everything that differs between trainer families -- launcher prefix,
script root, config-file flag, run-config filter policy, env setup, and
optional pre-train command builders. Exactly one backend ("sd_scripts") is
implemented in this project. All command construction in builder.py must go
through a BackendSpec; nothing here or in builder.py/config_io.py may
hardcode sd-scripts paths. Adding a second backend (e.g. musubi-tuner, same
`accelerate launch <script> --config_file <flat.toml>` contract) later
should require only a new BackendSpec + ArchitectureSpec entries + FieldSpecs
-- zero changes to fields.py, builder.py, or config_io.py.

Explicitly rejected here: launching backends via in-process module import
(`import train_network; train_network.train(args)`) instead of
`subprocess`-based `accelerate launch`. Kept as subprocess because (a)
accelerate's multi-GPU/multi-node model spawns a fresh process per worker
with per-worker env (RANK/WORLD_SIZE/LOCAL_RANK); (b) Accelerate's
AcceleratorState is a process-global singleton, so even sequential
single-GPU runs in one long-lived GUI process would inherit stale state;
(c) process isolation means a CUDA OOM/crash doesn't kill the GUI and Stop
is a clean process kill; (d) sd-scripts entry points are __main__-style
CLIs, not libraries. See the plan file for the full validated rationale.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(frozen=True)
class BackendSpec:
    key: str
    launcher: list  # e.g. ["uv", "run", "accelerate", "launch"]
    script_root: str  # directory containing the trainer's entry scripts, relative to project root
    config_file_flag: str = "--config_file"
    keep_falsy_default: bool = (
        False  # default keep_if_falsy policy when a FieldSpec doesn't override it
    )
    pre_train_commands: Optional[Callable] = (
        None  # optional: (values, arch_spec) -> list[list[str]]
    )


@dataclass(frozen=True)
class ArchitectureSpec:
    key: str
    label: str
    backend: str  # BackendSpec.key
    training_type: str
    train_script: str  # relative to the backend's script_root
    notes: str = ""


SD_SCRIPTS_BACKEND = BackendSpec(
    key="sd_scripts",
    launcher=["uv", "run", "accelerate", "launch"],
    script_root="sd-scripts",
)

BACKENDS = {
    SD_SCRIPTS_BACKEND.key: SD_SCRIPTS_BACKEND,
}


class ArchitectureRegistry:
    """Ordered collection of ArchitectureSpecs, keyed and grouped by training type."""

    def __init__(self, specs: list):
        seen = set()
        for spec in specs:
            if spec.key in seen:
                raise ValueError(f"Duplicate ArchitectureSpec key: {spec.key!r}")
            if spec.backend not in BACKENDS:
                raise ValueError(
                    f"ArchitectureSpec {spec.key!r} references unknown backend {spec.backend!r}"
                )
            seen.add(spec.key)
        self._specs = list(specs)
        self._by_key = {s.key: s for s in specs}

    def __iter__(self):
        return iter(self._specs)

    def __getitem__(self, key: str) -> ArchitectureSpec:
        return self._by_key[key]

    def get(self, key: str) -> Optional[ArchitectureSpec]:
        return self._by_key.get(key)

    def for_training_type(self, training_type: str):
        return [s for s in self._specs if s.training_type == training_type]

    def backend_for(self, key: str) -> BackendSpec:
        return BACKENDS[self._by_key[key].backend]


# Populated incrementally as each training type's field module is authored
# (Move 4+). Empty in Phase A -- the framework must construct and validate
# with zero entries.
ARCHITECTURES = ArchitectureRegistry([])
