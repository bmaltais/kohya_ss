"""Shared 'Accelerate launch' FieldSpecs -- the launcher-level flags consumed
by ``accelerate launch`` itself, not by any sd-scripts trainer.

These never appear in a trainer's ``setup_parser()``, so the argparse-driven
generators (``scripts/gen_*_fields.py``) can never discover them -- that's
why they were absent from the six ``*_fields_generated.py`` files even
though the "Accelerate launch" section/title has always existed in
``layout_map.py``. Ported by hand from ``kohya_gui/class_accelerate_launch.py``,
mirroring its defaults/choices/info text.

All fields are ``gui_only=True``: they must round-trip through the saved
config TOML (so a preset remembers "multi_gpu=True", etc.) but must never
leak into the *run* TOML passed via ``--config_file`` -- ``accelerate
launch`` rejects unknown CLI-only options there. ``tab_builder.py``'s
``do_train`` reads them out of the submitted values and appends them to the
command line ahead of the script path, same as the old
``AccelerateLaunch.run_cmd``.
"""

from .fields import FieldSpec, Widget


def _to_int(v):
    return int(v) if v not in (None, "") else v


# FieldSpec.name -> True for the launcher-level flags read directly by
# tab_builder.py's do_train() to build the `accelerate launch` command line.
ACCELERATE_LAUNCH_FIELD_NAMES = frozenset(
    {
        "num_processes",
        "num_machines",
        "multi_gpu",
        "gpu_ids",
        "main_process_port",
        "num_cpu_threads_per_process",
        "dynamo_mode",
        "dynamo_use_fullgraph",
        "dynamo_use_dynamic",
        "extra_accelerate_launch_args",
    }
)


def accelerate_launch_fields(training_type: str) -> list:
    """FieldSpecs for the launcher-level accelerate-launch controls missing
    from the argparse-generated field lists (`mixed_precision` and
    `dynamo_backend` are real trainer args and already come from the
    generator, so they are intentionally excluded here).
    """
    tt = frozenset({training_type})
    return [
        FieldSpec(
            name="num_processes",
            widget=Widget.NUMBER,
            default=1,
            label="Number of processes",
            info="The total number of processes to be launched in parallel.",
            gui_only=True,
            group="accelerate_launch",
            training_types=tt,
            to_toml=_to_int,
            from_toml=_to_int,
        ),
        FieldSpec(
            name="num_machines",
            widget=Widget.NUMBER,
            default=1,
            label="Number of machines",
            info="The total number of machines used in this training.",
            gui_only=True,
            group="accelerate_launch",
            training_types=tt,
            to_toml=_to_int,
            from_toml=_to_int,
        ),
        FieldSpec(
            name="num_cpu_threads_per_process",
            widget=Widget.NUMBER,
            default=2,
            label="Number of CPU threads per core",
            info="The number of CPU threads per process.",
            gui_only=True,
            group="accelerate_launch",
            training_types=tt,
            to_toml=_to_int,
            from_toml=_to_int,
        ),
        FieldSpec(
            name="dynamo_mode",
            widget=Widget.DROPDOWN,
            default="default",
            choices=["default", "reduce-overhead", "max-autotune"],
            label="Dynamo mode",
            info="Choose a mode to optimize your training with dynamo.",
            gui_only=True,
            group="accelerate_launch",
            training_types=tt,
        ),
        FieldSpec(
            name="dynamo_use_fullgraph",
            widget=Widget.CHECKBOX,
            default=False,
            label="Dynamo use fullgraph",
            info=(
                "Whether to use full graph mode for dynamo or it is ok to "
                "break model into several subgraphs"
            ),
            gui_only=True,
            group="accelerate_launch",
            training_types=tt,
        ),
        FieldSpec(
            name="dynamo_use_dynamic",
            widget=Widget.CHECKBOX,
            default=False,
            label="Dynamo use dynamic",
            info="Whether to enable dynamic shape tracing.",
            gui_only=True,
            group="accelerate_launch",
            training_types=tt,
        ),
        FieldSpec(
            name="multi_gpu",
            widget=Widget.CHECKBOX,
            default=False,
            label="Multi GPU",
            info="Whether or not this should launch a distributed GPU training.",
            gui_only=True,
            group="accelerate_launch",
            training_types=tt,
        ),
        FieldSpec(
            name="gpu_ids",
            widget=Widget.TEXTBOX,
            default="",
            label="GPU IDs",
            info=(
                "What GPUs (by id) should be used for training on this "
                "machine as a comma-separated list. Example: 0,1"
            ),
            gui_only=True,
            group="accelerate_launch",
            training_types=tt,
        ),
        FieldSpec(
            name="main_process_port",
            widget=Widget.NUMBER,
            default=0,
            label="Main process port",
            info="The port to use to communicate with the machine of rank 0.",
            gui_only=True,
            group="accelerate_launch",
            training_types=tt,
            to_toml=_to_int,
            from_toml=_to_int,
        ),
        FieldSpec(
            name="extra_accelerate_launch_args",
            widget=Widget.TEXTBOX,
            default="",
            label="Extra accelerate launch arguments",
            info="List of extra parameters to pass to accelerate launch. Example: --same_network --machine_rank 4",
            gui_only=True,
            group="accelerate_launch",
            training_types=tt,
        ),
    ]
