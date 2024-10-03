import gradio as gr
import os
import shlex

from .class_gui_config import KohyaSSGUIConfig
from .custom_logging import setup_logging

# Set up logging
log = setup_logging()


class AccelerateLaunch:
    def __init__(
        self,
        config: KohyaSSGUIConfig = {},
    ) -> None:
        self.config = config

        with gr.Accordion("Resource Selection", open=True):
            with gr.Row():
                self.mixed_precision = gr.Dropdown(
                    label="Mixed precision",
                    choices=["no", "fp16", "bf16", "fp8"],
                    value=self.config.get("accelerate_launch.mixed_precision", "fp16"),
                    info="Whether or not to use mixed precision training.",
                )
                self.num_processes = gr.Number(
                    label="Number of processes",
                    value=self.config.get("accelerate_launch.num_processes", 1),
                    # precision=0,
                    step=1,
                    minimum=1,
                    info="The total number of processes to be launched in parallel.",
                )
                self.num_machines = gr.Number(
                    label="Number of machines",
                    value=self.config.get("accelerate_launch.num_machines", 1),
                    # precision=0,
                    step=1,
                    minimum=1,
                    info="The total number of machines used in this training.",
                )
                self.num_cpu_threads_per_process = gr.Slider(
                    minimum=1,
                    maximum=os.cpu_count(),
                    step=1,
                    label="Number of CPU threads per core",
                    value=self.config.get(
                        "accelerate_launch.num_cpu_threads_per_process", 2
                    ),
                    info="The number of CPU threads per process.",
                )
            with gr.Row():
                self.dynamo_backend = gr.Dropdown(
                    label="Dynamo backend",
                    choices=[
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
                    value=self.config.get("accelerate_launch.dynamo_backend", "no"),
                    info="The backend to use for the dynamo JIT compiler.",
                )
                self.dynamo_mode = gr.Dropdown(
                    label="Dynamo mode",
                    choices=[
                        "default",
                        "reduce-overhead",
                        "max-autotune",
                    ],
                    value=self.config.get("accelerate_launch.dynamo_mode", "default"),
                    info="Choose a mode to optimize your training with dynamo.",
                )
                self.dynamo_use_fullgraph = gr.Checkbox(
                    label="Dynamo use fullgraph",
                    value=self.config.get(
                        "accelerate_launch.dynamo_use_fullgraph", False
                    ),
                    info="Whether to use full graph mode for dynamo or it is ok to break model into several subgraphs",
                )
                self.dynamo_use_dynamic = gr.Checkbox(
                    label="Dynamo use dynamic",
                    value=self.config.get(
                        "accelerate_launch.dynamo_use_dynamic", False
                    ),
                    info="Whether to enable dynamic shape tracing.",
                )

        with gr.Accordion("Hardware Selection", open=True):
            with gr.Row():
                self.multi_gpu = gr.Checkbox(
                    label="Multi GPU",
                    value=self.config.get("accelerate_launch.multi_gpu", False),
                    info="Whether or not this should launch a distributed GPU training.",
                )
        with gr.Accordion("Distributed GPUs", open=True):
            with gr.Row():
                self.gpu_ids = gr.Textbox(
                    label="GPU IDs",
                    value=self.config.get("accelerate_launch.gpu_ids", ""),
                    placeholder="example: 0,1",
                    info=" What GPUs (by id) should be used for training on this machine as a comma-separated list",
                )

                def validate_gpu_ids(value):
                    if value == "":
                        return
                    if not (
                        value.isdigit() and int(value) >= 0 and int(value) <= 128
                    ):
                        log.error("GPU IDs must be an integer between 0 and 128")
                        return
                    else:
                        for id in value.split(","):
                            if not id.isdigit() or int(id) < 0 or int(id) > 128:
                                log.error(
                                    "GPU IDs must be an integer between 0 and 128"
                                )

                self.gpu_ids.blur(fn=validate_gpu_ids, inputs=self.gpu_ids)

                self.main_process_port = gr.Number(
                    label="Main process port",
                    value=self.config.get("accelerate_launch.main_process_port", 0),
                    # precision=1,
                    step=1,
                    minimum=0,
                    maximum=65535,
                    info="The port to use to communicate with the machine of rank 0.",
                )
        with gr.Row():
            self.extra_accelerate_launch_args = gr.Textbox(
                label="Extra accelerate launch arguments",
                value=self.config.get(
                    "accelerate_launch.extra_accelerate_launch_args", ""
                ),
                placeholder="example: --same_network --machine_rank 4",
                info="List of extra parameters to pass to accelerate launch",
            )

    def run_cmd(run_cmd: list, **kwargs):
        if "dynamo_backend" in kwargs and kwargs.get("dynamo_backend"):
            run_cmd.append("--dynamo_backend")
            run_cmd.append(kwargs["dynamo_backend"])

        if "dynamo_mode" in kwargs and kwargs.get("dynamo_mode"):
            run_cmd.append("--dynamo_mode")
            run_cmd.append(kwargs["dynamo_mode"])

        if "dynamo_use_fullgraph" in kwargs and kwargs.get("dynamo_use_fullgraph"):
            run_cmd.append("--dynamo_use_fullgraph")

        if "dynamo_use_dynamic" in kwargs and kwargs.get("dynamo_use_dynamic"):
            run_cmd.append("--dynamo_use_dynamic")

        if (
            "extra_accelerate_launch_args" in kwargs
            and kwargs["extra_accelerate_launch_args"] != ""
        ):
            extra_accelerate_launch_args = kwargs[
                "extra_accelerate_launch_args"
            ].replace('"', "")
            for arg in extra_accelerate_launch_args.split():
                run_cmd.append(shlex.quote(arg))

        if "gpu_ids" in kwargs and kwargs.get("gpu_ids") != "":
            run_cmd.append("--gpu_ids")
            run_cmd.append(shlex.quote(kwargs["gpu_ids"]))

        if "main_process_port" in kwargs and kwargs.get("main_process_port", 0) > 0:
            run_cmd.append("--main_process_port")
            run_cmd.append(str(int(kwargs["main_process_port"])))

        if "mixed_precision" in kwargs and kwargs.get("mixed_precision"):
            run_cmd.append("--mixed_precision")
            run_cmd.append(shlex.quote(kwargs["mixed_precision"]))

        if "multi_gpu" in kwargs and kwargs.get("multi_gpu"):
            run_cmd.append("--multi_gpu")

        if "num_processes" in kwargs and int(kwargs.get("num_processes", 0)) > 0:
            run_cmd.append("--num_processes")
            run_cmd.append(str(int(kwargs["num_processes"])))

        if "num_machines" in kwargs and int(kwargs.get("num_machines", 0)) > 0:
            run_cmd.append("--num_machines")
            run_cmd.append(str(int(kwargs["num_machines"])))

        if (
            "num_cpu_threads_per_process" in kwargs
            and int(kwargs.get("num_cpu_threads_per_process", 0)) > 0
        ):
            run_cmd.append("--num_cpu_threads_per_process")
            run_cmd.append(str(int(kwargs["num_cpu_threads_per_process"])))

        return run_cmd
