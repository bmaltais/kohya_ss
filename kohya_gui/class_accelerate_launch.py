import gradio as gr
import os
import shlex

from .class_gui_config import KohyaSSGUIConfig


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
                    precision=0,
                    minimum=1,
                    info="The total number of processes to be launched in parallel.",
                )
                self.num_machines = gr.Number(
                    label="Number of machines",
                    value=self.config.get("accelerate_launch.num_machines", 1),
                    precision=0,
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
                self.main_process_port = gr.Number(
                    label="Main process port",
                    value=self.config.get("accelerate_launch.main_process_port", 0),
                    precision=1,
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
        if "extra_accelerate_launch_args" in kwargs:
            extra_accelerate_launch_args = kwargs.get("extra_accelerate_launch_args")
            if extra_accelerate_launch_args != "":
                run_cmd.append(extra_accelerate_launch_args)

        if "gpu_ids" in kwargs:
            gpu_ids = kwargs.get("gpu_ids")
            if not gpu_ids == "":
                run_cmd.append(f"--gpu_ids={shlex.quote(gpu_ids)}")

        if "main_process_port" in kwargs:
            main_process_port = kwargs.get("main_process_port")
            if main_process_port > 0:
                run_cmd.append(f"--main_process_port={int(main_process_port)}")

        if "mixed_precision" in kwargs:
            run_cmd.append(f"--mixed_precision={shlex.quote(kwargs.get('mixed_precision'))}")

        if "multi_gpu" in kwargs:
            if kwargs.get("multi_gpu"):
                run_cmd.append("--multi_gpu")

        if "num_processes" in kwargs:
            num_processes = kwargs.get("num_processes")
            if int(num_processes) > 0:
                run_cmd.append(f"--num_processes={int(num_processes)}")

        if "num_machines" in kwargs:
            num_machines = kwargs.get("num_machines")
            if int(num_machines) > 0:
                run_cmd.append(f"--num_machines={int(num_machines)}")

        if "num_cpu_threads_per_process" in kwargs:
            num_cpu_threads_per_process = kwargs.get("num_cpu_threads_per_process")
            if int(num_cpu_threads_per_process) > 0:
                run_cmd.append(
                    f"--num_cpu_threads_per_process={int(num_cpu_threads_per_process)}"
                )

        return run_cmd
