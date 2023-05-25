import argparse
import logging
from servicefoundry import Build, Service, DockerFileBuild, Resources, GPU, GPUType, VolumeMount, LocalSource

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--workspace_fqn", required=True, type=str)
args = parser.parse_args()

image = Build(
    build_source=LocalSource(),
    build_spec=DockerFileBuild()
)

service = Service(
    name="kohya-ss-ui",
    image=image,
    ports=[{"port": 7860, "host": "tfy.dashtoon.ai", "path": "/dev/kohyassui/"}],
    resources=Resources(
        cpu_request=3,
        cpu_limit=3.5,
        memory_limit=14000,
        memory_request=10000,
        ephemeral_storage_request=25000,
        ephemeral_storage_limit=50000,
        gpu=GPU(type=GPUType.A10G),
    ),
    mounts=[
        VolumeMount(
            mount_path="/models",
            volume_fqn="tfy-volume://k8s-aws-mum:dev:dev-models"
        )
    ]
)
service.deploy(workspace_fqn=args.workspace_fqn)
