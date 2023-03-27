import argparse
import os
from pathlib import Path
import threading
from typing import *

from huggingface_hub import HfApi


def fire_in_thread(f):
    def wrapped(*args, **kwargs):
        threading.Thread(target=f, args=args, kwargs=kwargs).start()
    return wrapped


def huggingface_exists_repo(
    repo_id: str, repo_type: str, revision: str = "main", hf_token: str = None
):
    api = HfApi()
    try:
        api.repo_info(
            repo_id=repo_id, token=hf_token, revision=revision, repo_type=repo_type
        )
        return True
    except:
        return False


@fire_in_thread
def huggingface_upload(
    src: Union[str, Path, bytes, BinaryIO],
    args: argparse.Namespace,
    dest_suffix: str = "",
):
    repo_id = args.huggingface_repo_id
    repo_type = args.huggingface_repo_type
    hf_token = args.huggingface_token
    path_in_repo = args.huggingface_path_in_repo + dest_suffix
    private = args.huggingface_repo_visibility == "private"
    api = HfApi()
    if not huggingface_exists_repo(
        repo_id=repo_id, repo_type=repo_type, hf_token=hf_token
    ):
        api.create_repo(
            token=hf_token, repo_id=repo_id, repo_type=repo_type, private=private
        )

    is_folder = (type(src) == str and os.path.isdir(src)) or (
        isinstance(src, Path) and src.is_dir()
    )
    if is_folder:
        api.upload_folder(
            repo_id=repo_id,
            repo_type=repo_type,
            folder_path=src,
            path_in_repo=path_in_repo,
        )
    else:
        api.upload_file(
            repo_id=repo_id,
            repo_type=repo_type,
            path_or_fileobj=src,
            path_in_repo=path_in_repo,
        )
