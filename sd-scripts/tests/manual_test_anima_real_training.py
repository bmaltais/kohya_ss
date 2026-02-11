"""
Test script that actually runs anima_train.py and anima_train_network.py
for a few steps to verify --cache_text_encoder_outputs works.

Usage:
    python test_anima_real_training.py \
        --image_dir /path/to/images_with_txt \
        --dit_path /path/to/dit.safetensors \
        --qwen3_path /path/to/qwen3 \
        --vae_path /path/to/vae.safetensors \
        [--t5_tokenizer_path /path/to/t5] \
        [--resolution 512]

This will run 4 tests:
    1. anima_train.py           (full finetune, no cache)
    2. anima_train.py           (full finetune, --cache_text_encoder_outputs)
    3. anima_train_network.py   (LoRA, no cache)
    4. anima_train_network.py   (LoRA, --cache_text_encoder_outputs)

Each test runs only 2 training steps then stops.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import shutil


def create_dataset_toml(image_dir: str, resolution: int, toml_path: str):
    """Create a minimal dataset toml config."""
    content = f"""[general]
resolution = {resolution}
enable_bucket = true
bucket_reso_steps = 8
min_bucket_reso = 256
max_bucket_reso = 1024

[[datasets]]
batch_size = 1

  [[datasets.subsets]]
  image_dir = "{image_dir}"
  num_repeats = 1
  caption_extension = ".txt"
"""
    with open(toml_path, "w", encoding="utf-8") as f:
        f.write(content)
    return toml_path


def run_test(test_name: str, cmd: list, timeout: int = 300) -> dict:
    """Run a training command and capture result."""
    print(f"\n{'=' * 70}")
    print(f"TEST: {test_name}")
    print(f"{'=' * 70}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

        stdout = result.stdout
        stderr = result.stderr
        returncode = result.returncode

        # Print last N lines of output
        all_output = stdout + "\n" + stderr
        lines = all_output.strip().split("\n")
        print(f"--- Last 30 lines of output ---")
        for line in lines[-30:]:
            print(f"  {line}")
        print(f"--- End output ---\n")

        if returncode == 0:
            print(f"RESULT: PASS (exit code 0)")
            return {"status": "PASS", "detail": "completed successfully"}
        else:
            # Check if it's a known error
            if "TypeError: 'NoneType' object is not iterable" in all_output:
                print(f"RESULT: FAIL - input_ids_list is None (the cache_text_encoder_outputs bug)")
                return {"status": "FAIL", "detail": "input_ids_list is None - cache TE outputs bug"}
            elif "steps:   0%" in all_output and "Error" in all_output:
                # Find the actual error
                error_lines = [l for l in lines if "Error" in l or "Traceback" in l or "raise" in l.lower()]
                detail = error_lines[-1] if error_lines else f"exit code {returncode}"
                print(f"RESULT: FAIL - {detail}")
                return {"status": "FAIL", "detail": detail}
            else:
                print(f"RESULT: FAIL (exit code {returncode})")
                return {"status": "FAIL", "detail": f"exit code {returncode}"}

    except subprocess.TimeoutExpired:
        print(f"RESULT: TIMEOUT (>{timeout}s)")
        return {"status": "TIMEOUT", "detail": f"exceeded {timeout}s"}
    except Exception as e:
        print(f"RESULT: ERROR - {e}")
        return {"status": "ERROR", "detail": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Test Anima real training with cache flags")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory with image+txt pairs")
    parser.add_argument("--dit_path", type=str, required=True,
                        help="Path to Anima DiT safetensors")
    parser.add_argument("--qwen3_path", type=str, required=True,
                        help="Path to Qwen3 model")
    parser.add_argument("--vae_path", type=str, required=True,
                        help="Path to WanVAE safetensors")
    parser.add_argument("--t5_tokenizer_path", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout per test in seconds (default: 300)")
    parser.add_argument("--only", type=str, default=None,
                        choices=["finetune", "lora"],
                        help="Only run finetune or lora tests")
    args = parser.parse_args()

    # Validate paths
    for name, path in [("image_dir", args.image_dir), ("dit_path", args.dit_path),
                        ("qwen3_path", args.qwen3_path), ("vae_path", args.vae_path)]:
        if not os.path.exists(path):
            print(f"ERROR: {name} does not exist: {path}")
            sys.exit(1)

    # Create temp dir for outputs
    tmp_dir = tempfile.mkdtemp(prefix="anima_test_")
    print(f"Temp directory: {tmp_dir}")

    # Create dataset toml
    toml_path = os.path.join(tmp_dir, "dataset.toml")
    create_dataset_toml(args.image_dir, args.resolution, toml_path)
    print(f"Dataset config: {toml_path}")

    output_dir = os.path.join(tmp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    python = sys.executable

    # Common args for both scripts
    common_anima_args = [
        "--dit_path", args.dit_path,
        "--qwen3_path", args.qwen3_path,
        "--vae_path", args.vae_path,
        "--pretrained_model_name_or_path", args.dit_path,  # required by base parser
        "--output_dir", output_dir,
        "--output_name", "test",
        "--dataset_config", toml_path,
        "--max_train_steps", "2",
        "--learning_rate", "1e-5",
        "--mixed_precision", "bf16",
        "--save_every_n_steps", "999",  # don't save
        "--max_data_loader_n_workers", "0",  # single process for clarity
        "--logging_dir", os.path.join(tmp_dir, "logs"),
        "--cache_latents",
    ]
    if args.t5_tokenizer_path:
        common_anima_args += ["--t5_tokenizer_path", args.t5_tokenizer_path]

    results = {}

    # TEST 1: anima_train.py - NO cache_text_encoder_outputs
    if args.only is None or args.only == "finetune":
        cmd = [python, "anima_train.py"] + common_anima_args + [
            "--optimizer_type", "AdamW8bit",
        ]
        results["finetune_no_cache"] = run_test(
            "anima_train.py (full finetune, NO text encoder cache)",
            cmd, args.timeout,
        )

        # TEST 2: anima_train.py - WITH cache_text_encoder_outputs
        cmd = [python, "anima_train.py"] + common_anima_args + [
            "--optimizer_type", "AdamW8bit",
            "--cache_text_encoder_outputs",
        ]
        results["finetune_with_cache"] = run_test(
            "anima_train.py (full finetune, WITH --cache_text_encoder_outputs)",
            cmd, args.timeout,
        )

    # TEST 3: anima_train_network.py - NO cache_text_encoder_outputs
    if args.only is None or args.only == "lora":
        lora_args = common_anima_args + [
            "--optimizer_type", "AdamW8bit",
            "--network_module", "networks.lora_anima",
            "--network_dim", "4",
            "--network_alpha", "1",
        ]

        cmd = [python, "anima_train_network.py"] + lora_args
        results["lora_no_cache"] = run_test(
            "anima_train_network.py (LoRA, NO text encoder cache)",
            cmd, args.timeout,
        )

        # TEST 4: anima_train_network.py - WITH cache_text_encoder_outputs
        cmd = [python, "anima_train_network.py"] + lora_args + [
            "--cache_text_encoder_outputs",
        ]
        results["lora_with_cache"] = run_test(
            "anima_train_network.py (LoRA, WITH --cache_text_encoder_outputs)",
            cmd, args.timeout,
        )

    # SUMMARY
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    all_pass = True
    for test_name, result in results.items():
        status = result["status"]
        icon = "OK" if status == "PASS" else "FAIL"
        if status != "PASS":
            all_pass = False
        print(f"  [{icon:4s}] {test_name}: {result['detail']}")

    print(f"\nTemp directory (can delete): {tmp_dir}")

    # Cleanup
    try:
        shutil.rmtree(tmp_dir)
        print("Temp directory cleaned up.")
    except Exception:
        print(f"Note: could not clean up {tmp_dir}")

    if all_pass:
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
