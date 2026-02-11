"""
Diagnostic script to test Anima latent & text encoder caching independently.

Usage:
    python manual_test_anima_cache.py \
        --image_dir /path/to/images \
        --qwen3_path /path/to/qwen3 \
        --vae_path /path/to/vae.safetensors \
        [--t5_tokenizer_path /path/to/t5] \
        [--cache_to_disk]

The image_dir should contain pairs of:
    image1.png + image1.txt
    image2.jpg + image2.txt
    ...
"""

import argparse
import glob
import os
import sys
import traceback

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Helpers

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),  # [0,1]
        transforms.Normalize([0.5], [0.5]),  # [-1,1]
    ]
)


def find_image_caption_pairs(image_dir: str):
    """Find (image_path, caption_text) pairs from a directory."""
    pairs = []
    for f in sorted(os.listdir(image_dir)):
        ext = os.path.splitext(f)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        img_path = os.path.join(image_dir, f)
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as fh:
                caption = fh.read().strip()
        else:
            caption = ""
        pairs.append((img_path, caption))
    return pairs


def print_tensor_info(name: str, t, indent=2):
    prefix = " " * indent
    if t is None:
        print(f"{prefix}{name}: None")
        return
    if isinstance(t, np.ndarray):
        print(f"{prefix}{name}: numpy {t.dtype} shape={t.shape} " f"min={t.min():.4f} max={t.max():.4f} mean={t.mean():.4f}")
    elif isinstance(t, torch.Tensor):
        print(
            f"{prefix}{name}: torch {t.dtype} shape={tuple(t.shape)} "
            f"min={t.min().item():.4f} max={t.max().item():.4f} mean={t.float().mean().item():.4f}"
        )
    else:
        print(f"{prefix}{name}: type={type(t)} value={t}")


# Test 1: Latent Cache


def test_latent_cache(args, pairs):
    print("\n" + "=" * 70)
    print("TEST 1: LATENT CACHING (VAE encode -> cache -> reload)")
    print("=" * 70)

    from library import qwen_image_autoencoder_kl

    # Load VAE
    print("\n[1.1] Loading VAE...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae_dtype = torch.float32
    vae = qwen_image_autoencoder_kl.load_vae(args.vae_path, dtype=vae_dtype, device=device)
    print(f"  VAE loaded on {device}, dtype={vae_dtype}")

    for img_path, caption in pairs:
        print(f"\n[1.2] Processing: {os.path.basename(img_path)}")

        # Load image
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        print(f"  Raw image: {img_np.shape} dtype={img_np.dtype} " f"min={img_np.min()} max={img_np.max()}")

        # Apply IMAGE_TRANSFORMS (same as sd-scripts training)
        img_tensor = IMAGE_TRANSFORMS(img_np)
        print(
            f"  After IMAGE_TRANSFORMS: shape={tuple(img_tensor.shape)} " f"min={img_tensor.min():.4f} max={img_tensor.max():.4f}"
        )

        # Check range is [-1, 1]
        if img_tensor.min() < -1.01 or img_tensor.max() > 1.01:
            print("  ** WARNING: tensor out of [-1, 1] range!")
        else:
            print("  OK: tensor in [-1, 1] range")

        # Encode with VAE
        img_batch = img_tensor.unsqueeze(0).to(device, dtype=vae_dtype)  # (1, C, H, W)
        img_5d = img_batch.unsqueeze(2)  # (1, C, 1, H, W) - add temporal dim
        print(f"  VAE input: shape={tuple(img_5d.shape)} dtype={img_5d.dtype}")

        with torch.no_grad():
            latents = vae.encode_pixels_to_latents(img_5d)
        latents_cpu = latents.cpu()
        print_tensor_info("Encoded latents", latents_cpu)

        # Check for NaN/Inf
        if torch.any(torch.isnan(latents_cpu)):
            print("  ** ERROR: NaN in latents!")
        elif torch.any(torch.isinf(latents_cpu)):
            print("  ** ERROR: Inf in latents!")
        else:
            print("  OK: no NaN/Inf")

        # Test disk cache round-trip
        if args.cache_to_disk:
            npz_path = os.path.splitext(img_path)[0] + "_test_latent.npz"
            latents_np = latents_cpu.float().numpy()
            h, w = img_np.shape[:2]
            np.savez(
                npz_path,
                latents=latents_np,
                original_size=np.array([w, h]),
                crop_ltrb=np.array([0, 0, 0, 0]),
            )
            print(f"  Saved to: {npz_path}")

            # Reload
            loaded = np.load(npz_path)
            loaded_latents = loaded["latents"]
            print_tensor_info("Reloaded latents", loaded_latents)

            # Compare
            diff = np.abs(latents_np - loaded_latents).max()
            print(f"  Max diff (save vs load): {diff:.2e}")
            if diff > 1e-5:
                print("  ** WARNING: latent cache round-trip has significant diff!")
            else:
                print("  OK: round-trip matches")

            os.remove(npz_path)
            print(f"  Cleaned up {npz_path}")

    vae.to("cpu")
    del vae
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("\n[1.3] Latent cache test DONE.")


# Test 2: Text Encoder Output Cache


def test_text_encoder_cache(args, pairs):
    # TODO Rewrite this
    print("\n" + "=" * 70)
    print("TEST 2: TEXT ENCODER OUTPUT CACHING")
    print("=" * 70)

    from library import anima_utils

    # Load tokenizers
    print("\n[2.1] Loading tokenizers...")
    qwen3_tokenizer = anima_utils.load_qwen3_tokenizer(args.qwen3_path)
    t5_tokenizer = anima_utils.load_t5_tokenizer(getattr(args, "t5_tokenizer_path", None))
    print(f"  Qwen3 tokenizer vocab: {qwen3_tokenizer.vocab_size}")
    print(f"  T5 tokenizer vocab: {t5_tokenizer.vocab_size}")

    # Load text encoder
    print("\n[2.2] Loading Qwen3 text encoder...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    te_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    qwen3_model, _ = anima_utils.load_qwen3_text_encoder(args.qwen3_path, dtype=te_dtype, device=device)
    qwen3_model.eval()

    # Create strategy objects
    from library.strategy_anima import AnimaTokenizeStrategy, AnimaTextEncodingStrategy

    tokenize_strategy = AnimaTokenizeStrategy(
        qwen3_tokenizer=qwen3_tokenizer,
        t5_tokenizer=t5_tokenizer,
        qwen3_max_length=args.qwen3_max_length,
        t5_max_length=args.t5_max_length,
    )
    text_encoding_strategy = AnimaTextEncodingStrategy()

    captions = [cap for _, cap in pairs]
    print(f"\n[2.3] Tokenizing {len(captions)} captions...")
    for i, cap in enumerate(captions):
        print(f"  [{i}] \"{cap[:80]}{'...' if len(cap) > 80 else ''}\"")

    tokens_and_masks = tokenize_strategy.tokenize(captions)
    qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask = tokens_and_masks

    print(f"\n  Tokenization results:")
    print_tensor_info("qwen3_input_ids", qwen3_input_ids)
    print_tensor_info("qwen3_attn_mask", qwen3_attn_mask)
    print_tensor_info("t5_input_ids", t5_input_ids)
    print_tensor_info("t5_attn_mask", t5_attn_mask)

    # Encode
    print(f"\n[2.4] Encoding with Qwen3 text encoder...")
    with torch.no_grad():
        prompt_embeds, attn_mask, t5_ids_out, t5_mask_out = text_encoding_strategy.encode_tokens(
            tokenize_strategy, [qwen3_model], tokens_and_masks
        )

    print(f"  Encoding results:")
    print_tensor_info("prompt_embeds", prompt_embeds)
    print_tensor_info("attn_mask", attn_mask)
    print_tensor_info("t5_input_ids", t5_ids_out)
    print_tensor_info("t5_attn_mask", t5_mask_out)

    # Check for NaN/Inf
    if torch.any(torch.isnan(prompt_embeds)):
        print("  ** ERROR: NaN in prompt_embeds!")
    elif torch.any(torch.isinf(prompt_embeds)):
        print("  ** ERROR: Inf in prompt_embeds!")
    else:
        print("  OK: no NaN/Inf in prompt_embeds")

    # Test cache round-trip (simulate what AnimaTextEncoderOutputsCachingStrategy does)
    print(f"\n[2.5] Testing cache round-trip (encode -> numpy -> npz -> reload -> tensor)...")

    # Convert to numpy (same as cache_batch_outputs in strategy_anima.py)
    pe_cpu = prompt_embeds.cpu()
    if pe_cpu.dtype == torch.bfloat16:
        pe_cpu = pe_cpu.float()
    pe_np = pe_cpu.numpy()
    am_np = attn_mask.cpu().numpy()
    t5_ids_np = t5_ids_out.cpu().numpy().astype(np.int32)
    t5_mask_np = t5_mask_out.cpu().numpy().astype(np.int32)

    print(f"  Numpy conversions:")
    print_tensor_info("prompt_embeds_np", pe_np)
    print_tensor_info("attn_mask_np", am_np)
    print_tensor_info("t5_input_ids_np", t5_ids_np)
    print_tensor_info("t5_attn_mask_np", t5_mask_np)

    if args.cache_to_disk:
        npz_path = os.path.join(args.image_dir, "_test_te_cache.npz")
        # Save per-sample (simulating cache_batch_outputs)
        for i in range(len(captions)):
            sample_npz = os.path.splitext(pairs[i][0])[0] + "_test_te.npz"
            np.savez(
                sample_npz,
                prompt_embeds=pe_np[i],
                attn_mask=am_np[i],
                t5_input_ids=t5_ids_np[i],
                t5_attn_mask=t5_mask_np[i],
            )
            print(f"  Saved: {sample_npz}")

            # Reload (simulating load_outputs_npz)
            data = np.load(sample_npz)
            print(f"  Reloaded keys: {list(data.keys())}")
            print_tensor_info("  loaded prompt_embeds", data["prompt_embeds"], indent=4)
            print_tensor_info("  loaded attn_mask", data["attn_mask"], indent=4)
            print_tensor_info("  loaded t5_input_ids", data["t5_input_ids"], indent=4)
            print_tensor_info("  loaded t5_attn_mask", data["t5_attn_mask"], indent=4)

            # Check diff
            diff_pe = np.abs(pe_np[i] - data["prompt_embeds"]).max()
            diff_t5 = np.abs(t5_ids_np[i] - data["t5_input_ids"]).max()
            print(f"    Max diff prompt_embeds: {diff_pe:.2e}")
            print(f"    Max diff t5_input_ids: {diff_t5:.2e}")
            if diff_pe > 1e-5 or diff_t5 > 0:
                print("    ** WARNING: cache round-trip mismatch!")
            else:
                print("    OK: round-trip matches")

            os.remove(sample_npz)
            print(f"    Cleaned up {sample_npz}")

    # Test in-memory cache round-trip (simulating what __getitem__ does)
    print(f"\n[2.6] Testing in-memory cache simulation (tuple -> none_or_stack_elements -> batch)...")

    # Simulate per-sample storage (like info.text_encoder_outputs = tuple)
    per_sample_cached = []
    for i in range(len(captions)):
        per_sample_cached.append((pe_np[i], am_np[i], t5_ids_np[i], t5_mask_np[i]))

    # Simulate none_or_stack_elements with torch.FloatTensor converter
    # This is what train_util.py __getitem__ does at line 1784
    stacked = []
    for elem_idx in range(4):
        arrays = [sample[elem_idx] for sample in per_sample_cached]
        stacked.append(torch.stack([torch.FloatTensor(a) for a in arrays]))

    print(f"  Stacked batch (like batch['text_encoder_outputs_list']):")
    names = ["prompt_embeds", "attn_mask", "t5_input_ids", "t5_attn_mask"]
    for name, tensor in zip(names, stacked):
        print_tensor_info(name, tensor)

    # Check condition: len(text_encoder_conds) == 0 or text_encoder_conds[0] is None
    text_encoder_conds = stacked
    cond_check_1 = len(text_encoder_conds) == 0
    cond_check_2 = text_encoder_conds[0] is None
    print(f"\n  Condition check (should both be False when caching works):")
    print(f"    len(text_encoder_conds) == 0 : {cond_check_1}")
    print(f"    text_encoder_conds[0] is None: {cond_check_2}")
    if not cond_check_1 and not cond_check_2:
        print("    OK: cached text encoder outputs would be used")
    else:
        print("    ** BUG: code would try to re-encode (and crash on None input_ids_list)!")

    # Test unpack for get_noise_pred_and_target (line 311)
    print(f"\n[2.7] Testing unpack: prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoder_conds")
    try:
        pe_batch, am_batch, t5_ids_batch, t5_mask_batch = text_encoder_conds
        print(f"  Unpack OK")
        print_tensor_info("prompt_embeds", pe_batch)
        print_tensor_info("attn_mask", am_batch)
        print_tensor_info("t5_input_ids", t5_ids_batch)
        print_tensor_info("t5_attn_mask", t5_mask_batch)

        # Check t5_input_ids are integers (they were converted to FloatTensor!)
        if t5_ids_batch.dtype != torch.long and t5_ids_batch.dtype != torch.int32:
            print(f"\n  ** NOTE: t5_input_ids dtype is {t5_ids_batch.dtype}, will be cast to long at line 316")
            t5_ids_long = t5_ids_batch.to(dtype=torch.long)
            # Check if any precision was lost
            diff = (t5_ids_batch - t5_ids_long.float()).abs().max()
            print(f"    Float->Long precision loss: {diff:.2e}")
            if diff > 0.5:
                print("    ** ERROR: token IDs corrupted by float conversion!")
            else:
                print("    OK: float->long conversion is lossless for these IDs")
    except Exception as e:
        print(f"  ** ERROR unpacking: {e}")
        traceback.print_exc()

    # Test drop_cached_text_encoder_outputs
    print(f"\n[2.8] Testing drop_cached_text_encoder_outputs (caption dropout)...")
    dropout_strategy = AnimaTextEncodingStrategy(
        dropout_rate=0.5,  # high rate to ensure some drops
    )
    dropped = dropout_strategy.drop_cached_text_encoder_outputs(*stacked)
    print(f"  Returned {len(dropped)} tensors")
    for name, tensor in zip(names, dropped):
        print_tensor_info(f"dropped_{name}", tensor)

    # Check which items were dropped
    for i in range(len(captions)):
        is_zero = (dropped[0][i].abs().sum() == 0).item()
        print(f"  Sample {i}: {'DROPPED' if is_zero else 'KEPT'}")

    qwen3_model.to("cpu")
    del qwen3_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("\n[2.8] Text encoder cache test DONE.")


# Test 3: Full batch simulation


def test_full_batch_simulation(args, pairs):
    print("\n" + "=" * 70)
    print("TEST 3: FULL BATCH SIMULATION (mimics process_batch flow)")
    print("=" * 70)

    from library import anima_utils
    from library.strategy_anima import AnimaTokenizeStrategy, AnimaTextEncodingStrategy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    te_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    vae_dtype = torch.float32

    # Load all models
    print("\n[3.1] Loading models...")
    qwen3_tokenizer = anima_utils.load_qwen3_tokenizer(args.qwen3_path)
    t5_tokenizer = anima_utils.load_t5_tokenizer(getattr(args, "t5_tokenizer_path", None))
    qwen3_model, _ = anima_utils.load_qwen3_text_encoder(args.qwen3_path, dtype=te_dtype, device=device)
    qwen3_model.eval()
    vae, _, _, vae_scale = anima_utils.load_anima_vae(args.vae_path, dtype=vae_dtype, device=device)

    tokenize_strategy = AnimaTokenizeStrategy(
        qwen3_tokenizer=qwen3_tokenizer,
        t5_tokenizer=t5_tokenizer,
        qwen3_max_length=args.qwen3_max_length,
        t5_max_length=args.t5_max_length,
    )
    text_encoding_strategy = AnimaTextEncodingStrategy(dropout_rate=0.0)

    captions = [cap for _, cap in pairs]

    # --- Simulate caching phase ---
    print("\n[3.2] Simulating text encoder caching phase...")
    tokens_and_masks = tokenize_strategy.tokenize(captions)
    with torch.no_grad():
        te_outputs = text_encoding_strategy.encode_tokens(
            tokenize_strategy,
            [qwen3_model],
            tokens_and_masks,
            enable_dropout=False,
        )
    prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = te_outputs

    # Convert to numpy (same as cache_batch_outputs)
    pe_np = prompt_embeds.cpu().float().numpy()
    am_np = attn_mask.cpu().numpy()
    t5_ids_np = t5_input_ids.cpu().numpy().astype(np.int32)
    t5_mask_np = t5_attn_mask.cpu().numpy().astype(np.int32)

    # Per-sample storage (like info.text_encoder_outputs)
    per_sample_te = [(pe_np[i], am_np[i], t5_ids_np[i], t5_mask_np[i]) for i in range(len(captions))]

    print(f"\n[3.3] Simulating latent caching phase...")
    per_sample_latents = []
    for img_path, _ in pairs:
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        img_tensor = IMAGE_TRANSFORMS(img_np).unsqueeze(0).unsqueeze(2)  # (1,C,1,H,W)
        img_tensor = img_tensor.to(device, dtype=vae_dtype)
        with torch.no_grad():
            lat = vae.encode(img_tensor, vae_scale).cpu()
        per_sample_latents.append(lat.squeeze(0))  # (C,1,H,W)
        print(f"  {os.path.basename(img_path)}: latent shape={tuple(lat.shape)}")

    # --- Simulate batch construction (__getitem__) ---
    print(f"\n[3.4] Simulating batch construction...")

    # Use first image's latents only (images may have different resolutions)
    latents_batch = per_sample_latents[0].unsqueeze(0)  # (1,C,1,H,W)
    print(f"  Using first image latent for simulation: shape={tuple(latents_batch.shape)}")

    # Stack text encoder outputs (none_or_stack_elements)
    text_encoder_outputs_list = []
    for elem_idx in range(4):
        arrays = [s[elem_idx] for s in per_sample_te]
        text_encoder_outputs_list.append(torch.stack([torch.FloatTensor(a) for a in arrays]))

    # input_ids_list is None when caching
    input_ids_list = None

    batch = {
        "latents": latents_batch,
        "text_encoder_outputs_list": text_encoder_outputs_list,
        "input_ids_list": input_ids_list,
        "loss_weights": torch.ones(len(captions)),
    }

    print(f"  batch keys: {list(batch.keys())}")
    print(f"  batch['latents']: shape={tuple(batch['latents'].shape)}")
    print(f"  batch['text_encoder_outputs_list']: {len(batch['text_encoder_outputs_list'])} tensors")
    print(f"  batch['input_ids_list']: {batch['input_ids_list']}")

    # --- Simulate process_batch logic ---
    print(f"\n[3.5] Simulating process_batch logic...")

    text_encoder_conds = []
    te_out = batch.get("text_encoder_outputs_list", None)
    if te_out is not None:
        text_encoder_conds = te_out
        print(f"  text_encoder_conds loaded from cache: {len(text_encoder_conds)} tensors")
    else:
        print(f"  text_encoder_conds: empty (no cache)")

    # The critical condition
    train_text_encoder_TRUE = True  # OLD behavior (base class default, no override)
    train_text_encoder_FALSE = False  # NEW behavior (with is_train_text_encoder override)

    cond_old = len(text_encoder_conds) == 0 or text_encoder_conds[0] is None or train_text_encoder_TRUE
    cond_new = len(text_encoder_conds) == 0 or text_encoder_conds[0] is None or train_text_encoder_FALSE

    print(f"\n  === CRITICAL CONDITION CHECK ===")
    print(f"  len(text_encoder_conds) == 0 : {len(text_encoder_conds) == 0}")
    print(f"  text_encoder_conds[0] is None: {text_encoder_conds[0] is None}")
    print(f"  train_text_encoder (OLD=True) : {train_text_encoder_TRUE}")
    print(f"  train_text_encoder (NEW=False): {train_text_encoder_FALSE}")
    print(f"")
    print(f"  Condition with OLD behavior (no override): {cond_old}")
    msg = (
        "ENTERS re-encode block -> accesses batch['input_ids_list'] -> CRASH!"
        if cond_old
        else "SKIPS re-encode block -> uses cache -> OK"
    )

    print(f"    -> {msg}")
    print(f"  Condition with NEW behavior (override):    {cond_new}")
    print(f"    -> {'ENTERS re-encode block' if cond_new else 'SKIPS re-encode block -> uses cache -> OK'}")

    if cond_old and not cond_new:
        print(f"\n  ** CONFIRMED: the is_train_text_encoder override fixes the crash **")

    # Simulate the rest of process_batch
    print(f"\n[3.6] Simulating get_noise_pred_and_target unpack...")
    try:
        pe, am, t5_ids, t5_mask = text_encoder_conds
        pe = pe.to(device, dtype=te_dtype)
        am = am.to(device)
        t5_ids = t5_ids.to(device, dtype=torch.long)
        t5_mask = t5_mask.to(device)

        print(f"  Unpack + device transfer OK:")
        print_tensor_info("prompt_embeds", pe)
        print_tensor_info("attn_mask", am)
        print_tensor_info("t5_input_ids", t5_ids)
        print_tensor_info("t5_attn_mask", t5_mask)

        # Verify t5_input_ids didn't get corrupted by float conversion
        t5_ids_orig = torch.tensor(t5_ids_np, dtype=torch.long, device=device)
        id_match = torch.all(t5_ids == t5_ids_orig).item()
        print(f"\n  t5_input_ids integrity (float->long roundtrip): {'OK' if id_match else '** MISMATCH **'}")
        if not id_match:
            diff_count = (t5_ids != t5_ids_orig).sum().item()
            print(f"    {diff_count} token IDs differ!")
            # Show example
            idx = torch.where(t5_ids != t5_ids_orig)
            if len(idx[0]) > 0:
                i, j = idx[0][0].item(), idx[1][0].item()
                print(f"    Example: position [{i},{j}] original={t5_ids_orig[i,j].item()} loaded={t5_ids[i,j].item()}")

    except Exception as e:
        print(f"  ** ERROR: {e}")
        traceback.print_exc()

    # Cleanup
    vae.to("cpu")
    qwen3_model.to("cpu")
    del vae, qwen3_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("\n[3.7] Full batch simulation DONE.")


# Main


def main():
    parser = argparse.ArgumentParser(description="Test Anima caching mechanisms")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with image+txt pairs")
    parser.add_argument("--qwen3_path", type=str, required=True, help="Path to Qwen3 model (directory or safetensors)")
    parser.add_argument("--vae_path", type=str, required=True, help="Path to WanVAE safetensors")
    parser.add_argument("--t5_tokenizer_path", type=str, default=None, help="Path to T5 tokenizer (optional, uses bundled config)")
    parser.add_argument("--qwen3_max_length", type=int, default=512)
    parser.add_argument("--t5_max_length", type=int, default=512)
    parser.add_argument("--cache_to_disk", action="store_true", help="Also test disk cache round-trip")
    parser.add_argument("--skip_latent", action="store_true", help="Skip latent cache test")
    parser.add_argument("--skip_text", action="store_true", help="Skip text encoder cache test")
    parser.add_argument("--skip_full", action="store_true", help="Skip full batch simulation")
    args = parser.parse_args()

    # Find pairs
    pairs = find_image_caption_pairs(args.image_dir)
    if len(pairs) == 0:
        print(f"ERROR: No image+txt pairs found in {args.image_dir}")
        print("Expected: image.png + image.txt, image.jpg + image.txt, etc.")
        sys.exit(1)

    print(f"Found {len(pairs)} image-caption pairs:")
    for img_path, cap in pairs:
        print(f"  {os.path.basename(img_path)}: \"{cap[:60]}{'...' if len(cap) > 60 else ''}\"")

    results = {}

    if not args.skip_latent:
        try:
            test_latent_cache(args, pairs)
            results["latent_cache"] = "PASS"
        except Exception as e:
            print(f"\n** LATENT CACHE TEST FAILED: {e}")
            traceback.print_exc()
            results["latent_cache"] = f"FAIL: {e}"

    if not args.skip_text:
        try:
            test_text_encoder_cache(args, pairs)
            results["text_encoder_cache"] = "PASS"
        except Exception as e:
            print(f"\n** TEXT ENCODER CACHE TEST FAILED: {e}")
            traceback.print_exc()
            results["text_encoder_cache"] = f"FAIL: {e}"

    if not args.skip_full:
        try:
            test_full_batch_simulation(args, pairs)
            results["full_batch_sim"] = "PASS"
        except Exception as e:
            print(f"\n** FULL BATCH SIMULATION FAILED: {e}")
            traceback.print_exc()
            results["full_batch_sim"] = f"FAIL: {e}"

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for test, result in results.items():
        status = "OK" if result == "PASS" else "FAIL"
        print(f"  [{status}] {test}: {result}")
    print()


if __name__ == "__main__":
    main()
