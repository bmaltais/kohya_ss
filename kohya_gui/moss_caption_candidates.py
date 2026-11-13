"""Offline tag-dropout caption candidates for MOSS-Image LoRA training.

MOSS-Image caches the Qwen3.5-9B text encoder outputs, and sd-scripts rejects
``--caption_tag_dropout_rate`` (and ``--shuffle_caption``) whenever that cache is
enabled (``library/dataset.py::is_text_encoder_output_cacheable``). MOSS-Image
does support multiple caption candidates per image though: with
``enable_multiline_caption`` every candidate is cached once and ONE IS CHOSEN AT
RANDOM EVERY STEP (``library/dataset.py`` ``random.randrange(len(cached_captions))``).

Pre-generating tag-dropout variants as candidates therefore reproduces the
per-step regularisation of tag dropout while keeping the text encoder off the
GPU.

Nothing original is touched:

* candidates are written to ``<image>.multi`` next to the image; the original
  ``<image>.txt`` (or whatever ``caption_extension`` the subset uses) is left
  exactly as it was, so other models keep training on the same captions;
* the user's dataset TOML is never edited - a derived copy
  ``<name>.moss-multi.toml`` is written next to it and handed to the trainer.
"""

import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import toml

log = logging.getLogger(__name__)

# Must match sd-scripts' DEFAULT_MULTILINE_CAPTION_SEPARATOR.
DEFAULT_SEPARATOR = "---===---"
CANDIDATE_EXTENSION = ".multi"
DERIVED_CONFIG_SUFFIX = ".moss-multi.toml"

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
)


def drop_tags(caption: str, rate: float, keep_tokens: int, rng: random.Random) -> str:
    """Drop comma separated tags with probability ``rate``.

    The first ``keep_tokens`` tags are never dropped (same semantics as
    sd-scripts' ``--keep_tokens``). The result is never empty: if every
    droppable tag was removed, the first one is kept.
    """
    tokens = [t.strip() for t in caption.strip().split(",")]
    tokens = [t for t in tokens if t]
    if not tokens:
        return ""
    keep_tokens = max(0, min(int(keep_tokens), len(tokens)))
    fixed, flexible = tokens[:keep_tokens], tokens[keep_tokens:]
    kept = [t for t in flexible if rng.random() >= rate]
    if not kept and flexible:
        kept = flexible[:1]
    return ", ".join(fixed + kept)


def build_candidates(
    caption: str,
    count: int,
    rate: float,
    keep_tokens: int,
    rng: random.Random,
) -> List[str]:
    """Return ``count`` candidates: the original caption plus dropout variants."""
    candidates = [caption.strip()]
    for _ in range(max(0, count - 1)):
        variant = drop_tags(caption, rate, keep_tokens, rng)
        candidates.append(variant if variant else caption.strip())
    return candidates


def render_candidate_file(
    candidates: List[str], rate: float, separator: str = DEFAULT_SEPARATOR
) -> str:
    """Render candidates as a multi-candidate caption file."""
    blocks: List[str] = []
    for index, candidate in enumerate(candidates):
        if index:
            blocks.append(f"{separator} # tag-dropout variant {index} (rate={rate})")
        blocks.append(candidate)
    return "\n".join(blocks) + "\n"


def _subsets_with_images(config: dict) -> List[Tuple[dict, dict, dict]]:
    """Yield (dataset, subset, general) triples for DreamBooth-style subsets."""
    general = config.get("general", {}) or {}
    result = []
    for dataset in config.get("datasets", []) or []:
        for subset in dataset.get("subsets", []) or []:
            if subset.get("image_dir"):
                result.append((dataset, subset, general))
    return result


def _effective_caption_extension(dataset: dict, subset: dict, general: dict) -> str:
    for level in (subset, dataset, general):
        value = level.get("caption_extension")
        if value:
            return value
    return ".caption"


def generate_for_directory(
    image_dir: str,
    source_extension: str,
    count: int,
    rate: float,
    keep_tokens: int,
    regenerate: bool = False,
    separator: str = DEFAULT_SEPARATOR,
    rng: Optional[random.Random] = None,
) -> Dict[str, int]:
    """Write ``<image>.multi`` files for every image in ``image_dir``.

    Existing candidate files are skipped unless ``regenerate`` is set, so
    repeated training runs do not rewrite the dataset.
    """
    rng = rng if rng is not None else random.Random()
    stats = {"written": 0, "skipped": 0, "missing": 0}
    if not os.path.isdir(image_dir):
        raise ValueError(f"image_dir does not exist: {image_dir}")

    for name in sorted(os.listdir(image_dir)):
        stem, extension = os.path.splitext(name)
        if extension.lower() not in IMAGE_EXTENSIONS:
            continue
        source = os.path.join(image_dir, stem + source_extension)
        target = os.path.join(image_dir, stem + CANDIDATE_EXTENSION)
        if not os.path.isfile(source):
            stats["missing"] += 1
            continue
        if os.path.exists(target) and not regenerate:
            stats["skipped"] += 1
            continue
        with open(source, "r", encoding="utf-8") as handle:
            caption = handle.read().strip()
        if not caption:
            stats["missing"] += 1
            continue
        candidates = build_candidates(caption, count, rate, keep_tokens, rng)
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(render_candidate_file(candidates, rate, separator))
        stats["written"] += 1
    return stats


def prepare_dataset_config(
    dataset_config: str,
    candidate_count: int = 3,
    dropout_rate: float = 0.15,
    keep_tokens: int = 0,
    regenerate: bool = False,
    separator: str = DEFAULT_SEPARATOR,
    seed: int = 20261110,
) -> str:
    """Generate caption candidates and return a derived dataset TOML path.

    The original TOML and the original caption files are never modified. The
    derived copy points every DreamBooth subset at ``.multi`` and enables
    ``enable_multiline_caption``.
    """
    if not dataset_config or not os.path.isfile(dataset_config):
        raise ValueError(f"dataset config not found: {dataset_config}")
    if int(candidate_count) < 2:
        raise ValueError("Caption Candidates must be at least 2")
    if not 0 < float(dropout_rate) < 1:
        raise ValueError("Caption Tag Dropout Rate must be between 0 and 1 (exclusive)")

    with open(dataset_config, "r", encoding="utf-8") as handle:
        config = toml.load(handle)

    triples = _subsets_with_images(config)
    if not triples:
        raise ValueError(
            "Multi-caption candidates need a dataset config with image_dir subsets "
            "(DreamBooth style). Fine-tuning subsets use caption metadata instead."
        )

    rng = random.Random(seed)
    totals = {"written": 0, "skipped": 0, "missing": 0}
    for dataset, subset, general in triples:
        source_extension = _effective_caption_extension(dataset, subset, general)
        if source_extension == CANDIDATE_EXTENSION:
            # Already a candidate file: keep it, just make sure it is used as such.
            log.info(
                f"Subset already uses {CANDIDATE_EXTENSION}: {subset['image_dir']}"
            )
        else:
            stats = generate_for_directory(
                subset["image_dir"],
                source_extension,
                int(candidate_count),
                float(dropout_rate),
                int(keep_tokens),
                regenerate=regenerate,
                separator=separator,
                rng=rng,
            )
            for key in totals:
                totals[key] += stats[key]
            if stats["missing"] and not stats["written"] and not stats["skipped"]:
                log.warning(
                    f"No '{source_extension}' captions found in {subset['image_dir']}; "
                    "that subset keeps its class tokens."
                )
        subset["caption_extension"] = CANDIDATE_EXTENSION
        subset["enable_multiline_caption"] = True
        subset["multiline_caption_separator"] = separator

    derived_path = os.path.splitext(dataset_config)[0] + DERIVED_CONFIG_SUFFIX
    header = (
        "# Generated by the kohya_ss GUI (MOSS-Image multi-caption candidates).\n"
        f"# Source: {os.path.basename(dataset_config)} - the original file is untouched.\n"
        f"# Candidates: {int(candidate_count)}, tag dropout rate: {float(dropout_rate)}, "
        f"keep_tokens: {int(keep_tokens)}.\n"
        f"# Caption files: <image>{CANDIDATE_EXTENSION} (originals untouched).\n"
    )
    with open(derived_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(header)
        toml.dump(config, handle)

    log.info(
        "MOSS-Image caption candidates: "
        f"written={totals['written']} skipped={totals['skipped']} "
        f"missing={totals['missing']} -> {derived_path}"
    )
    return derived_path
