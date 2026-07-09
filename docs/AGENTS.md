# docs/

## Purpose

Reference documentation supplementing the root `README.md`: per-feature guides, installation instructions, and localized (`-ja`/`-zh`) copies of training READMEs.

## Ownership

Content-only; no code. Owned jointly by whoever maintains the referenced feature.

## Local Contracts

- `Installation/` — per-platform/package-manager install guides (`pip_linux.md`, `pip_windows.md`, `uv_linux.md`, `uv_windows.md`).
- `LoRA/`, `Finetuning/` — feature-specific guides, each with a `top_level.md` overview.
- Root-level `*_README.md` / `*_README-ja.md` / `*_README-zh.md` pairs are translated copies — when updating an English doc that has translated siblings, note in the PR that translations are now stale (translations are community-maintained, not auto-synced).
- `image_folder_structure.md`, `troubleshooting_tesla_v100.md`, `installation_docker.md`, `installation_runpod.md`, `installation_novita.md` are standalone single-topic guides, not part of a series.

## Work Guidance

- New feature guides should follow the existing `<Feature>/top_level.md` pattern rather than a single flat file, if the feature is expected to grow multiple sub-topics (options, troubleshooting, etc.) — otherwise a flat root-level `.md` is fine.
- Don't duplicate installation steps already covered in the root `README.md`'s Installation Options section; link to it instead.

## Verification

- `.github/workflows/typos.yaml` lints spelling across the repo, docs included.

## Child DOX Index

None — sub-folders here are content collections, not durable code boundaries.
