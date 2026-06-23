# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

DefMambaJSCC (paper repo "MambaJSCC", arXiv:2409.16592): a Mamba/VSSM-based **Joint Source-Channel Coding** system for transmitting images over a noisy wireless channel, extended with a downstream **classification** task on the channel-output latent. The research contribution layered on top is a **deformable scan "third path"** (the `tri_merge` / DefScan branch).

## Two independent tasks (do not re-couple them)

The codebase is split into two tasks that share a backbone but never import each other. Both build only on `utils/engine.py`.

- **Reconstruction** — `run/train.py` (`train_reconstruction`) and `run/eval.py` (`evaluate_reconstruction`, `test_reconstruction`). Entry: `main.py`. Pipeline: `image → Mamba_encoder → Channel → Mamba_decoder → image`, loss `L_rec` (PSNR/MS-SSIM/LPIPS via `TRAIN.LOSS`).
- **Classification** — `tasks/classification/` (`train_classification`, `evaluate_classification`, `test_classification`). Entry: `tasks/classification/main_cls.py`. Pipeline: `image → Mamba_encoder → Channel → LatentClassifierHead → logits`, loss `L_cls` (CrossEntropy). Two stages via `CLS.STAGE`: `from_scratch` or `finetune_encoder` (warm-start encoder from a reconstruction checkpoint). Both stages train encoder + head.

A previous "multitask serial fine-tuning" flow (joint loss, `head_only` stage, `run/train_multitask.py`, `run/eval_multitask.py`, `models/task_head.py`) was **removed**. Do not reintroduce it. The classifier head now lives at `tasks/classification/task_head.py`.

## Shared engine — `utils/engine.py`

Both tasks pull plumbing from here; it depends only on low-level utils (never on `run`/`tasks`). Reuse these instead of re-implementing: `setup_distributed`/`is_main_process`/`reduce_scalar`/`unwrap_parallel`, `build_adamw_with_tri_merge`, `build_warmup_cosine_schedulers`, `apply_channel`, `checkpoint_tag`, `get_log_dir`, `collect_tri_path_weight_stats`/`save_tri_path_weight_curve`, `print_pipeline_profile`.

- `apply_channel(channel, config, feature, snr)` is the single source of truth for channel post-processing (Rayleigh inverse filter + `cat(real, imag) * sqrt(pwr)`). The encoder/decoder forward signature is `forward(x, snr)`.
- The DefScan branch is trained with a **separate optimizer param group** (params whose name contains `tri_merge`, LR scaled by `MODEL.VSSM.TRI_MERGE_LR_MULT`, no weight decay) — handled inside `build_adamw_with_tri_merge`. Its learned weights are logged to `tri_path_weight_curve.{csv,png}` each run.

## Config system (yacs, multi-file merge)

`configs/config.py` holds defaults (`_C`). `get_config(args)` merges, in order, `args.model_config_path` then `args.train_config_path`. `main.py` derives both from `--config_name`: `configs/vssm/vssm_tiny_<name>.yaml` (architecture) + `configs/train/vssm_tiny_<name>.yaml` (training/data/channel).

- **yacs errors on unknown keys** — any key in a YAML must exist in `_C`. When adding config, add it to `configs/config.py` first.
- Classification settings live under the `CLS` namespace (renamed from the old `TASK`); reconstruction does not use `CLS`.
- Key architecture knobs: `MODEL.VSSM.OUT_CHANS` (latent dim), `MODEL.VSSM.USE_DEFSCAN`, `CHANNEL.ADAPTIVE` (`ssm`/`attn`/`no`), `CHANNEL.TYPE` (`awgn`/`rayleigh`), `CHANNEL.SNR` (list; randomly sampled per batch when training, swept per-SNR when evaluating).

## Hard environment dependency: custom CUDA cores

`models/vmamba.py` imports a compiled CUDA extension at module load and has **no pure-PyTorch fallback** — the model cannot even be imported (let alone trained) without one built. Requires Linux + NVIDIA GPU + CUDA toolkit + MSVC/gcc host compiler.

- `adaptive_selective_scan_cuda_core` — required for `CHANNEL.ADAPTIVE='ssm'` (the paper's CSI-ReST method). Build: `cd adaptive_selective_scan && rm -rf dist build && pip install .`
- `selective_scan_cuda` — only for `attn`/`no` ablations (optional). Build: `cd selective_scan && rm -rf dist build && pip install .`

Reconstruction also hardcodes `cuda` and `ReconstructionLoss` builds `lpips().cuda()`, so a GPU is mandatory.

## Commands

```bash
# Install
pip install -r requirements.txt
cd adaptive_selective_scan && rm -rf dist build && pip install .   # mandatory CUDA core

# Reconstruction (config_name maps to configs/{vssm,train}/vssm_tiny_<config_name>.yaml)
python main.py --mode train --config_name CIFAR10_multitask_defscan
python main.py --mode test  --config_name CIFAR10_multitask_defscan

# Classification (independent entry; --stage overrides CLS.STAGE)
python tasks/classification/main_cls.py --mode train --stage from_scratch \
  --config tasks/classification/configs/CIFAR10_cls_from_scratch.yaml
python tasks/classification/main_cls.py --mode train --stage finetune_encoder \
  --config tasks/classification/configs/CIFAR10_cls_finetune_encoder.yaml \
  --pretrain_encoder <path/to/reconstruction_encoder.pt>

# Multi-GPU: launch via torchrun (code keys off the WORLD_SIZE env var, only rank 0 saves/logs)

# CUDA-core correctness test (the only pytest in the repo)
pytest selective_scan/test_selective_scan.py
```

There is no project-wide unit-test suite or linter config. `tools/` holds diagnostics (`inspect_triproj_ablation.py`, `inspect_triproj_weight.py`, `smoke_defscan.py`) for analyzing the DefScan branch.

## Gotchas

- **Paths in YAMLs are absolute Linux server paths** (`/home/LYC/...`, `/mnt/wutong/...`). Edit dataset/checkpoint paths per environment before running.
- **Checkpoints are full pickled model objects** (`torch.save(model)`), loaded with `weights_only=False`. Filenames are generated by `engine.checkpoint_tag(config)` (reconstruction) and `classification_checkpoint_name(config)` (classification, prefixed `CLS_{stage}_`); changing the config knobs in the tag changes which file `--mode test` looks for.
- CIFAR-10 inputs are interpolated to `DATA.IMG_SIZE` (128) inside the pipeline.
