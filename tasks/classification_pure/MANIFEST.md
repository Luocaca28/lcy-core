# PURE_CLS baseline — removal checklist

This directory is a **self-contained, removable baseline** hosting the two
**no-channel** rungs of a 3-tier attribution ladder (`image -> encoder ->
classifier`, `z_hat = feature`). It exists to be deleted once the comm version
resumes. Tier (3) is your existing comm code and is NOT here.

Ladder (each rung differs from the next by exactly one thing):
- **(1) pure compression** — `Cifar100_pure_cls.yaml` / `Imagenet100_pure_cls.yaml`:
  `ADAPTIVE='no'`, `USE_SNR_EMBED=False`. Encoder as a pure compressor; SNR unused.
  = the representation upper bound.
- **(2) SNR=20, no channel** — `Cifar100_snr20_nochan.yaml` /
  `Imagenet100_snr20_nochan.yaml`: `ADAPTIVE='ssm'`, `USE_SNR_EMBED=True`, fed a
  constant 20 dB. Structurally identical to the comm version, just no noise.
- **(3) normal channel** — your existing comm code, untouched.

Attribution: **(1) vs (2)** = effect of the CSI-ReST + SNR-embed mechanism itself
(noise-free); **(2) vs (3)** = the pure channel-noise cost (structures identical).
Both (2) and (3) must share the SAME seed / epochs / aug / batch / LR for the
(2)-vs-(3) diff to mean "noise only"; likewise (1) and (2) here are kept identical
except the two structural knobs.

## To delete ("删除纯压缩分类任务相关的代码")

1. Delete this directory: `tasks/classification_pure/`
2. Delete the product directories (checkpoints + logs), i.e. any path containing
   `PURE_CLS/` — e.g. `.../MambaJSCCcheckpoints/PURE_CLS/`
3. Confirm no code residue:
   `grep -rn "pure_cls" .` and `grep -rn "PURE_CLS" .` → expect **zero** hits
   outside deleted paths.
4. This experiment modified **NO existing file**. To prove it:
   `git diff --stat before-pure-cls` should show only NEW files under
   `tasks/classification_pure/` (git tag `before-pure-cls` marks the pre-baseline
   state). Any *modification* to an existing file is unexpected residue.

   Exception (not part of this baseline, keep it): the tier-3 comm config
   `tasks/classification/configs/Cifar100_cls_ssm.yaml` is a permanent
   comm-experiment artifact that runs on the untouched comm code. It survives the
   tear-off; do NOT delete it with the pure baseline.

## What it reuses (imports only — never modified)

- `models/network.py` `Mamba_encoder`, `tasks/classification/task_head.py`
  `LatentClassifierHead`, `data/datasets.py` `get_loader`/`TransformSubset`,
  and `utils/engine.py` helpers (optimizer/scheduler/logging/checkpoint naming).
- All `CHANNEL.*` keys it reads already exist in `configs/config.py` defaults, so
  `config.py` is untouched; `ADAPTIVE: 'no'` makes the encoder ignore SNR
  (`models/vmamba.py` selective-scan non-`ssm` path drops the `snr` arg).

## Caveat (do not misuse the number)

Because this baseline uses `ADAPTIVE='no'` + no SNR embed + a *copied* train loop,
it is a clean **standalone baseline**, not a single-variable comparison against the
`ssm` comm run. Do **not** subtract the two to claim a "channel cost" — that needs
a same-encoder `ADAPTIVE='no'`-with-channel run.

## Run

```bash
# default config is CIFAR-100 (Cifar100_pure_cls.yaml); ImageNet-100 also available.
python tasks/classification_pure/main_pure_cls.py --mode train \
  --config tasks/classification_pure/configs/Cifar100_pure_cls.yaml \
  > train_cifar100_pure_cls.log 2>&1

# test only (reuses the saved checkpoints):
python tasks/classification_pure/main_pure_cls.py --mode test \
  --config tasks/classification_pure/configs/Cifar100_pure_cls.yaml
```
