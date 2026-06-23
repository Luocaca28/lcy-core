# Classification Task

This folder keeps the downstream CIFAR-10 classification task separate from the
main reconstruction entry. It reuses the shared encoder and channel modules from
`models/`, but trains only a task-specific classifier head.

The classification loss is independent:

```text
L_cls = CrossEntropy(logits, labels)
```

No decoder is created and no reconstruction loss is computed in this task.

Example commands:

```bash
python tasks/classification/main_cls.py \
  --config tasks/classification/configs/CIFAR10_cls_finetune_encoder.yaml \
  --mode train \
  --pretrain_encoder /path/to/reconstruction/encoder.pt

python tasks/classification/main_cls.py \
  --config tasks/classification/configs/CIFAR10_cls_from_scratch.yaml \
  --mode train
```

Recommended comparison:

```text
head_only              load reconstruction encoder, freeze encoder, train classifier
cls_finetune_encoder   load reconstruction encoder, train encoder + classifier
cls_from_scratch       train encoder + classifier from scratch
```
