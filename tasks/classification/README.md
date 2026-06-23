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

Stages (`CLS.STAGE`, override with `--stage`) -- both train encoder + classifier:

```text
from_scratch       train encoder + classifier from random init
finetune_encoder   warm-start encoder from a reconstruction checkpoint, then fine-tune
```
