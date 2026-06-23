# DefMambaJSCC 双任务汇报：重建任务 + 分类任务

## 总体设计

DefMambaJSCC 不再使用"重建-分类多任务串行微调"的耦合流程，而是拆分为**两个相互独立的任务**，各自拥有独立的训练/评估代码与配置，共享同一套底层引擎（`utils/engine.py`：分布式、优化器/调度器、信道前向、模型 profiling、可变形第三路径权重统计、checkpoint 命名）。

```text
            ┌──────────────────────────────┐
            │      共享主干 Mamba Encoder      │
            │      共享信道 Channel            │
            │      共享引擎 utils/engine.py     │
            └──────────────┬───────────────┘
            ┌──────────────┴───────────────┐
   重建任务 (run/)                     分类任务 (tasks/classification/)
   encoder + decoder                  encoder + latent classifier
   L = L_rec (PSNR/MS-SSIM/LPIPS)      L = L_cls (CrossEntropy)
```

两个任务彼此不互相 import，结构对称、命名清晰，便于单独训练、评估和对比。

## 任务一：图像重建（Reconstruction）

| 项目 | 说明 |
|---|---|
| 入口 | `run/train.py: train_reconstruction` / `run/eval.py: test_reconstruction` |
| 命令 | `python main.py --mode train --config_name <recon配置>` |
| 结构 | `input -> encoder -> channel -> decoder -> reconstructed image` |
| 损失 | `L = L_rec`（由 `TRAIN.LOSS` 选择 PSNR/MSSSIM/LPIPS） |
| 指标 | PSNR、MS-SSIM、SNR 曲线、第三路径权重曲线 |
| 目的 | 学到稳定、信道鲁棒的 JSCC 重建主干 |

预期：PSNR/MS-SSIM 随 SNR 增大上升；train/val loss 稳定下降；可变形第三路径权重被模型合理利用。

## 任务二：信道后分类（Classification）

分类头 `LatentClassifierHead` 作用在信道后 latent 上，支持注入 SNR 嵌入。提供两种独立可选的训练模式（`CLS.STAGE`），两者都训练 encoder + 分类头：

| 阶段 (`CLS.STAGE`) | 初始化 | 损失 | 目的 |
|---|---|---|---|
| `from_scratch` | encoder + 分类头从随机初始化联合训练 | `L_cls` | 端到端学习面向分类的表示（直接评估模型分类能力） |
| `finetune_encoder` | 载入重建预训练 encoder 后与分类头一起微调 | `L_cls` | 从重建主干迁移，再为分类微调 |

| 项目 | 说明 |
|---|---|
| 入口 | `tasks/classification/train_cls.py: train_classification` / `eval_cls.py: test_classification` |
| 命令 | `python tasks/classification/main_cls.py --mode train --stage from_scratch` |
| 结构 | `input -> encoder -> channel -> latent classifier -> logits` |
| 指标 | Accuracy、SNR-Accuracy 曲线、train/val cls loss |
| 目的 | 评估信道后 latent 的分类判别能力，尤其低信噪比表现 |

预期：Accuracy 明显高于 CIFAR-10 随机猜测的 10%；SNR 越高准确率趋势越好。

## 与旧多任务流程的关系

旧的"Recon-only → Head-only → Encoder-head → Seq-joint"四阶段串行联合损失流程已废弃，对应的 `run/train_multitask.py`、`run/eval_multitask.py` 已删除。如需复现"先重建预训练、再用其 encoder 做分类"的实验，改为两步独立运行：

1. 跑重建任务得到 encoder checkpoint；
2. 跑分类任务 `--stage finetune_encoder --pretrain_encoder <重建得到的 encoder.pt>`。

这样能更清晰地区分"重建 latent 本身已有的语义"与"分类监督带来的任务导向增强"。
