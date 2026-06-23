# DefMambaJSCC 多任务串行微调汇报

## 总体目标

本实验将 DefMambaJSCC 从单一图像重建任务扩展到重建-分类联合任务。核心思路不是直接从随机初始化开始联合训练，而是采用串行微调策略：

```text
Recon-only
    -> Head-only
    -> Encoder-head
    -> Seq-joint
```

这样可以逐步回答三个问题：

1. DefMambaJSCC 是否具备稳定的信道重建能力。
2. 只经过重建任务训练的信道后 latent 是否已经包含分类语义。
3. 分类监督是否能进一步增强 latent 的判别能力，同时保持重建质量。

## 四阶段训练策略

| 阶段 | 训练策略 | 冻结/更新对象 | 损失函数 | 目的 |
|---|---|---|---|---|
| Step 1: Recon-only | 纯重建预训练 | 更新 encoder + decoder；不训练 classifier | `L = L_rec` | 先得到稳定的 JSCC 重建主干 |
| Step 2: Head-only | 冻结主干，只训练分类头 | 冻结 encoder + decoder；更新 classifier | `L = L_cls` | 探测 reconstruction latent 是否包含分类语义 |
| Step 3: Encoder-head | 固定 decoder，调整 encoder 和分类头 | 更新 encoder + classifier；冻结 decoder | `L = 1.0 * L_rec + 0.1 * L_cls` | 让信道后 latent 更适合分类，同时保留可重建性 |
| Step 4: Seq-joint | 全模型联合微调 | 更新 encoder + decoder + classifier | `L = 1.0 * L_rec + 0.05 * L_cls` | 最终平衡图像重建质量和下游分类性能 |

## Step 1: Recon-only

结构：

```text
input image
    -> encoder
    -> channel
    -> decoder
    -> reconstructed image
```

训练目标：

text
L = L_rec

这一阶段只优化图像重建任务，不关注分类准确率。其目的是先让 DefMambaJSCC 学到稳定的信道鲁棒重建能力，并为后续分类任务提供一个可靠的 encoder-decoder backbone。

重点观察指标：

text
PSNR
MS-SSIM
train loss / val loss
tri_path_weight_curve


预期现象：

text
PSNR 和 MS-SSIM 随 SNR 增大而上升；
train/val loss 稳定下降；
Def 第三路径权重逐渐被模型使用。

## Step 2: Head-only

结构：

text
input image
    -> frozen encoder
    -> channel
    -> received latent
    -> classifier

训练目标：

text
L = L_cls

这一阶段冻结 encoder 和 decoder，只训练分类头。它相当于 latent probing，用来验证只经过重建任务训练的信道后 latent 是否已经包含可用于分类的语义判别信息。

重点观察指标：

text
Accuracy
SNR-Accuracy curve
train cls loss
val acc

预期现象：

text
Accuracy 明显高于 CIFAR-10 随机猜测的 10%；
SNR 越高，Accuracy 趋势上越好；
cls loss 稳定下降。

## Step 3: Encoder-head

结构：

```text
input image
    -> trainable encoder
    -> channel
    -> received latent
        ├── frozen decoder -> reconstructed image
        └── classifier -> logits
```

训练目标：

```text
L = 1.0 * L_rec + 0.1 * L_cls
```

这一阶段训练 encoder 和 classifier，冻结 decoder。虽然 decoder 不更新，但重建损失仍会通过 frozen decoder 的前向图反传到 encoder。因此，encoder 同时受到两个约束：

```text
rec_loss -> frozen decoder -> z_hat -> encoder
cls_loss -> classifier -> z_hat -> encoder
```

这样可以让信道后 latent 更适合分类，同时避免 encoder 产生完全偏离原 decoder 可重建空间的特征。

重点观察指标：

```text
Accuracy
SNR-Accuracy curve
train cls loss
val acc
PSNR / MS-SSIM
tri_path_weight_curve
```

预期现象：

```text
Accuracy 高于 Step 2；
PSNR/MS-SSIM 不发生明显崩溃；
第三路径权重继续发生合理变化。
```

## Step 4: Seq-joint

结构：

```text
input image
    -> encoder
    -> channel
    -> received latent
        ├── decoder -> reconstructed image
        └── classifier -> logits
```

训练目标：

```text
L = 1.0 * L_rec + 0.05 * L_cls
```

这一阶段解冻 encoder、decoder 和 classifier，进行最终联合微调。分类损失权重从 Step 3 的 `0.1` 降到 `0.05`，目的是避免过强分类监督破坏重建质量，在 PSNR/MS-SSIM 和 Accuracy 之间取得平衡。

重点观察指标：

```text
PSNR
MS-SSIM
Accuracy
SNR-Accuracy curve
train/val loss
tri_path_weight_curve
```

预期现象：

```text
Accuracy 保持或高于 Step 3；
PSNR/MS-SSIM 保持稳定；
SNR 越高，PSNR/MS-SSIM/Accuracy 整体更好；
第三路径权重稳定，没有异常发散。
```



## 汇报总结

本实验采用串行微调策略：先用重建任务训练 JSCC 主干，再冻结主干训练分类头探测 latent 语义，随后解冻 encoder 让信道后 latent 更适合分类，最后全模型联合微调，在重建质量和分类性能之间取得平衡。该流程能够区分 reconstruction latent 本身已有的语义信息和分类监督带来的任务导向语义增强，从而更清晰地解释 DefMambaJSCC 在下游分类任务中的作用。
