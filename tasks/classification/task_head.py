import torch
import torch.nn as nn


class LatentClassifierHead(nn.Module):
    """信道后 latent -> 分类 logits.

    head_type="pool"  (旧版, 默认兼容): 全局平均池化, 塌成 latent_dim 向量.
    head_type="attn_pool" (新): 用一组可学习 query 对信道后的
      (B, C, H, W) / (B, C, N) 特征做 cross-attention 读出, 保留空间维度里的
      判别信息, 再分类. 纯 nn.MultiheadAttention 实现, 不依赖任何 Mamba
      CUDA 算子, 不改 encoder / 信道, 零新依赖.
    """

    def __init__(
        self,
        latent_dim,
        num_classes,
        hidden_dim=256,
        snr_embed_dim=32,
        use_snr=True,
        snr_max=20.0,
        dropout=0.0,
        head_type="pool",
        num_queries=4,
        attn_heads=4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_snr = use_snr
        self.snr_max = max(float(snr_max), 1.0)
        self.head_type = head_type

        if head_type == "attn_pool":
            self.queries = nn.Parameter(torch.randn(1, num_queries, latent_dim) * 0.02)
            self.attn = nn.MultiheadAttention(
                embed_dim=latent_dim, num_heads=attn_heads, batch_first=True
            )
            self.pre_norm = nn.LayerNorm(latent_dim)
            pooled_dim = latent_dim * num_queries
        else:
            pooled_dim = latent_dim

        self.norm = nn.LayerNorm(pooled_dim)

        if use_snr:
            self.snr_embed = nn.Sequential(
                nn.Linear(1, snr_embed_dim),
                nn.GELU(),
                nn.Linear(snr_embed_dim, snr_embed_dim),
                nn.GELU(),
            )
            head_in_dim = pooled_dim + snr_embed_dim
        else:
            self.snr_embed = None
            head_in_dim = pooled_dim

        self.head = nn.Sequential(
            nn.Linear(head_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _to_tokens(self, z):
        if z.dim() == 4:
            b, c, h, w = z.shape
            if c != self.latent_dim:
                raise ValueError(
                    f"LatentClassifierHead expects latent dim {self.latent_dim}, got {c}"
                )
            return z.flatten(2).transpose(1, 2)
        elif z.dim() == 3:
            if z.shape[-1] == self.latent_dim:
                return z
            return z.transpose(1, 2)
        elif z.dim() == 2:
            return z.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported latent shape {z.shape}")

    def _pool_latent(self, z):
        if z.dim() == 4:
            pooled = z.mean(dim=(2, 3))
        elif z.dim() == 3:
            pooled = z.mean(dim=1)
        elif z.dim() == 2:
            pooled = z
        else:
            pooled = z.flatten(1)
        if pooled.shape[-1] != self.latent_dim:
            raise ValueError(
                f"LatentClassifierHead expects latent dim {self.latent_dim}, "
                f"got {pooled.shape[-1]}"
            )
        return pooled

    def _attn_pool(self, z):
        tokens = self._to_tokens(z)
        tokens = self.pre_norm(tokens)
        b = tokens.shape[0]
        queries = self.queries.expand(b, -1, -1)
        out, _ = self.attn(queries, tokens, tokens)
        return out.flatten(1)

    def _snr_tensor(self, snr, batch_size, device, dtype):
        if torch.is_tensor(snr):
            snr_tensor = snr.to(device=device, dtype=dtype).view(-1, 1)
            if snr_tensor.shape[0] == 1:
                snr_tensor = snr_tensor.expand(batch_size, -1)
            return snr_tensor
        return torch.full((batch_size, 1), float(snr), device=device, dtype=dtype)

    def forward(self, z, snr=None):
        if self.head_type == "attn_pool":
            pooled = self._attn_pool(z)
        else:
            pooled = self._pool_latent(z)
        pooled = self.norm(pooled)

        if self.use_snr:
            if snr is None:
                raise ValueError("snr must be provided when use_snr=True")
            snr_tensor = self._snr_tensor(snr, pooled.shape[0], pooled.device, pooled.dtype)
            snr_feat = self.snr_embed(snr_tensor / self.snr_max)
            pooled = torch.cat([pooled, snr_feat], dim=-1)
        return self.head(pooled)
