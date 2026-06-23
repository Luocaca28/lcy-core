import torch
import torch.nn as nn


class LatentClassifierHead(nn.Module):
    def __init__(
        self,
        latent_dim,
        num_classes,
        hidden_dim=256,
        snr_embed_dim=32,
        use_snr=True,
        snr_max=20.0,
        dropout=0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_snr = use_snr
        self.snr_max = max(float(snr_max), 1.0)
        self.norm = nn.LayerNorm(latent_dim)

        if use_snr:
            self.snr_embed = nn.Sequential(
                nn.Linear(1, snr_embed_dim),
                nn.GELU(),
                nn.Linear(snr_embed_dim, snr_embed_dim),
                nn.GELU(),
            )
            head_in_dim = latent_dim + snr_embed_dim
        else:
            self.snr_embed = None
            head_in_dim = latent_dim

        self.head = nn.Sequential(
            nn.Linear(head_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

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
        return self.norm(pooled)

    def _snr_tensor(self, snr, batch_size, device, dtype):
        if torch.is_tensor(snr):
            snr_tensor = snr.to(device=device, dtype=dtype).view(-1, 1)
            if snr_tensor.shape[0] == 1:
                snr_tensor = snr_tensor.expand(batch_size, -1)
            return snr_tensor
        return torch.full((batch_size, 1), float(snr), device=device, dtype=dtype)

    def forward(self, z, snr=None):
        pooled = self._pool_latent(z)
        if self.use_snr:
            if snr is None:
                raise ValueError("snr must be provided when use_snr=True")
            snr_tensor = self._snr_tensor(snr, pooled.shape[0], pooled.device, pooled.dtype)
            snr_feat = self.snr_embed(snr_tensor / self.snr_max)
            pooled = torch.cat([pooled, snr_feat], dim=-1)
        return self.head(pooled)
