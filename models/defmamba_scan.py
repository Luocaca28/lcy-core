import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class DeformablePathTrans(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, de_index):
        B, C, N = x.shape
        _, indices = torch.topk(de_index, k=N, dim=-1, largest=False)
        x_gathered = torch.gather(
            x, 2, indices.unsqueeze(1).expand(-1, C, -1)
        ).contiguous()
        x_out = x_gathered.permute(0, 2, 1).contiguous()
        ctx.save_for_backward(x, de_index, indices)
        return x_out, indices

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        x, de_index, indices = ctx.saved_tensors
        grad_x = torch.zeros_like(x)
        grad_x.scatter_add_(
            2,
            indices.unsqueeze(1).expand(-1, x.shape[1], -1),
            grad_output.permute(0, 2, 1).contiguous(),
        ).contiguous()
        grad_de_index = (
            grad_output.permute(0, 2, 1).contiguous() - grad_x
        ).mean(dim=1)
        grad_de_index = grad_de_index.view_as(de_index)
        return grad_x, grad_de_index


class ConvOffset(nn.Module):
    def __init__(self, embed_dim, kk, pad_size):
        super().__init__()
        self.conv1 = nn.Conv2d(
            embed_dim, embed_dim, kk, 1, pad_size, groups=embed_dim
        )
        self.ca = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 16),
            nn.GELU(),
            nn.Linear(embed_dim // 16, embed_dim),
            nn.Sigmoid(),
        )
        self.ln = nn.LayerNorm(embed_dim)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(embed_dim, 3, 1, 1, 0, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x_c = F.adaptive_avg_pool2d(x, (1, 1))
        x_c = self.ca(x_c.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x1 * x_c.expand_as(x)
        x = self.gelu(self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        return self.conv2(x)


class DeformableLayer(nn.Module):
    def __init__(self, index=0, embed_dim=192, debug=False, h=0, w=0):
        super().__init__()
        self.ksize = [9, 7, 5, 3]
        if index < 0 or index >= len(self.ksize):
            raise ValueError(f"stage index must be in [0, 3], got {index}")
        kk = self.ksize[index]
        pad_size = kk // 2 if kk != 1 else 0
        self.debug = debug
        self.conv_offset = ConvOffset(embed_dim, kk, pad_size)
        self.rpe_table = nn.Parameter(torch.zeros(embed_dim, 7, 7))
        trunc_normal_(self.rpe_table, std=0.01)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing="ij",
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        return ref[None, ...].expand(B, -1, -1, -1)

    @torch.no_grad()
    def _get_key_ref_points(self, H, W, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0, H, H, dtype=dtype, device=device),
            torch.linspace(0, W, W, dtype=dtype, device=device),
            indexing="ij",
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        return ref[None, ...].expand(B, -1, -1, -1)

    @torch.no_grad()
    def _get_path_ref_points(self, N, B, dtype, device):
        ref_path = torch.linspace(0.5, N - 0.5, N, dtype=dtype, device=device)
        ref_path.div_(N - 1.0).mul_(2.0).sub_(1.0)
        return ref_path[None, ...].expand(B, -1)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table", "rpe_table"}

    def forward(self, x):
        dtype, device = x.dtype, x.device
        B, C, H, W = x.size()
        N = H * W

        offset = self.conv_offset(x).contiguous()
        offset, de_index = torch.split(offset, [2, 1], dim=1)
        Hk, Wk = offset.size(2), offset.size(3)

        offset_range = torch.tensor(
            [1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device
        ).reshape(1, 2, 1, 1)
        offset = offset.tanh().mul(offset_range)
        offset = einops.rearrange(offset, "b p h w -> b h w p").contiguous()
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        de_index = de_index.tanh().flatten(1)
        path_reference = self._get_path_ref_points(N, B, dtype, device)
        pos = offset + reference
        path_pos = de_index + path_reference

        x_sampled = F.grid_sample(
            input=x,
            grid=pos[..., (1, 0)],
            mode="bilinear",
            align_corners=True,
        )

        rpe_bias = self.rpe_table[None, ...].expand(B, -1, -1, -1)
        rpe_bias = F.interpolate(
            rpe_bias, size=(H, W), mode="bilinear", align_corners=False
        )
        key_grid = self._get_key_ref_points(H, W, B, dtype, device)
        displacement = (key_grid - pos) * 0.5
        pos_bias = F.grid_sample(
            input=rpe_bias,
            grid=displacement[..., (1, 0)],
            mode="bilinear",
            align_corners=True,
        )

        x = (x_sampled + pos_bias).flatten(2)
        x, indices = DeformablePathTrans.apply(x, path_pos)
        return x, indices


class DeformableLayerReverse(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, indices=None):
        if indices is None:
            raise ValueError("indices are required for deformable reverse restore")
        x = x.flatten(2)
        B, C, N = x.size()
        index_re = torch.zeros_like(indices, device=x.device)
        index_re.scatter_add_(
            1,
            indices,
            torch.arange(indices.size(-1), device=x.device)
            .unsqueeze(0)
            .expand(indices.size(0), -1),
        )
        return torch.gather(x, 2, index_re.unsqueeze(1).expand(-1, C, -1))
