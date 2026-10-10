from __future__ import annotations

import math

from v1e_field_data_common import PARAM_KEYS, torch

nn = torch.nn

class SpectralConv2d(nn.Module):


    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        scale = 1 / math.sqrt(max(1, in_channels * out_channels))
        self.weights_pos = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat))
        self.weights_neg = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat))

    @staticmethod
    def compl_mul2d(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        if self.modes_x > height // 2 or self.modes_y > width // 2:
            raise ValueError(f"FNO modes ({self.modes_x},{self.modes_y}) exceed conservative Nyquist for grid {height}x{width}")
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            height,
            width // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes_x, : self.modes_y] = self.compl_mul2d(
            x_ft[:, :, : self.modes_x, : self.modes_y], self.weights_pos
        )
        out_ft[:, :, -self.modes_x :, : self.modes_y] = self.compl_mul2d(
            x_ft[:, :, -self.modes_x :, : self.modes_y], self.weights_neg
        )
        return torch.fft.irfft2(out_ft, s=(height, width), dim=(-2, -1))

class FNOBlock2d(nn.Module):
    def __init__(self, width: int, modes_x: int, modes_y: int, dropout: float) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_x, modes_y)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.spectral(x) + self.pointwise(x)))

class FNOFieldParametric(torch.nn.Module):
    def __init__(self, in_channels: int, desc_dim: int, width: int, modes: int, layers: int, dropout: float, use_descriptors: bool) -> None:
        super().__init__()
        self.use_descriptors = use_descriptors
        self.lift = torch.nn.Conv2d(in_channels, width, 1)
        self.blocks = torch.nn.ModuleList([FNOBlock2d(width, modes, modes, dropout) for _ in range(layers)])
        self.field_head = torch.nn.Sequential(torch.nn.Conv2d(width, width, 1), torch.nn.GELU(), torch.nn.Dropout2d(dropout), torch.nn.Conv2d(width, 4, 1))
        if use_descriptors and desc_dim:
            self.desc_encoder = torch.nn.Sequential(torch.nn.Linear(desc_dim, 128), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(128, 64), torch.nn.GELU())
            head_in = width + 64
        else:
            self.desc_encoder = None
            head_in = width
        self.param_head = torch.nn.Sequential(torch.nn.Linear(head_in, width), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(width, len(PARAM_KEYS)))
        self.area_head = torch.nn.Linear(head_in, 3)
        self.condition_head = torch.nn.Linear(head_in, 4)

    def forward(self, x, desc=None):
        h = self.lift(x)
        for block in self.blocks:
            h = block(h)
        pooled = h.mean(dim=(2, 3))
        if self.desc_encoder is not None and desc is not None and desc.shape[1] > 0:
            pooled = torch.cat([pooled, self.desc_encoder(desc)], dim=1)
        return self.field_head(h), self.param_head(pooled), self.area_head(pooled), self.condition_head(pooled)
