from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CorrelationOutput:
    response: torch.Tensor


class DepthwiseCrossCorrelation(nn.Module):
    """
    Depthwise cross-correlation for Siamese tracking.

    Inputs:
        z: template features [B, C, Hz, Wz]
        x: search features   [B, C, Hx, Wx]

    Output:
        response map         [B, C, Hr, Wr]

    Notes:
    - Uses grouped convolution to perform per-channel correlation.
    - Each sample in the batch is handled independently.
    - This is the common lightweight choice for Siamese trackers.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if z.ndim != 4 or x.ndim != 4:
            raise ValueError(
                f"Expected 4D tensors [B, C, H, W], got z={tuple(z.shape)}, x={tuple(x.shape)}"
            )

        bz, cz, hz, wz = z.shape
        bx, cx, hx, wx = x.shape

        if bz != bx:
            raise ValueError(f"Batch size mismatch: z batch={bz}, x batch={bx}")
        if cz != cx:
            raise ValueError(f"Channel mismatch: z channels={cz}, x channels={cx}")
        if hz > hx or wz > wx:
            raise ValueError(
                f"Template feature map must not be larger than search feature map: "
                f"z={tuple(z.shape)}, x={tuple(x.shape)}"
            )

        batch_size, channels = bz, cz

        # Grouped convolution trick:
        # reshape x to [1, B*C, Hx, Wx]
        # reshape z to [B*C, 1, Hz, Wz]
        # groups = B*C
        # => each channel of each sample is correlated independently
        x_reshaped = x.reshape(1, batch_size * channels, hx, wx)
        z_reshaped = z.reshape(batch_size * channels, 1, hz, wz)

        response = F.conv2d(
            input=x_reshaped,
            weight=z_reshaped,
            bias=None,
            stride=1,
            padding=0,
            groups=batch_size * channels,
        )

        hr, wr = response.shape[-2:]
        response = response.reshape(batch_size, channels, hr, wr)
        return response

    def forward_with_output(self, z: torch.Tensor, x: torch.Tensor) -> CorrelationOutput:
        return CorrelationOutput(response=self.forward(z, x))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z = torch.randn(2, 96, 4, 4, device=device)
    x = torch.randn(2, 96, 8, 8, device=device)

    corr = DepthwiseCrossCorrelation().to(device)
    out = corr(z, x)

    print("template features:", z.shape)
    print("search features:", x.shape)
    print("response map:", out.shape)