from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfPointBranch(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.size()
        query = self.query_conv(x).view(batch, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch, -1, width * height)
        attention = self.softmax(torch.bmm(query, key))
        value = self.value_conv(x).view(batch, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch, channels, height, width)
        return self.gamma * out + x


class SelfChannelBranch(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_dim, in_dim // 6, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_dim // 6, in_dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        attention = self.sigmoid(max_out + avg_out)
        return x + self.gamma * attention * x


class AdaptiveConcat(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.add = nn.ConvTranspose2d(in_dim * 2, in_dim, kernel_size=1, stride=1)
        self.fc1 = nn.Conv2d(in_dim, in_dim // 6, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_dim // 6, in_dim, 1, bias=False)
        self.gamma_add = nn.Parameter(torch.zeros(1))
        self.gamma_att = nn.Parameter(torch.zeros(1))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        weight = self.sigmoid(self.fc2(self.relu(self.fc1(self.avg_pool(z)))))
        combined = self.relu(self.add(torch.cat((x, z), dim=1)))
        return x + self.gamma_add * combined + self.gamma_att * weight * x


class APN(nn.Module):
    def __init__(self, apn_channels: int = 384, head_channels: int = 256) -> None:
        super().__init__()
        self.conv_shape = nn.Sequential(
            nn.Conv2d(head_channels, head_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
        )
        self.anchor = nn.Conv2d(head_channels, 4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(apn_channels, head_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(head_channels, head_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(head_channels, head_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
        )
        self.adcat = AdaptiveConcat(head_channels)
        self._init_layers([self.conv3, self.conv5, self.conv6, self.conv_shape, self.anchor])

    @staticmethod
    def _init_layers(modules: list[nn.Module]) -> None:
        for module in modules:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    @staticmethod
    def xcorr_depthwise(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        batch, channel = kernel.size(0), kernel.size(1)
        x = x.view(1, batch * channel, x.size(2), x.size(3))
        kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch * channel)
        return out.view(batch, channel, out.size(2), out.size(3))

    def forward(self, search_feat: tuple[torch.Tensor, torch.Tensor], template_feat: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        res2 = self.conv3(self.xcorr_depthwise(search_feat[0], template_feat[0]))
        ress = self.xcorr_depthwise(self.conv5(search_feat[1]), self.conv6(template_feat[1]))
        ress = self.adcat(ress, res2)
        shape_pred = self.anchor(self.conv_shape(ress))
        return shape_pred, ress


class ClsAndLoc(nn.Module):
    def __init__(self, channels: int = 256, group_channels: int = 32) -> None:
        super().__init__()
        self.conv1 = self._conv_bn_relu(channels, channels)
        self.conv2 = self._conv_bn_relu(channels, channels)
        self.conv4 = self._conv_bn_relu(channels, channels)
        self.convloc = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(group_channels, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(group_channels, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(group_channels, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(group_channels, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 4, kernel_size=3, stride=1, padding=1),
        )
        self.convcls = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(group_channels, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(group_channels, channels),
            nn.ReLU(inplace=True),
        )
        self.channel = SelfChannelBranch(channels)
        self.point = SelfPointBranch(channels)
        self.cls1 = nn.Conv2d(channels, 2, kernel_size=3, stride=1, padding=1)
        self.cls2 = nn.Conv2d(channels, 2, kernel_size=3, stride=1, padding=1)
        self.cls3 = nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1)
        self.adcat = AdaptiveConcat(channels)
        APN._init_layers([self.convloc, self.convcls, self.cls1, self.conv1, self.conv2, self.cls2, self.cls3])

    @staticmethod
    def _conv_bn_relu(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def xcorr_depthwise(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        return APN.xcorr_depthwise(x, kernel)

    def forward(
        self,
        search_feat: tuple[torch.Tensor, torch.Tensor],
        template_feat: tuple[torch.Tensor, torch.Tensor],
        ress: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        res = self.xcorr_depthwise(self.conv1(search_feat[1]), self.conv2(template_feat[1]))
        point = self.point(res)
        channel = self.conv4(self.channel(point))
        res = self.adcat(channel, ress)
        cls = self.convcls(res)
        return self.cls1(cls), self.cls2(cls), self.cls3(cls), self.convloc(res)
