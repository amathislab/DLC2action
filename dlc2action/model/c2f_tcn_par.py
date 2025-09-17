#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
# Incorporates code adapted from C2F-TCN by dipika-singhania
# Original work Copyright (c) 2021 dipika-singhania
# Source: https://github.com/dipika-singhania/C2F-TCN
# Originally licensed under MIT License
# Combined work licensed under GNU AGPLv3
#
from functools import partial
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from dlc2action.model.base_model import Model

nonlinearity = partial(F.relu, inplace=True)


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Forward pass."""
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        """Forward pass."""
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        """Forward pass."""
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(nn.MaxPool1d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        """Forward pass."""
        x = self.max_pool_conv(x)
        return x


class up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )

        self.conv = double_conv(in_channels, out_channels)

    def forward(self, x1, x2):
        """Forward pass."""
        x1 = self.up(x1)
        # input is CHW
        diff = torch.tensor([x2.size()[2] - x1.size()[2]])

        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class TPPblock(nn.Module):
    def __init__(self, in_channels):
        super(TPPblock, self).__init__()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.pool4 = nn.MaxPool1d(kernel_size=6, stride=6)

        self.conv = nn.Conv1d(
            in_channels=in_channels, out_channels=1, kernel_size=1, padding=0
        )

    def forward(self, x):
        """Forward pass."""
        self.in_channels, t = x.size(1), x.size(2)
        self.layer1 = F.upsample(
            self.conv(self.pool1(x)), size=t, mode="linear", align_corners=True
        )
        self.layer2 = F.upsample(
            self.conv(self.pool2(x)), size=t, mode="linear", align_corners=True
        )
        self.layer3 = F.upsample(
            self.conv(self.pool3(x)), size=t, mode="linear", align_corners=True
        )
        self.layer4 = F.upsample(
            self.conv(self.pool4(x)), size=t, mode="linear", align_corners=True
        )

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


class Predictor(nn.Module):
    def __init__(self, dim, num_classes):
        super(Predictor, self).__init__()
        self.num_classes = num_classes
        self.conv_out_1 = nn.Conv1d(dim, dim, kernel_size=1)
        self.conv_out_2 = nn.Conv1d(dim, num_classes, kernel_size=1)

    def forward(self, x):
        """Forward pass."""
        x = self.conv_out_1(x)
        x = F.relu(x)
        x = self.conv_out_2(x)
        x = x.reshape((4, -1, self.num_classes, x.shape[-1]))
        return x


class C2F_TCN_P_Module(nn.Module):
    def __init__(self, n_channels, output_dim, num_f_maps):
        super().__init__()
        self.c2f_tcn = C2F_TCN_Module(
            n_channels, output_dim, num_f_maps, use_predictor=True
        )

    def forward(self, x):
        """Forward pass."""
        output = []
        for ind_x in x:
            output.append(self.c2f_tcn(ind_x))
        return torch.cat(output, dim=1)


class C2F_TCN_Module(nn.Module):
    """
    Features are extracted at the last layer of decoder.
    """

    def __init__(self, n_channels, output_dim, num_f_maps, use_predictor=False):
        super().__init__()
        self.use_predictor = use_predictor
        self.inc = inconv(n_channels, num_f_maps * 2)
        self.down1 = down(num_f_maps * 2, num_f_maps * 2)
        self.down2 = down(num_f_maps * 2, num_f_maps * 2)
        self.down3 = down(num_f_maps * 2, num_f_maps)
        self.down4 = down(num_f_maps, num_f_maps)
        self.down5 = down(num_f_maps, num_f_maps)
        self.down6 = down(num_f_maps, num_f_maps)
        self.up = up(num_f_maps * 2 + 4, num_f_maps)
        self.outcc0 = outconv(num_f_maps, output_dim)
        self.up0 = up(num_f_maps * 2, num_f_maps)
        self.outcc1 = outconv(num_f_maps, output_dim)
        self.up1 = up(num_f_maps * 2, num_f_maps)
        self.outcc2 = outconv(num_f_maps, output_dim)
        self.up2 = up(num_f_maps * 3, num_f_maps)
        self.outcc3 = outconv(num_f_maps, output_dim)
        self.up3 = up(num_f_maps * 3, num_f_maps)
        self.outcc4 = outconv(num_f_maps, output_dim)
        self.up4 = up(num_f_maps * 3, num_f_maps)
        self.outcc = outconv(num_f_maps, output_dim)
        self.tpp = TPPblock(num_f_maps)
        self.weights = torch.nn.Parameter(torch.ones(6))

    def forward(self, x):
        """Forward pass."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        # x7 = self.dac(x7)
        x7 = self.tpp(x7)
        x = self.up(x7, x6)
        y1 = self.outcc0(F.relu(x))
        # print("y1.shape=", y1.shape)
        x = self.up0(x, x5)
        y2 = self.outcc1(F.relu(x))
        # print("y2.shape=", y2.shape)
        x = self.up1(x, x4)
        y3 = self.outcc2(F.relu(x))
        # print("y3.shape=", y3.shape)
        x = self.up2(x, x3)
        y4 = self.outcc3(F.relu(x))
        # print("y4.shape=", y4.shape)
        x = self.up3(x, x2)
        y5 = self.outcc4(F.relu(x))
        # print("y5.shape=", y5.shape)
        x = self.up4(x, x1)
        y = self.outcc(x)
        # print("y.shape=", y.shape)
        output = [y]
        for outp_ele in [y5, y4, y3]:
            output.append(
                F.upsample(
                    outp_ele, size=y.shape[-1], mode="linear", align_corners=True
                )
            )
        output = torch.stack(output, dim=0)
        if self.use_predictor:
            K, B, C, T = output.shape
            output = output.reshape((-1, C, T))
        return output


class C2F_TCN_P(Model):
    def __init__(
        self,
        num_classes,
        input_dims,
        num_f_maps=128,
        feature_dim=None,
        state_dict_path=None,
        ssl_constructors=None,
        ssl_types=None,
        ssl_modules=None,
    ):
        if feature_dim is None:
            feature_dim = num_f_maps
        keys = [
            key
            for key in input_dims.keys()
            if len(key.split("---")) != 1 and len(key.split("---")[-1].split("+")) != 2
        ]
        num_ind = len(set([key.split("---")[-1] for key in keys]))
        key = keys[0]
        ind = key.split("---")[-1]
        input_dims = int(
            sum([v[0] for k, v in input_dims.items() if k.split("---")[-1] == ind])
        )
        self.f_shape = torch.Size([feature_dim * num_ind])
        self.params_predictor = {
            "dim": int(feature_dim * num_ind),
            "num_classes": num_classes,
        }
        self.params = {
            "output_dim": int(feature_dim),
            "n_channels": int(input_dims),
            "num_f_maps": int(num_f_maps),
        }
        super().__init__(ssl_constructors, ssl_modules, ssl_types, state_dict_path)

    def _feature_extractor(self) -> Union[torch.nn.Module, List]:
        return C2F_TCN_P_Module(**self.params)

    def _predictor(self) -> torch.nn.Module:
        return Predictor(**self.params_predictor)

    def features_shape(self) -> Optional[torch.Size]:
        return self.f_shape
