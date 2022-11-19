#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
#
# Adapted from ASRF by yiskw713
# Adapted from https://github.com/yiskw713/asrf/blob/main/libs/models/tcn.py
# Licensed under MIT License
#
""" EDTCN

Adapted from https://github.com/yiskw713/asrf/blob/main/libs/models/tcn.py
"""

import torch
from torch import nn
from typing import Tuple, Any
from torch.nn import functional as F
from dlc2action.model.base_model import Model
from typing import Union, List


class _NormalizedReLU(nn.Module):
    """
    Normalized ReLU Activation prposed in the original TCN paper.
    the values are divided by the max computed per frame
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        x = x / (x.max(dim=1, keepdim=True)[0] + self.eps)

        return x


class _EDTCNModule(nn.Module):
    """
    Encoder Decoder Temporal Convolutional Network
    """

    def __init__(
        self,
        in_channel: int,
        output_dim: int,
        kernel_size: int = 25,
        mid_channels: Tuple[int, int] = [128, 160],
        **kwargs: Any
    ) -> None:
        """
        Args:
            in_channel: int. the number of the channels of input feature
            output_dim: int. output classes
            kernel_size: int. 25 is proposed in the original paper
            mid_channels: list. the list of the number of the channels of the middle layer.
                        [96 + 32*1, 96 + 32*2] is proposed in the original paper
        Note that this implementation only supports n_layer=2
        """
        super().__init__()

        # encoder
        self.enc1 = nn.Conv1d(
            in_channel,
            mid_channels[0],
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout1 = nn.Dropout(0.3)
        self.relu1 = _NormalizedReLU()

        self.enc2 = nn.Conv1d(
            mid_channels[0],
            mid_channels[1],
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout2 = nn.Dropout(0.3)
        self.relu2 = _NormalizedReLU()

        # decoder
        self.dec1 = nn.Conv1d(
            mid_channels[1],
            mid_channels[1],
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout3 = nn.Dropout(0.3)
        self.relu3 = _NormalizedReLU()

        self.dec2 = nn.Conv1d(
            mid_channels[1],
            mid_channels[0],
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout4 = nn.Dropout(0.3)
        self.relu4 = _NormalizedReLU()

        self.conv_out = nn.Conv1d(mid_channels[0], output_dim, 1, bias=True)

        self.init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder 1
        x1 = self.relu1(self.dropout1(self.enc1(x)))
        t1 = x1.shape[2]
        x1 = F.max_pool1d(x1, 2)

        # encoder 2
        x2 = self.relu2(self.dropout2(self.enc2(x1)))
        t2 = x2.shape[2]
        x2 = F.max_pool1d(x2, 2)

        # decoder 1
        x3 = F.interpolate(x2, size=(t2,), mode="nearest")
        x3 = self.relu3(self.dropout3(self.dec1(x3)))

        # decoder 2
        x4 = F.interpolate(x3, size=(t1,), mode="nearest")
        x4 = self.relu4(self.dropout4(self.dec2(x4)))

        out = self.conv_out(x4)

        return out

    def init_weight(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


class _Predictor(nn.Module):
    def __init__(self, dim, num_classes):
        super(_Predictor, self).__init__()
        self.num_classes = num_classes
        self.conv_out_1 = nn.Conv1d(dim, dim, kernel_size=1)
        self.conv_out_2 = nn.Conv1d(dim, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv_out_1(x)
        x = F.relu(x)
        x = self.conv_out_2(x)
        return x


class EDTCN(Model):
    """
    An implementation of EDTCN (Endoder-Decoder TCN)
    """

    def __init__(
        self,
        num_classes,
        input_dims,
        kernel_size,
        mid_channels,
        feature_dim=None,
        state_dict_path=None,
        ssl_constructors=None,
        ssl_types=None,
        ssl_modules=None,
    ):
        input_dims = int(sum([s[0] for s in input_dims.values()]))
        if feature_dim is None:
            feature_dim = num_classes
            self.params_predictor = None
            self.f_shape = None
        else:
            self.params_predictor = {
                "dim": int(feature_dim),
                "num_classes": int(num_classes),
            }
            self.f_shape = torch.Size([int(feature_dim)])
        self.params = {
            "output_dim": int(feature_dim),
            "in_channel": input_dims,
            "kernel_size": int(kernel_size),
            "mid_channels": mid_channels,
        }
        super().__init__(ssl_constructors, ssl_modules, ssl_types, state_dict_path)

    def _feature_extractor(self) -> Union[torch.nn.Module, List]:
        return _EDTCNModule(**self.params)

    def _predictor(self) -> torch.nn.Module:
        if self.params_predictor is None:
            return nn.Identity()
        else:
            return _Predictor(**self.params_predictor)

    def features_shape(self) -> torch.Size:
        return self.f_shape
