#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
"""
Network modules used by implementations of `dlc2action.ssl.base_ssl.SSLConstructor`
"""

import torch
from torch import nn
import copy
from math import floor
import torch.nn.functional as F
from torch.nn import Linear


class _FeatureExtractorTCN(nn.Module):
    """
    A module that extracts clip-level features with a TCN
    """

    def __init__(
        self,
        num_f_maps: int,
        output_dim: int,
        len_segment: int,
        kernel_1: int,
        kernel_2: int,
        stride: int,
        decrease_f_maps: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        num_f_maps : int
            number of features in input
        output_dim : int
            number of features in output
        len_segment : int
            length of segment in input
        kernel_1 : int
            kernel size of the first layer
        kernel_2 : int
            kernel size of the second layer
        stride : int
            stride
        decrease_f_maps : bool, default False
            if `True`, number of feature maps is halved at each new layer
        """

        super().__init__()
        num_f_maps = int(num_f_maps)
        output_dim = int(output_dim)
        if decrease_f_maps:
            f_maps_2 = max(num_f_maps // 2, 1)
            f_maps_3 = max(num_f_maps // 4, 1)
        else:
            f_maps_2 = f_maps_3 = num_f_maps
        length = int(floor((len_segment - kernel_1) / stride + 1))
        length = floor((length - kernel_2) / stride + 1)
        features = length * f_maps_3
        self.conv = nn.ModuleList()
        self.conv.append(
            nn.Conv1d(num_f_maps, f_maps_2, kernel_1, padding=0, stride=stride)
        )
        self.conv.append(
            nn.Conv1d(f_maps_2, f_maps_3, kernel_2, padding=0, stride=stride)
        )
        self.conv_1x1_out = nn.Conv1d(features, output_dim, 1)
        self.dropout = nn.Dropout()

    def forward(self, f):
        for conv in self.conv:
            f = conv(f)
            f = F.relu(f)
            f = self.dropout(f)
        f = f.reshape((f.shape[0], -1, 1))
        f = self.conv_1x1_out(f).squeeze()
        return f


class _MFeatureExtractorTCN(nn.Module):
    """
    A module that extracts segment-level features with a TCN
    """

    def __init__(
        self,
        num_f_maps: int,
        output_dim: int,
        len_segment: int,
        kernel_1: int,
        kernel_2: int,
        stride: int,
        start: int,
        end: int,
        num_layers: int = 3,
    ):
        """
        Parameters
        ----------
        num_f_maps : int
            number of features in input
        output_dim : int
            number of features in output
        len_segment : int
            length of segment in input
        kernel_1 : int
            kernel size of the first layer
        kernel_2 : int
            kernel size of the second layer
        stride : int
            stride
        start : int
            the start index of the segment to extract
        end : int
            the end index of the segment to extract
        num_layers : int
            number of layers
        """

        super(_MFeatureExtractorTCN, self).__init__()
        self.main_module = _DilatedTCN(num_layers, num_f_maps, num_f_maps)
        self.extractor = _FeatureExtractorTCN(
            num_f_maps, output_dim, len_segment, kernel_1, kernel_2, stride
        )
        in_features = int(len_segment * num_f_maps)
        out_features = int((end - start) * num_f_maps)
        self.linear = Linear(in_features=in_features, out_features=out_features)
        self.start = int(start)
        self.end = int(end)

    def forward(self, f, extract_features=True):
        if extract_features:
            f = self.main_module(f)
            f = F.relu(f)
            f = f.reshape((f.shape[0], -1))
            f = self.linear(f)
        else:
            f = f[:, :, self.start : self.end]
            f = f.reshape((f.shape[0], -1))
        return f


class _FC(nn.Module):
    """
    Fully connected module that predicts input data given features
    """

    def __init__(
        self, dim: int, num_f_maps: int, num_ssl_layers: int, num_ssl_f_maps: int
    ) -> None:
        """
        Parameters
        ----------
        dim : int
            output number of features
        num_f_maps : int
            number of features in input
        num_ssl_layers : int
            number of layers in the module
        num_ssl_f_maps : int
            number of feature maps in the module
        """

        super().__init__()
        dim = int(dim)
        num_f_maps = int(num_f_maps)
        num_ssl_layers = int(num_ssl_layers)
        num_ssl_f_maps = int(num_ssl_f_maps)
        self.layers = nn.ModuleList(
            [nn.Linear(in_features=num_f_maps, out_features=num_ssl_f_maps)]
        )
        for _ in range(num_ssl_layers - 2):
            self.layers.append(
                nn.Linear(in_features=num_ssl_f_maps, out_features=num_ssl_f_maps)
            )
        self.layers.append(nn.Linear(in_features=num_ssl_f_maps, out_features=dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, F = x.shape
        x = x.transpose(1, 2).reshape(-1, C)
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(N, F, -1).transpose(1, 2)
        return x


class _DilatedTCN(nn.Module):
    """
    TCN module that predicts input data given features
    """

    def __init__(self, num_layers, input_dim, output_dim):
        """
        Parameters
        ----------
        output_dim : int
            output number of features
        input_dim : int
            number of features in input
        num_layers : int
            number of layers in the module
        """

        super().__init__()
        num_layers = int(num_layers)
        input_dim = int(input_dim)
        output_dim = int(output_dim)
        self.num_layers = num_layers
        self.conv_dilated_1 = nn.ModuleList(
            (
                nn.Conv1d(
                    input_dim,
                    input_dim,
                    3,
                    padding=2 ** (num_layers - 1 - i),
                    dilation=2 ** (num_layers - 1 - i),
                )
                for i in range(num_layers)
            )
        )

        self.conv_dilated_2 = nn.ModuleList(
            (
                nn.Conv1d(input_dim, input_dim, 3, padding=2**i, dilation=2**i)
                for i in range(num_layers)
            )
        )

        self.conv_fusion = nn.ModuleList(
            (nn.Conv1d(2 * input_dim, input_dim, 1) for i in range(num_layers))
        )

        self.conv_1x1_out = nn.Conv1d(input_dim, output_dim, 1)

        self.dropout = nn.Dropout()

    def forward(self, f):
        for i in range(self.num_layers):
            f_in = copy.copy(f)
            f = self.conv_fusion[i](
                torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1)
            )
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in
        f = self.conv_1x1_out(f)
        return f
