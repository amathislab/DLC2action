#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
from typing import List, Union

import torch
from dlc2action.model.base_model import Model
from dlc2action.model.ms_tcn_modules import MSRefinement
from torch import nn
from torch.nn import functional as F


class ResLayer3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)
        )
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.bn3 = nn.BatchNorm3d(out_channels)
        if in_channels != out_channels:
            self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1))
        else:
            self.conv3 = None

    def forward(self, x):
        f = self.conv1(x)
        f = F.relu(self.bn1(f))
        f = self.conv2(f)
        f = self.bn2(f)
        if self.conv3:
            x = self.bn3(self.conv3(x))
        return F.relu(f + x)


class C3D(nn.Module):
    def __init__(self, dim, loaded_dim):
        super(C3D, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(ResLayer3D(dim, 32))
        self.layers.append(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))

        self.layers.append(ResLayer3D(32, 64))
        self.layers.append(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))

        self.layers.append(ResLayer3D(64, 128))
        self.layers.append(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
        self.layers.append(ResLayer3D(128, 128))
        self.layers.append(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))

        self.layers.append(ResLayer3D(128, 128))
        self.layers.append(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))

        if loaded_dim is not None:
            self.conv1 = nn.Conv1d(loaded_dim, 16, 3)
            self.conv2 = nn.Conv1d(16, 16, 3)
            self.conv3 = nn.Conv1d(128 + 16, 128, 3)

    def forward(self, x):
        loaded = None
        if isinstance(x, list):
            x, loaded = x
        for layer in self.layers:
            x = layer(x)
        x = torch.reshape(x, (*x.shape[:-2], -1))
        x = torch.mean(x, dim=-1)
        if loaded is not None:
            loaded = F.relu(self.conv1(loaded))
            loaded = F.relu(self.conv2(loaded))
            x = F.relu(self.conv3(torch.cat([x, loaded], dim=1)))
        return x


class Predictor(nn.Module):
    def __init__(self, dim, num_classes):
        super(Predictor, self).__init__()
        self.conv_out_1 = nn.Conv1d(dim, 64, kernel_size=1)
        self.conv_out_2 = nn.Conv1d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv_out_1(x)
        x = F.relu(x)
        x = self.conv_out_2(x)
        return x


class C3D_A(Model):
    def __init__(
        self,
        dims,
        num_classes,
        state_dict_path=None,
        ssl_constructors=None,
        ssl_types=None,
        ssl_modules=None,
    ):
        dim = sum([x[0] for k, x in dims.items() if k != "loaded"])
        if "loaded" in dims:
            loaded_dim = dims["loaded"][0]
        else:
            loaded_dim = None
        self.pars1 = {"dim": dim, "loaded_dim": loaded_dim}
        output_dims = C3D(**self.pars1)(
            torch.ones((1, dim, *list(dims.values())[0][1:]))
        ).shape
        self.num_f_maps = output_dims[1]
        self.pars2 = {"dim": self.num_f_maps, "num_classes": num_classes}
        super().__init__(ssl_constructors, ssl_modules, ssl_types, state_dict_path)

    def _feature_extractor(self) -> Union[torch.nn.Module, List]:
        return C3D(**self.pars1)

    def _predictor(self) -> torch.nn.Module:
        return Predictor(**self.pars2)

    def features_shape(self) -> torch.Size:
        return torch.Size([128])


class C3D_A_MS(Model):
    def __init__(
        self,
        dims,
        num_classes,
        num_layers_R,
        num_R,
        num_f_maps_R,
        dropout_rate,
        direction,
        skip_connections,
        exclusive,
        attention_R="none",
        block_size_R=0,
        num_heads=1,
        state_dict_path=None,
        ssl_constructors=None,
        ssl_types=None,
        ssl_modules=None,
    ):
        dim = sum([x[0] for k, x in dims.items() if k != "loaded"])
        if "loaded" in dims:
            loaded_dim = dims["loaded"][0]
        else:
            loaded_dim = None
        self.pars1 = {"dim": dim, "loaded_dim": loaded_dim}
        output_dims = C3D(**self.pars1)(
            torch.ones((1, dim, *list(dims.values())[0][1:]))
        ).shape
        self.num_f_maps = output_dims[1]
        self.pars_R = {
            "exclusive": exclusive,
            "num_layers_R": int(num_layers_R),
            "num_R": int(num_R),
            "num_f_maps_input": 128,
            "num_f_maps": int(num_f_maps_R),
            "num_classes": int(num_classes),
            "dropout_rate": dropout_rate,
            "skip_connections": skip_connections,
            "direction": direction,
            "block_size": int(block_size_R),
            "num_heads": int(num_heads),
            "attention": attention_R,
        }
        super().__init__(ssl_constructors, ssl_modules, ssl_types, state_dict_path)

    def _feature_extractor(self) -> Union[torch.nn.Module, List]:
        return C3D(**self.pars1)

    def _predictor(self) -> torch.nn.Module:
        return MSRefinement(**self.pars_R)

    def features_shape(self) -> torch.Size:
        return torch.Size([128])
