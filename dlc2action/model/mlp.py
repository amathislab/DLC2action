#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
from dlc2action.model.base_model import Model
import torch
from torch import nn
from typing import List, Union
from torch.nn import functional as F


class _MLPModule(nn.Module):
    def __init__(self, f_maps_list, input_dims, num_classes, dropout_rates=None):
        super(_MLPModule, self).__init__()
        input_dims = int(sum([s[0] for s in input_dims.values()]))
        if dropout_rates is None:
            dropout_rates = 0.5
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates for _ in range(input_dims)]
        input_f_maps = [input_dims] + f_maps_list
        output_f_maps = f_maps_list + [num_classes]
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(in_f_maps, out_f_maps, 1)
                for in_f_maps, out_f_maps in zip(input_f_maps, output_f_maps)
            ]
        )
        self.dropout = nn.ModuleList([nn.Dropout(r) for r in dropout_rates])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.dropout[i](x)
                x = F.relu(x)
        return x


class MLP(Model):
    """
    A Multi-Layer Perceptron
    """

    def __init__(
        self,
        f_maps_list,
        input_dims,
        num_classes,
        dropout_rates=None,
        state_dict_path=None,
        ssl_constructors=None,
        ssl_types=None,
        ssl_modules=None,
    ):
        self.params = {
            "f_maps_list": f_maps_list,
            "input_dims": input_dims,
            "num_classes": num_classes,
            "dropout_rates": dropout_rates,
        }
        super().__init__(ssl_constructors, ssl_modules, ssl_types, state_dict_path)

    def _feature_extractor(self) -> Union[torch.nn.Module, List]:
        return _MLPModule(**self.params)

    def _predictor(self) -> torch.nn.Module:
        return nn.Identity()

    def features_shape(self) -> torch.Size:
        return None
