#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
from typing import Dict, Tuple, Union, List
import torch
from dlc2action.ssl.base_ssl import SSLConstructor
from abc import ABC, abstractmethod
from dlc2action.ssl.modules import _FeatureExtractorTCN
import torch
from torch import nn
from copy import deepcopy
from itertools import permutations

from torch.nn import CrossEntropyLoss, Linear, BCEWithLogitsLoss


class ReverseSSL(SSLConstructor, ABC):
    """
    A flip detection SSL

    Reverse some of the segments and predict the flip with a binary classifier.
    """

    type = "ssl_input"

    def __init__(self, num_f_maps: torch.Size, len_segment: int) -> None:
        """
        Parameters
        ----------
        num_f_maps : torch.Size
            the number of input feature maps
        len_segment : int
            the length of the input segments
        """

        super().__init__()
        self.ce = BCEWithLogitsLoss()
        if len(num_f_maps) > 1:
            raise RuntimeError(
                "The ContrastiveSSL constructor expects the input data to be 2-dimensional; "
                f"got {len(num_f_maps) + 1} dimensions"
            )
        num_f_maps = int(num_f_maps[0])
        self.pars = {
            "num_f_maps": num_f_maps,
            "len_segment": len_segment,
            "output_dim": 1,
            "kernel_1": 5,
            "kernel_2": 5,
            "stride": 2,
            "decrease_f_maps": True,
        }

    def transformation(self, sample_data: Dict) -> Tuple:
        """
        Do the flip
        """

        ssl_target = torch.randint(2, (1,), dtype=torch.float)
        ssl_input = deepcopy(sample_data)
        if ssl_target == 1:
            for key, value in sample_data.items():
                ssl_input[key] = value.flip(-1)
        return ssl_input, {"order": ssl_target}

    def loss(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """
        Cross-entropy loss
        """

        loss = self.ce(predicted, target.squeeze())
        return loss

    def construct_module(self) -> nn.Module:
        """
        Construct the SSL prediction module using the parameters specified at initialization
        """

        module = _FeatureExtractorTCN(**self.pars)
        return module


class OrderSSL(SSLConstructor, ABC):
    """
    An order prediction SSL

    Cut out segments from the features, permute them and predict the order.
    """

    type = "ssl_target"

    def __init__(
        self,
        num_f_maps: torch.Size,
        len_segment: int,
        num_segments: int = 3,
        ssl_features: int = 32,
        skip_frames: int = 10,
    ) -> None:
        """
        Parameters
        ----------
        num_f_maps : torch.Size
            the number of the input feature maps
        len_segment : int
            the length of the input segments
        num_segments : int, default 3
            the number of segments to permute
        ssl_features : int, default 32
            the number of features per permuted segment
        skip_frames : int, default 10
            the number of frames to cut from each permuted segment
        """

        super().__init__()
        self.ce = CrossEntropyLoss(ignore_index=-100)
        if len(num_f_maps) > 1:
            raise RuntimeError(
                "The ContrastiveSSL constructor expects the input data to be 2-dimensional; "
                f"got {len(num_f_maps) + 1} dimensions"
            )
        num_f_maps = int(num_f_maps[0])
        self.orders = [list(x) for x in permutations(range(num_segments), num_segments)]
        self.len_segment = len_segment // num_segments
        self.num_segments = num_segments
        self.skip_frames = skip_frames
        self.pars = {
            "num_f_maps": num_f_maps,
            "len_segment": len_segment // num_segments,
            "output_dim": ssl_features,
            "kernel_1": 5,
            "kernel_2": 5,
            "stride": 2,
            "decrease_f_maps": True,
        }

    def transformation(self, sample_data: Dict) -> Tuple:
        """
        Empty transformation
        """

        return torch.tensor(float("nan")), torch.tensor(float("nan"))

    def loss(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """
        Cross-entropy loss
        """

        predicted, target = predicted
        loss = self.ce(predicted, target)
        return loss

    def construct_module(self) -> nn.Module:
        """
        Construct the SSL prediction module using the parameters specified at initialization
        """

        class Classifier(nn.Module):
            def __init__(self, num_segments, num_classes, skip_frames, **pars):
                super().__init__()
                self.len_segment = pars["len_segment"]
                pars["len_segment"] -= skip_frames
                self.extractor = _FeatureExtractorTCN(**pars)
                self.num_segments = num_segments
                self.skip_frames = skip_frames
                self.fc = Linear(pars["output_dim"] * self.num_segments, num_classes)
                self.orders = torch.tensor(
                    [list(x) for x in permutations(range(num_segments), num_segments)]
                )

            def forward(self, x):
                target = torch.randint(len(self.orders), (x.shape[0],)).to(x.device)
                order = self.orders[target]
                x = x[:, :, : self.num_segments * self.len_segment]
                B, F, L = x.shape
                x = x.reshape((B, F, -1, self.len_segment))
                x = x[:, :, :, : -self.skip_frames]
                x = x[
                    torch.arange(x.shape[0]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                    torch.arange(x.shape[1]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0),
                    order.unsqueeze(1).unsqueeze(-1),
                    torch.arange(x.shape[-1]).unsqueeze(0).unsqueeze(0).unsqueeze(0),
                ]
                x = x.transpose(1, 2).reshape(
                    (-1, F, self.len_segment - self.skip_frames)
                )
                x = self.extractor(x).reshape((B, -1))
                x = self.fc(x)
                return (x, target)

        module = Classifier(
            self.num_segments,
            len(self.orders),
            skip_frames=self.skip_frames,
            **self.pars,
        )
        return module
