#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
"""
Implementations of `dlc2action.ssl.base_ssl.SSLConstructor` of the `'contrastive'` type
"""

from typing import Dict, Tuple, Union
from dlc2action.ssl.base_ssl import SSLConstructor
from dlc2action.loss.contrastive import _NTXent, _CircleLoss, _TripletLoss
from dlc2action.loss.contrastive_frame import _ContrastiveRegressionLoss
from dlc2action.ssl.modules import _FeatureExtractorTCN, _MFeatureExtractorTCN, _FC
import torch
from torch import nn
from copy import deepcopy


class ContrastiveSSL(SSLConstructor):
    """
    A contrastive SSL class with an NT-Xent loss

    The SSL input and target are left empty (the SSL input is generated as an augmentation of the
    input sample at runtime).
    """

    type = "contrastive"

    def __init__(
        self,
        num_f_maps: torch.Size,
        len_segment: int,
        ssl_features: int = 128,
        tau: float = 1,
    ) -> None:
        """
        Parameters
        ----------
        num_f_maps : torch.Size
            shape of feature extractor output
        len_segment : int
            length of segment in the base feature extractor output
        ssl_features : int, default 128
            the final number of features per clip
        tau : float, default 1
            the tau parameter of NT-Xent loss
        """

        super().__init__()
        self.loss_function = _NTXent(tau)
        if len(num_f_maps) > 1:
            raise RuntimeError(
                "The ContrastiveSSL constructor expects the input data to be 2-dimensional; "
                f"got {len(num_f_maps) + 1} dimensions"
            )
        num_f_maps = int(num_f_maps[0])
        self.pars = {
            "num_f_maps": num_f_maps,
            "len_segment": len_segment,
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
        NT-Xent loss
        """

        features1, features2 = predicted
        loss = self.loss_function(features1, features2)
        return loss

    def construct_module(self) -> Union[nn.Module, None]:
        """
        Clip-wise feature TCN extractor
        """

        module = _FeatureExtractorTCN(**self.pars)
        return module


class ContrastiveMaskedSSL(SSLConstructor):
    """
    A contrastive masked SSL class with an NT-Xent loss

    A few frames in the middle of each segment are masked and then the output of the second layer of
    feature extraction for the segment is used to predict the output of the first layer for the missing frames.
    The SSL input and target are left empty (the SSL input is generated as an augmentation of the
    input sample at runtime).
    """

    type = "contrastive_2layers"

    def __init__(
        self,
        num_f_maps: torch.Size,
        len_segment: int,
        ssl_features: int = 128,
        tau: float = 1,
        num_masked: int = 10,
    ) -> None:
        """
        Parameters
        ----------
        num_f_maps : torch.Size
            shape of feature extractor output
        len_segment : int
            length of segment in the base feature extractor output
        ssl_features : int, default 128
            the final number of features per clip
        tau : float, default 1
            the tau parameter of NT-Xent loss
        """

        super().__init__()
        self.start = int(len_segment // 2 - num_masked // 2)
        self.end = int(len_segment // 2 + num_masked // 2)
        self.loss_function = _NTXent(tau)
        if len(num_f_maps) > 1:
            raise RuntimeError(
                "The ContrastiveMaskedSSL constructor expects the input data to be 2-dimensional; "
                f"got {len(num_f_maps) + 1} dimensions"
            )
        num_f_maps = int(num_f_maps[0])
        self.pars = {
            "num_f_maps": num_f_maps,
            "len_segment": len_segment,
            "output_dim": ssl_features,
            "kernel_1": 3,
            "kernel_2": 3,
            "stride": 1,
            "start": self.start,
            "end": self.end,
        }

    def transformation(self, sample_data: Dict) -> Tuple:
        """
        Empty transformation
        """

        data = deepcopy(sample_data)
        for key in data.keys():
            data[key][:, self.start : self.end] = 0
        return data, torch.tensor(float("nan"))
        # return torch.tensor(float("nan")), torch.tensor(float("nan"))

    def loss(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """
        NT-Xent loss
        """

        features, ssl_features = predicted
        loss = self.loss_function(features, ssl_features)
        return loss

    def construct_module(self) -> Union[nn.Module, None]:
        """
        Clip-wise feature TCN extractor
        """

        module = _MFeatureExtractorTCN(**self.pars)
        return module


class PairwiseSSL(SSLConstructor):
    """
    A pairwise SSL class with triplet or circle loss

    The SSL input and target are left empty (the SSL input is generated as an augmentation of the
    input sample at runtime).
    """

    type = "contrastive"

    def __init__(
        self,
        num_f_maps: torch.Size,
        len_segment: int,
        ssl_features: int = 128,
        margin: float = 0,
        distance: str = "cosine",
        loss: str = "triplet",
        gamma: float = 1,
    ) -> None:
        """
        Parameters
        ----------
        num_f_maps : torch.Size
            shape of feature extractor output
        len_segment : int
            length of segment in feature extractor output
        ssl_features : int, default 128
            final number of features per clip
        margin : float, default 0
            the margin parameter of triplet or circle loss
        distance : {'cosine', 'euclidean'}
            the distance calculation method for triplet or circle loss
        loss : {'triplet', 'circle'}
            the loss function name
        gamma : float, default 1
            the gamma parameter of circle loss
        """

        super().__init__()
        if loss == "triplet":
            self.loss_function = _TripletLoss(margin=margin, distance=distance)
        elif loss == "circle":
            self.loss_function = _CircleLoss(
                margin=margin, gamma=gamma, distance=distance
            )
        else:
            raise ValueError(
                f'The {loss} loss is unavailable, please choose from "triplet" and "circle"'
            )
        if len(num_f_maps) > 1:
            raise RuntimeError(
                "The PairwiseSSL constructor expects the input data to be 2-dimensional; "
                f"got {len(num_f_maps) + 1} dimensions"
            )
        num_f_maps = int(num_f_maps[0])
        self.pars = {
            "num_f_maps": num_f_maps,
            "len_segment": len_segment,
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
        Triplet or circle loss
        """

        features1, features2 = predicted
        loss = self.loss_function(features1, features2)
        return loss

    def construct_module(self) -> Union[nn.Module, None]:
        """
        Clip-wise feature TCN extractor
        """

        module = _FeatureExtractorTCN(**self.pars)
        return module


class PairwiseMaskedSSL(PairwiseSSL):

    type = "contrastive_2layers"

    def __init__(
        self,
        num_f_maps: torch.Size,
        len_segment: int,
        ssl_features: int = 128,
        margin: float = 0,
        distance: str = "cosine",
        loss: str = "triplet",
        gamma: float = 1,
        num_masked: int = 10,
    ) -> None:
        super().__init__(
            num_f_maps, len_segment, ssl_features, margin, distance, loss, gamma
        )
        self.num_masked = num_masked
        self.start = int(len_segment // 2 - num_masked // 2)
        self.end = int(len_segment // 2 + num_masked // 2)
        if len(num_f_maps) > 1:
            raise RuntimeError(
                "The PairwiseMaskedSSL constructor expects the input data to be 2-dimensional; "
                f"got {len(num_f_maps) + 1} dimensions"
            )
        num_f_maps = int(num_f_maps[0])
        self.pars = {
            "num_f_maps": num_f_maps,
            "len_segment": len_segment,
            "output_dim": ssl_features,
            "kernel_1": 3,
            "kernel_2": 3,
            "stride": 1,
            "start": self.start,
            "end": self.end,
        }

    def transformation(self, sample_data: Dict) -> Tuple:
        """
        Empty transformation
        """

        data = deepcopy(sample_data)
        for key in data.keys():
            data[key][:, self.start : self.end] = 0
        return data, torch.tensor(float("nan"))

    def construct_module(self) -> Union[nn.Module, None]:
        """
        Clip-wise feature TCN extractor
        """

        module = _MFeatureExtractorTCN(**self.pars)
        return module


class ContrastiveRegressionSSL(SSLConstructor):

    type = "contrastive"

    def __init__(
        self,
        num_f_maps: torch.Size,
        num_features: int = 128,
        num_ssl_layers: int = 1,
        distance: str = "cosine",
        temperature: float = 1,
        break_factor: int = None,
    ) -> None:
        if len(num_f_maps) > 1:
            raise RuntimeError(
                "The ContrastiveRegressionSSL constructor expects the input data to be 2-dimensional; "
                f"got {len(num_f_maps) + 1} dimensions"
            )
        num_f_maps = int(num_f_maps[0])
        self.loss_function = _ContrastiveRegressionLoss(
            temperature, distance, break_factor
        )
        self.pars = {
            "num_f_maps": num_f_maps,
            "num_ssl_layers": num_ssl_layers,
            "num_ssl_f_maps": num_features,
            "dim": num_features,
        }
        super().__init__()

    def loss(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """
        NT-Xent loss
        """

        features1, features2 = predicted
        loss = self.loss_function(features1, features2)
        return loss

    def transformation(self, sample_data: Dict) -> Tuple:
        """
        Empty transformation
        """

        return torch.tensor(float("nan")), torch.tensor(float("nan"))

    def construct_module(self) -> Union[nn.Module, None]:
        """
        Clip-wise feature TCN extractor
        """

        return _FC(**self.pars)
