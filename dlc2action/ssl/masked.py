#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
"""Implementations of `dlc2action.ssl.base_ssl.SSLConstructor` that predict masked input features."""

from typing import Dict, Tuple, Union, List

import torch

from dlc2action.ssl.base_ssl import SSLConstructor
from abc import ABC, abstractmethod
from dlc2action.loss.mse import MSE
from dlc2action.ssl.modules import *


class MaskedFeaturesSSL(SSLConstructor, ABC):
    """A base masked features SSL class.

    Mask some of the input features randomly and predict the initial data.
    """

    type = "ssl_input"

    def __init__(self, frac_masked: float = 0.2) -> None:
        """Initialize the constructor.

        Parameters
        ----------
        frac_masked : float
            fraction of features to real_lens

        """
        super().__init__()
        self.mse = MSE()
        self.frac_masked = frac_masked

    def transformation(self, sample_data: Dict) -> Tuple:
        """Mask some of the features randomly."""
        for key in sample_data:
            mask = torch.empty(sample_data[key].shape).normal_() > self.frac_masked
            sample_data[key] = sample_data[key] * mask
        ssl_target = torch.cat(list(sample_data.values()))
        return (sample_data, ssl_target)

    def loss(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """MSE loss."""
        loss = self.mse(predicted, target)
        return loss

    @abstractmethod
    def construct_module(self) -> nn.Module:
        """Construct the SSL prediction module using the parameters specified at initialization."""


class MaskedFeaturesSSL_FC(MaskedFeaturesSSL):
    """A fully connected masked features SSL class.

    Mask some of the input features randomly and predict the initial data.
    """

    type = "ssl_input"

    def __init__(
        self,
        dims: torch.Size,
        num_f_maps: torch.Size,
        frac_masked: float = 0.2,
        num_ssl_layers: int = 5,
        num_ssl_f_maps: int = 16,
    ) -> None:
        """Initialize the constructor.

        Parameters
        ----------
        dims : torch.Size
            the shape of features in model input
        num_f_maps : torch.Size
            shape of feature extraction output
        frac_masked : float, default 0.1
            fraction of features to real_lens
        num_ssl_layers : int, default 5
            number of layers in the SSL module
        num_ssl_f_maps : int, default 16
            number of feature maps in the SSL module

        """
        super().__init__(frac_masked)
        dim = int(sum([s[0] for s in dims.values()]))
        num_f_maps = int(num_f_maps[0])
        self.pars = {
            "dim": dim,
            "num_f_maps": num_f_maps,
            "num_ssl_layers": num_ssl_layers,
            "num_ssl_f_maps": num_ssl_f_maps,
        }

    def construct_module(self) -> Union[nn.Module, None]:
        """Construct a fully connected module."""
        module = FC(**self.pars)
        return module


class MaskedFeaturesSSL_TCN(MaskedFeaturesSSL):
    """A TCN masked features SSL class.

    Mask some of the input features randomly and predict the initial data.
    """

    def __init__(
        self,
        dims: Dict,
        num_f_maps: torch.Size,
        frac_masked: float = 0.2,
        num_ssl_layers: int = 5,
        num_ssl_f_maps:int = None,
    ) -> None:
        """Initialize the class.

        Parameters
        ----------
        dims : torch.Size
            the shape of features in model input
        num_f_maps : torch.Size
            shape of feature extraction output
        frac_masked : float, default 0.1
            fraction of features to real_lens
        num_ssl_layers : int, default 5
            number of layers in the SSL module

        """
        super().__init__(frac_masked)

        if not num_ssl_f_maps is None:
            print(f"num_ssl_f_maps is set to {num_ssl_f_maps} but is ignored for TCN")

        dim = int(sum([s[0] for s in dims.values()]))
        num_f_maps = int(num_f_maps[0])
        self.pars = {
            "input_dim": num_f_maps,
            "num_layers": num_ssl_layers,
            "output_dim": dim,
        }

    def construct_module(self) -> Union[nn.Module, None]:
        """Construct a TCN module."""
        module = DilatedTCN(**self.pars)
        return module


class MaskedKinematicSSL(SSLConstructor, ABC):
    """A base masked joints SSL class.

    Mask some of the joints randomly and predict the initial data.
    """

    type = "ssl_input"

    def __init__(self, frac_masked: float = 0.2) -> None:
        """Initialize the class.

        Parameters
        ----------
        frac_masked : float, default 0.1
            fraction of features to real_lens

        """
        super().__init__()
        self.mse = MSE()
        self.frac_masked = frac_masked

    def _get_keys(self, key_bases, x):
        """Get keys of x that start with one of the strings in key_bases."""
        keys = []
        for key in x:
            if key_bases.count(key) > 0:
                keys.append(key)
        return keys

    def transformation(self, sample_data: Dict) -> Tuple:
        """Mask joints randomly."""
        assert (
            "coords" in sample_data.keys() or "coord_diff" in sample_data.keys()
        ), "'coords' or 'coord_diff' features are required when using MaskedKinematicSSL"

        multi_dim_features = self._get_keys(
            (
                "coords",
                "coord_diff",
                "speed_direction",
            ),
            sample_data,
        )

        single_dim_features = self._get_keys(
            (
                "speed_joints",
                "acc_joints",
                "angle_joints_radian",
                "angle_speeds",
                "speed_value",
            ),
            sample_data,
        )

        assert (
            len(multi_dim_features) > 0
        ), "No multi-dimensional features found in sample_data"
        assert (
            len(single_dim_features) > 0
        ), "No single-dimensional features found in sample_data"

        features, frames = sample_data[multi_dim_features[0]].shape

        n_bp = features // 2
        masked_joints = torch.FloatTensor(n_bp).uniform_() > self.frac_masked

        keys = self._get_keys(("intra_distance", "inter_distance"), sample_data)
        for key in keys:
            mask = masked_joints.repeat(n_bp, frames, 1).transpose(1, 2)
            indices = torch.triu_indices(n_bp, n_bp, 1)

            X = torch.zeros((n_bp, n_bp, frames)).to(sample_data[key].device)
            X[indices[0], indices[1], :] = sample_data[key]
            X[mask] = 0
            X[mask.transpose(0, 1)] = 0
            sample_data[key] = X[indices[0], indices[1], :].reshape(-1, frames)

        for key in multi_dim_features + single_dim_features:
            mask = (
                masked_joints.repeat(sample_data[key].shape[0]//sample_data[single_dim_features[0]].shape[0], frames, 1)
                .transpose(0, 2)
                .reshape((-1, frames))
            )
            sample_data[key][mask] = 0

        return sample_data, torch.cat(list(sample_data.values()))

    def loss(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """MSE loss."""
        loss = self.mse(predicted, target)
        return loss

    @abstractmethod
    def construct_module(self) -> Union[nn.Module, None]:
        """Construct the SSL prediction module using the parameters specified at initialization."""


class MaskedKinematicSSL_FC(MaskedKinematicSSL):
    """Masked kinematic SSL class with fully connected module."""

    def __init__(
        self,
        dims: torch.Size,
        num_f_maps: torch.Size,
        frac_masked: float = 0.2,
        num_ssl_layers: int = 5,
        num_ssl_f_maps: int = 16,
    ) -> None:
        """Initialize the constructor.

        Parameters
        ----------
        dims : torch.Size
            the number of features in model input
        num_f_maps : torch.Size
            shape of feature extraction output
        frac_masked : float, default 0.1
            fraction of joints to real_lens
        num_ssl_layers : int, default 5
            number of layers in the SSL module
        num_ssl_f_maps : int, default 16
            number of feature maps in the SSL module

        """
        super().__init__(frac_masked)
        dim = int(sum([s[0] for s in dims.values()]))
        num_f_maps = int(num_f_maps[0])
        self.pars = {
            "dim": dim,
            "num_f_maps": num_f_maps,
            "num_ssl_layers": num_ssl_layers,
            "num_ssl_f_maps": num_ssl_f_maps,
        }

    def construct_module(self) -> Union[nn.Module, None]:
        """Construct a fully connected module."""
        module = FC(**self.pars)
        return module


class MaskedKinematicSSL_TCN(MaskedKinematicSSL):
    """Masked kinematic SSL using a TCN module."""

    def __init__(
        self,
        dims: torch.Size,
        num_f_maps: torch.Size,
        frac_masked: float = 0.2,
        num_ssl_layers: int = 5,
    ) -> None:
        """Initialise the constructor.

        Parameters
        ----------
        dims : torch.Size
            the shape of features in model input
        num_f_maps : torch.Size
            shape of feature extraction output
        frac_masked : float, default 0.1
            fraction of joints to real_lens
        num_ssl_layers : int, default 5
            number of layers in the SSL module

        """
        super().__init__(frac_masked)
        dim = int(sum([s[0] for s in dims.values()]))
        num_f_maps = int(num_f_maps[0])
        self.pars = {
            "input_dim": num_f_maps,
            "num_layers": num_ssl_layers,
            "output_dim": dim,
        }

    def construct_module(self) -> Union[nn.Module, None]:
        """Construct a TCN module."""
        module = DilatedTCN(**self.pars)
        return module


class MaskedFramesSSL(SSLConstructor, ABC):
    """Abstract class for masked frame SSL constructors.

    Generates the functions necessary to build a masked features SSL: real_lens some of the input features randomly
    and predict the initial data
    """

    type = "ssl_input"

    def __init__(self, frac_masked: float = 0.1) -> None:
        """Initialize the SSL constructor.

        Parameters
        ----------
        frac_masked : float, default 0.1
            fraction of frames to real_lens

        """
        super().__init__()
        self.frac_masked = frac_masked
        self.mse = MSE()

    def transformation(self, sample_data: Dict) -> Tuple:
        """Mask some of the frames randomly."""
        key = list(sample_data.keys())[0]
        num_frames = sample_data[key].shape[-1]
        mask = torch.empty(num_frames).normal_() > self.frac_masked
        mask = mask.unsqueeze(0)
        for key in sample_data:
            sample_data[key] = sample_data[key] * mask
        ssl_target = torch.cat(list(sample_data.values()))
        return (sample_data, ssl_target)

    def loss(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """MSE loss."""
        loss = self.mse(predicted, target)
        return loss

    @abstractmethod
    def construct_module(self) -> Union[nn.Module, None]:
        """Construct the SSL prediction module using the parameters specified at initialization."""


class MaskedFramesSSL_FC(MaskedFramesSSL):
    """Masked frames SSL with a fully connected module."""

    def __init__(
        self,
        dims: torch.Size,
        num_f_maps: torch.Size,
        frac_masked: float = 0.1,
        num_ssl_layers: int = 3,
        num_ssl_f_maps: int = 16,
    ) -> None:
        """Initialize the constructor.

        Parameters
        ----------
        dims : torch.Size
            the shape of features in model input
        num_f_maps : torch.Size
            shape of feature extraction output
        frac_masked : float, default 0.1
            fraction of frames to real_lens
        num_ssl_layers : int, default 5
            number of layers in the SSL module
        num_ssl_f_maps : int, default 16
            number of feature maps in the SSL module

        """
        super().__init__(frac_masked)
        dim = int(sum([s[0] for s in dims.values()]))
        num_f_maps = int(num_f_maps[0])
        self.pars = {
            "dim": dim,
            "num_f_maps": num_f_maps,
            "num_ssl_layers": num_ssl_layers,
            "num_ssl_f_maps": num_ssl_f_maps,
        }

    def construct_module(self) -> Union[nn.Module, None]:
        """Construct a fully connected module."""
        module = FC(**self.pars)
        return module


class MaskedFramesSSL_TCN(MaskedFramesSSL):
    """Masked frames SSL with a TCN module."""

    def __init__(
        self,
        dims: torch.Size,
        num_f_maps: torch.Size,
        frac_masked: float = 0.2,
        num_ssl_layers: int = 5,
    ) -> None:
        """Initialize the SSL constructor.

        Parameters
        ----------
        dims : torch.Size
            the number of features in model input
        num_f_maps : torch.Size
            shape of feature extraction output
        frac_masked : float, default 0.1
            fraction of frames to real_lens
        num_ssl_layers : int, default 5
            number of layers in the SSL module

        """
        super().__init__(frac_masked)
        dim = int(sum([s[0] for s in dims.values()]))
        num_f_maps = int(num_f_maps[0])
        self.pars = {
            "input_dim": num_f_maps,
            "num_layers": num_ssl_layers,
            "output_dim": dim,
        }

    def construct_module(self) -> Union[nn.Module, None]:
        """Construct a TCN module."""
        module = DilatedTCN(**self.pars)
        return module
