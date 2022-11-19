#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
"""
Abstract class for defining SSL tasks
"""

from typing import Callable, Tuple, Dict, Union
from abc import ABC, abstractmethod
import torch
from torch import nn


class SSLConstructor(ABC):
    """
    A base class for all SSL constructors

    An SSL method is defined by three things: a *transformation* that maps a sample into SSL input and output,
    a neural net *module* that takes features as input and predicts SSL target, a *type* and a *loss function*.
    """

    type = "none"
    """
    The `type` parameter defines interaction with the model:
    
    - `'ssl_input'`: a modification of the input data passes through the base network feature extraction module and the
    SSL module; it is returned as SSL output and compared to SSL target (or, if it is None, to the input data),
    - `'ssl_output'`:  the input data passes through the base network feature extraction module and the SSL module; it
    is returned as SSL output and compared to SSL target (or, if it is None, to the input data),
    - `'contrastive'`:  the input data and its modification pass through the base network feature extraction module and
    the SSL module; an (input results, modification results) tuple is returned as SSL output,
    - `'contrastive_2layers'`: the input data and its modification pass through the base network feature extraction module; 
    the output of the second feature extraction layer for the modified data goes through an SSL module and then, 
    optionally, that result and the first-level unmodified features pass another transformation; 
    an (input results, modified results) tuple is returned as SSL output,
    """

    def __init__(self, *args, **kwargs) -> None:
        ...

    @abstractmethod
    def transformation(self, sample_data: Dict) -> Tuple[Dict, Dict]:
        """
        Transform a sample feature dictionary into SSL input and target

        Either input, target or both can be left as `None`. Transformers can be configured to replace `None` SSL targets
        with the input sample at runtime and/or to replace `None SSL` inputs with a new augmentation of the input sample.
        If the keys of the feature dictionaries are recognized by the transformer, they will be augmented before
        all features are stacked together.

        Parameters
        ----------
        sample_data : dict
            a feature dictionary

        Returns
        -------
        ssl_input : dict | torch.float('nan')
            a feature dictionary of SSL inputs
        ssl_target : dict | torch.float('nan')
            a feature dictionary of SSL targets
        """

    @abstractmethod
    def loss(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """
        Calculate the SSL loss

        Parameters
        ----------
        predicted : torch.Tensor
            output of the SSL module
        target : torch.Tensor
            augmented and stacked SSL_target

        Returns
        -------
        loss : float
            the loss value
        """

    @abstractmethod
    def construct_module(self) -> nn.Module:
        """
        Construct the SSL module

        Returns
        -------
        ssl_module : torch.nn.Module
            a neural net module that takes features extracted by a model's feature extractor as input and
            returns SSL output
        """


class EmptySSL(SSLConstructor):
    """
    Empty SSL class
    """

    def transformation(self, sample_data: Dict) -> Tuple:
        """
        Empty transformation
        """

        return (torch.tensor(float("nan")), torch.tensor(float("nan")))

    def loss(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """
        Empty loss
        """

        return 0

    def construct_module(self) -> None:
        """
        Empty module
        """

        return None
