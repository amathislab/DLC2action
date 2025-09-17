#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
"""The mean squared error loss."""

import torch
from torch import nn


class MSE(nn.Module):
    """Mean square error with ignore_index parameter."""

    def __init__(self, ignore_index=-100):
        """Initialize the loss.

        Parameters
        ----------
        ignore_index : int
            the elements where target is equal to ignore_index will be ignored

        """
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """Compute the loss.

        Parameters
        ----------
        predicted, target : torch.Tensor
            a tensor of any shape

        Returns
        -------
        loss : float
            the loss value

        """
        mask = target != self.ignore_index
        return torch.mean((predicted[mask] - target[mask]) ** 2)
