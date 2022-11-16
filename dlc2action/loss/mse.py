#
# DLC2Action is not open-sourced yet.
# https://choosealicense.com/no-permission/
#
"""
The mean squared error loss
"""

from torch import nn
import torch


class _MSE(nn.Module):
    """
    Mean square error with ignore_index parameter
    """

    def __init__(self, ignore_index=-100):
        """
        Parameters
        ----------
        ignore_index : int
            the elements where target is equal to ignore_index will be ignored
        """

        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute the loss

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
