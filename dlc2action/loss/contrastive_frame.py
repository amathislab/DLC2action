#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
from torch import nn
import torch


class _ContrastiveRegressionLoss(nn.Module):
    def __init__(self, temperature: float, distance: str, break_factor: int):
        self.temp = temperature
        self.distance = distance
        self.break_factor = break_factor
        assert distance in ["l1", "l2", "cosine"]
        super(_ContrastiveRegressionLoss, self).__init__()

    def _get_distance_matrix(self, tensor1, tensor2) -> torch.Tensor:
        """
        Shape (#features, #frames)
        """

        if self.distance == "l1":
            dist = torch.cdist(tensor1.T, tensor2.T, p=1)
        elif self.distance == "l2":
            dist = torch.cdist(tensor1.T, tensor2.T, p=2)
        else:
            dist = torch.nn.functional.cosine_similarity(
                tensor1.t()[:, :, None], tensor2[None, :, :]
            )
        return dist

    def forward(self, tensor1, tensor2):
        """
        Compute the loss

        Parameters
        ----------
        tensor1, tensor2 : torch.Tensor
            tensor of shape `(#batch, #features, #frames)`
        Returns
        -------
        loss : float
            the loss value
        """

        loss = 0
        if self.break_factor is not None:
            B, C, T = tensor1.shape
            if T % self.break_factor != 0:
                tensor1 = tensor1[:, :, : -(T % self.break_factor)]
                tensor2 = tensor2[:, :, : -(T % self.break_factor)]
                T -= T % self.break_factor
            tensor1 = tensor1.reshape((B, C, self.break_factor, -1))
            tensor1 = torch.transpose(tensor1, 1, 2).reshape(
                (B * self.break_factor, C, -1)
            )
            tensor2 = tensor2.reshape((B, C, self.break_factor, -1))
            tensor2 = torch.transpose(tensor2, 1, 2).reshape(
                (B * self.break_factor, C, -1)
            )
        indices = torch.tensor(range(tensor1.shape[-1])).to(tensor1.device)
        for i in range(tensor1.shape[0]):
            out = torch.exp(
                self._get_distance_matrix(tensor1[i], tensor2[i]) / self.temp
            )
            out = out / (torch.sum(out, 1).unsqueeze(1) + 1e-7)
            out = torch.sum(out * indices.unsqueeze(0), 1)
            loss += torch.sum((out - indices) ** 2)
        return loss / (tensor1.shape[-1] * tensor1.shape[0])
