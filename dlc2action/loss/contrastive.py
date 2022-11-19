#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
"""
Losses used by contrastive SSL constructors (see `dlc2action.ssl.contrastive`)
"""

from torch import nn
import torch
import numpy as np
from itertools import combinations_with_replacement


class _NTXent(nn.Module):
    """
    NT-Xent loss for the contrastive SSL module
    """

    def __init__(self, tau: float):
        """
        Parameters
        ----------
        tau : float
            the tau parameter
        """

        super().__init__()
        self.tau = tau

    def _exp_similarity(self, tensor1, tensor2):
        """
        Compute exponential similarity
        """

        s = torch.cosine_similarity(tensor1, tensor2, dim=1) / self.tau
        s = torch.exp(s)
        return s

    def _loss(self, similarity, true_indices, denom):
        """
        Compute one of the symmetric components of the loss
        """

        l = torch.log(similarity[true_indices] / denom)
        l = -torch.mean(l)
        return l

    def forward(self, features1, features2):
        """
        Compute the loss
        Parameters
        ----------
        features1, features2 : torch.Tensor
            tensor of shape `(#batch, #features)`
        Returns
        -------
        loss : float
            the loss value
        """

        indices = list(combinations_with_replacement(list(range(len(features1))), 2))
        if len(indices) < 3:
            return 0
        indices1, indices2 = map(list, zip(*indices))
        true = np.unique(indices1, return_index=True)[1]
        similarity1 = self._exp_similarity(features1[indices1], features1[indices2])
        similarity2 = self._exp_similarity(features2[indices1], features2[indices2])
        similarity12 = self._exp_similarity(features1[indices1], features2[indices2])
        sum1 = similarity1.sum() - similarity1[true].sum()
        sum2 = similarity2.sum() - similarity2[true].sum()
        sum12 = similarity12.sum()
        loss = self._loss(similarity12, true, sum12 + sum1) + self._loss(
            similarity12, true, sum12 + sum2
        )
        return loss


class _TripletLoss(nn.Module):
    """
    Triplet loss for the pairwise SSL module
    A slightly modified version: a Softplus function is applied at the last step instead of ReLU to keep
    the result differentiable
    """

    def __init__(self, margin: float = 0, distance: str = "cosine"):
        """
        Parameters
        ----------
        margin : float, default 0
            the margin parameter
        distance : {'cosine', 'euclidean'}
            the distance metric (cosine similarity ot euclidean distance)
        """

        super().__init__()
        self.margin = margin
        self.nl = nn.Softplus()
        if distance == "euclidean":
            self.distance = self._euclidean_distance
        elif distance == "cosine":
            self.distance = self._cosine_similarity
        else:
            raise ValueError(
                f'The {distance} is not available, please choose from "euclidean" and "cosine"'
            )

    def _euclidean_distance(self, tensor1, tensor2):
        """
        Compute euclidean distance
        """

        return torch.sum((tensor1 - tensor2) ** 2, dim=1)

    def _cosine_similarity(self, tensor1, tensor2):
        """
        Compute cosine similarity
        """

        return torch.cosine_similarity(tensor1, tensor2, dim=1)

    def forward(self, features1, features2):
        """
        Compute the loss
        Parameters
        ----------
        features1, features2 : torch.Tensor
            tensor of shape `(#batch, #features)`
        Returns
        -------
        loss : float
            the loss value
        """

        negative = torch.cat([features2[1:], features2[:1]])
        positive_distance = self.distance(features1, features2)
        negative_distance = self.distance(features1, negative)
        loss = torch.mean(self.nl(positive_distance - negative_distance + self.margin))
        return loss


class _CircleLoss(nn.Module):
    """
    Circle loss for the pairwise SSL module
    """

    def __init__(self, gamma: float = 1, margin: float = 0, distance: str = "cosine"):
        """
        Parameters
        ----------
        gamma : float, default 1
            the gamma parameter
        margin : float, default 0
            the margin parameter
        distance : {'cosine', 'euclidean'}
            the distance metric (cosine similarity ot euclidean distance)
        """

        super().__init__()
        self.gamma = gamma
        self.margin = margin
        if distance == "euclidean":
            self.distance = self._euclidean_distance
        elif distance == "cosine":
            self.distance = self._cosine_similarity
        else:
            raise ValueError(
                f'The {distance} is not available, please choose from "euclidean" and "cosine"'
            )

    def _euclidean_distance(self, tensor1, tensor2):
        """
        Compute euclidean distance
        """

        return torch.sum((tensor1 - tensor2) ** 2, dim=1)

    def _cosine_similarity(self, tensor1, tensor2):
        """
        Compute cosine similarity
        """

        return torch.cosine_similarity(tensor1, tensor2, dim=1)

    def forward(self, features1, features2):
        """
        Compute the loss
        Parameters
        ----------
        features1, features2 : torch.Tensor
            tensor of shape `(#batch, #features)`
        Returns
        -------
        loss : float
            the loss value
        """

        indices = list(combinations_with_replacement(list(range(len(features1))), 2))
        indices1, indices2 = map(list, zip(*indices))
        true = np.unique(indices1, return_index=True)[1]
        mask = torch.zeros(len(indices1)).bool()
        mask[true] = True
        distances = self.distance(features1[indices1], features2[indices2])
        distances[mask] = distances[mask] + self.margin
        distances[~mask] = distances[~mask] * (-1)
        distances = torch.exp(self.gamma * distances)
        loss = torch.log(1 + torch.sum(distances[mask]) * torch.sum(distances[~mask]))
        return loss
