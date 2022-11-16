#
# DLC2Action Toolbox
# © A. Mathis Lab
# https://github.com/amathislab/DLC2Action/
#
# Adapted from MS-TCN++ by yabufarha
# Adapted from https://github.com/sj-li/MS-TCN2
# Licensed under MIT License
#
"""
Loss for the MS-TCN models

Adapted from https://github.com/sj-li/MS-TCN2
"""

from torch import nn
import torch.nn.functional as F
import torch
from collections.abc import Iterable
from copy import copy
import sys
from typing import Optional


class MS_TCN_Loss(nn.Module):
    """
    The MS-TCN loss
    Crossentropy + consistency loss (MSE over predicted probabilities)
    """

    def __init__(
        self,
        num_classes: int,
        weights: Iterable = None,
        exclusive: bool = True,
        ignore_index: int = -100,
        focal: bool = False,
        gamma: float = 1,
        alpha: float = 0.15,
        hard_negative_weight: float = 1,
    ) -> None:
        """
        Parameters
        ----------
        num_classes : int
            number of classes
        weights : iterable, optional
            class-wise cross-entropy weights
        exclusive : bool, default True
            True if single-label classification is used
        ignore_index : int, default -100
            the elements where target is equal to ignore_index will be ignored by cross-entropy
        focal : bool, default False
            if True, instead of regular cross-entropy the focal loss will be used
        gamma : float, default 1
            the gamma parameter of the focal loss
        alpha : float, default 0.15
            the weight of the consistency loss
        hard_negative_weight : float, default 1
            the weight assigned to the hard negative frames
        """

        super().__init__()
        self.weights = weights
        self.num_classes = int(num_classes)
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.focal = focal
        self.alpha = alpha
        self.exclusive = exclusive
        self.neg_weight = hard_negative_weight
        if exclusive:
            self.log_nl = lambda x: F.log_softmax(x, dim=1)
        else:
            self.log_nl = lambda x: torch.log(torch.sigmoid(x) + 1e-7)
        self.mse = nn.MSELoss(reduction="none")
        if self.weights is not None:
            self.need_init = True
        else:
            self.need_init = False
            self._init_ce()

    def _init_ce(self) -> None:
        """
        Initialize cross-entropy function
        """

        if self.exclusive:
            if self.focal:
                self.ce = nn.CrossEntropyLoss(
                    ignore_index=self.ignore_index,
                    weight=self.weights,
                    reduction="none",
                )
            else:
                self.ce = nn.CrossEntropyLoss(
                    ignore_index=self.ignore_index, weight=self.weights
                )
        else:
            self.ce = nn.BCEWithLogitsLoss(reduction="none")

    def _init_weights(self, device: str) -> None:
        """
        Initialize the weights vector and the cross-entropy function (after the device is known)
        """

        if self.exclusive:
            self.weights = torch.tensor(self.weights, device=device, dtype=torch.float)
        else:
            self.weights = {
                k: torch.tensor(v, device=device, dtype=torch.float)
                .unsqueeze(0)
                .unsqueeze(-1)
                for k, v in self.weights.items()
            }
        self._init_ce()
        self.need_init = False

    def _ce_loss(self, p: torch.Tensor, t: torch.Tensor) -> float:
        """
        Apply cross-entropy loss
        """

        if self.exclusive:
            p = p.transpose(2, 1).contiguous().view(-1, self.num_classes)
            t = t.view(-1)
            mask = t != self.ignore_index
            if torch.sum(mask) == 0:
                return 0
            if self.focal:
                pr = F.softmax(p[mask], dim=1)
                f = (1 - pr[range(torch.sum(mask)), t[mask]]) ** self.gamma
                loss = (f * self.ce(p, t)[mask]).mean()
                return loss
            else:
                loss = self.ce(p, t)
                return loss
        else:
            if self.weights is not None:
                weight0 = self.weights[0]
                weight1 = self.weights[1]
            else:
                weight0 = 1
                weight1 = 1
            neg_mask = t == 2
            target = copy(t)
            target[neg_mask] = 0
            loss = self.ce(p, target)
            loss[t == 1] = (loss * weight1)[t == 1]
            loss[t == 0] = (loss * weight0)[t == 0]
            if self.neg_weight > 1:
                loss[neg_mask] = (loss * self.neg_weight)[neg_mask]
            elif self.neg_weight < 1:
                inv_neg_mask = (~neg_mask) * (target == 0)
                loss[inv_neg_mask] = (loss * self.neg_weight)[inv_neg_mask]
            if self.focal:
                pr = torch.sigmoid(p)
                factor = target * ((1 - pr) ** self.gamma) + (1 - target) * (
                    pr**self.gamma
                )
                loss = loss * factor
            loss = loss[target != self.ignore_index]
            return loss.mean() if loss.size()[-1] != 0 else 0

    def consistency_loss(self, p: torch.Tensor) -> float:
        """
        Apply consistency loss
        """

        mse = self.mse(self.log_nl(p[:, :, 1:]), self.log_nl(p.detach()[:, :, :-1]))
        clamp = torch.clamp(mse, min=0, max=16)
        return torch.mean(clamp)

    def forward(self, predictions: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute the loss
        Parameters
        ----------
        predictions : torch.Tensor
            a tensor of shape (#batch, #classes, #frames)
        target : torch.Tensor
            a tensor of shape (#batch, #classes, #frames) or (#batch, #frames)
        Returns
        -------
        loss : float
            the loss value
        """

        if self.need_init:
            if isinstance(predictions, dict):
                device = predictions["device"]
            else:
                device = predictions.device
            self._init_weights(device)
        loss = 0
        if len(predictions.shape) == 4:
            for p in predictions:
                loss += self._ce_loss(p, target)
                loss += self.alpha * self.consistency_loss(p)
        else:
            loss += self._ce_loss(predictions, target)
            loss += self.alpha * self.consistency_loss(predictions)
        return loss / len(predictions)


class BoundaryRegressionLoss(nn.Module):
    """
    Boundary Regression Loss
        bce: Binary Cross Entropy Loss for Boundary Prediction
        mse: Mean Squared Error
    """

    def __init__(
        self,
        bce: bool = True,
        focal: bool = False,
        mse: bool = False,
        weight: Optional[torch.Tensor] = None,
        pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.criterions = []

        if bce:
            self.criterions.append(
                nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight)
            )

        if focal:
            self.criterions.append(FocalLoss())

        if mse:
            self.criterions.append(nn.MSELoss())

        if len(self.criterions) == 0:
            print("You have to choose at least one loss function.")
            sys.exit(1)

    def forward(self, preds: torch.Tensor, gts: torch.Tensor, masks: torch.Tensor):
        """
        Args:
            preds: torch.float (N, 1, T).
            gts: torch. (N, 1, T).
            masks: torch.bool (N, 1, T).
        """
        loss = 0.0
        batch_size = float(preds.shape[0])

        for criterion in self.criterions:
            for pred, gt, mask in zip(preds, gts, masks):
                loss += criterion(pred[mask], gt[mask].float())

        return loss / batch_size


class FocalLoss(nn.Module):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        size_average: bool = True,
        batch_average: bool = True,
        ignore_index: int = 255,
        gamma: float = 2.0,
        alpha: float = 0.25,
    ) -> None:
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.batch_average = batch_average
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_index, size_average=size_average
        )

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n, _, _ = logit.size()

        logpt = -self.criterion(logit, target.long())
        pt = torch.exp(logpt)

        if self.alpha is not None:
            logpt *= self.alpha

        loss = -((1 - pt) ** self.gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss


class ASRFLoss(MS_TCN_Loss):
    def __init__(
        self,
        num_classes: int,
        weights: Iterable = None,
        boundary_pos_weight: float = None,
        boundary_weight: float = 1,
        exclusive: bool = True,
        ignore_index: int = -100,
        focal: bool = False,
        gamma: float = 1,
        alpha: float = 0.15,
        hard_negative_weight: float = 1,
    ):
        self.pos_weight = [boundary_pos_weight]
        self.b_weight = boundary_weight
        if not exclusive:
            raise ValueError(
                "The ASRF loss is only implemented for exclusive classification"
            )
        super().__init__(
            num_classes,
            weights,
            exclusive,
            ignore_index,
            focal,
            gamma,
            alpha,
            hard_negative_weight,
        )

    def _init_ce(self) -> None:
        """
        Initialize cross-entropy function
        """

        super()._init_ce()
        self.bl = BoundaryRegressionLoss(pos_weight=self.pos_weight)

    def _init_weights(self, device: str) -> None:
        """
        Initialize the weights vector and the cross-entropy function (after the device is known)
        """

        if self.exclusive:
            self.weights = torch.tensor(self.weights, device=device, dtype=torch.float)
        else:
            self.weights = {
                k: torch.tensor(v, device=device, dtype=torch.float)
                .unsqueeze(0)
                .unsqueeze(-1)
                for k, v in self.weights.items()
            }
        self.pos_weight = torch.tensor(
            self.pos_weight, device=device, dtype=torch.float
        )
        self._init_ce()
        self.need_init = False

    def _compute_boundaries(self, tensor: torch.Tensor) -> torch.Tensor:
        f = tensor.flatten()
        _, inv = torch.unique_consecutive(f, return_inverse=True)
        boundary = torch.cat(
            [torch.tensor([0]).to(inv.device), torch.diff(inv)]
        ).reshape(tensor.shape)
        boundary[..., 0] = 0
        boundary = boundary.unsqueeze(1)
        return boundary

    def forward(self, predictions: torch.Tensor, target: torch.Tensor) -> float:
        boundaries = predictions[:, :, 0, :].unsqueeze(2)
        class_predictions = predictions[:, :, 1:, :]
        target_boundaries = self._compute_boundaries(target)
        mask = (target != -100).unsqueeze(1)
        loss = super().forward(class_predictions, target)
        for x in boundaries:
            loss += (
                self.b_weight * self.bl(x, target_boundaries, mask) / len(boundaries)
            )
        return loss
