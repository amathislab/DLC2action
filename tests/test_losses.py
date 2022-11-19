#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
import torch
import pytest
from copy import deepcopy
from dlc2action import options
import inspect
import numpy as np


def generate_sample_exclusive(frac_wrong: float):
    """
    Generate a sample with fixed fraction of mistakes for exclusive classification
    """

    prediction = torch.rand((5, 3, 100))
    _, target = torch.max(prediction, dim=1)
    wrong_len = int(frac_wrong * 100)
    wrong_mask = torch.zeros((5, 100))
    wrong_mask[:, :wrong_len] = 1
    wrong_mask = wrong_mask.bool()
    target[wrong_mask] = target[wrong_mask] - 1
    target[target == -1] = 2
    return prediction, target


def generate_sample_multilabel(frac_wrong: float):
    """
    Generate a sample with fixed fraction of mistakes for exclusive classification
    """

    prediction = torch.rand((5, 3, 100))
    target = (prediction > 0.5).float()
    wrong_len = int(frac_wrong * 100)
    wrong_mask = torch.zeros((5, 3, 100))
    wrong_mask[:, :, :wrong_len] = 1
    wrong_mask = wrong_mask.bool()
    target[wrong_mask] = target[wrong_mask] - 1
    target[target == -1] = 1
    return prediction, target


losses = options.losses


@pytest.mark.skip
@pytest.mark.parametrize("loss_name", losses)
@pytest.mark.parametrize("exclusive", [True, False])
def test_loss(loss_name: str, exclusive: bool):
    """
    Test that the metric correctly decreases/increases with increasing `frac_wrong`
    """

    loss_class = options.losses[loss_name]
    parameter_names = inspect.getfullargspec(loss_class).args
    pars = {}
    if "num_classes" in parameter_names:
        pars["num_classes"] = 3
    if "exclusive" in parameter_names:
        pars["exclusive"] = exclusive
    loss = loss_class(**pars)
    results = []
    for frac_wrong in [0, 0.25, 0.5, 0.75, 1]:
        res = []
        for i in range(50):
            if exclusive:
                predicted, target = generate_sample_exclusive(frac_wrong=frac_wrong)
            else:
                predicted, target = generate_sample_multilabel(frac_wrong=frac_wrong)
            if loss_name in options.losses_multistage:
                predicted = torch.stack(2 * [predicted])
            res.append(loss(predicted, target))
        results.append(sum(res) / len(res))
    assert np.sum(np.array(results[1:]) - np.array(results[:-1]) > 0) >= 3


# test_loss("clip", True)
