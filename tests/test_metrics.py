#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
import torch
from dlc2action import options
import inspect
import numpy as np
import pytest
from copy import deepcopy


def generate_sample_exclusive(frac_wrong: float):
    """
    Generate a sample with fixed fraction of mistakes for exclusive classification
    """

    target = torch.randint(0, 3, (5, 100))
    wrong_len = int(frac_wrong * 100)
    wrong_mask = torch.zeros((5, 100))
    wrong_mask[:, :wrong_len] = 1
    wrong_mask = wrong_mask.bool()
    prediction = deepcopy(target)
    prediction[wrong_mask] = prediction[wrong_mask] - 1
    prediction[prediction == -1] = 2
    return prediction, target


def generate_sample_multilabel(frac_wrong: float):
    """
    Generate a sample with fixed fraction of mistakes for multi-label classification
    """

    target = torch.randint(0, 2, (5, 3, 100))
    target[target == 0] = -1
    wrong_len = int(frac_wrong * 100)
    wrong_mask = torch.zeros((5, 3, 100))
    wrong_mask[:, :, :wrong_len] = 1
    wrong_mask = wrong_mask.bool()
    prediction = deepcopy(target)
    prediction[wrong_mask] = prediction[wrong_mask] * (-1)
    prediction[prediction == -1] = 0
    target[target == -1] = 0
    return prediction, target


metrics = [x for x in options.metrics if x not in options.metrics_no_direction]


@pytest.mark.parametrize("metric_name", metrics)
@pytest.mark.parametrize("exclusive", [True, False])
def test_metric(metric_name: str, exclusive: bool):
    """
    Test that the metric correctly decreases/increases with increasing `frac_wrong`
    """

    if metric_name.endswith("pr-auc") and exclusive:
        return
    metric_class = options.metrics[metric_name]
    parameter_names = inspect.getfullargspec(metric_class).args
    pars = {}
    if "num_classes" in parameter_names:
        pars["num_classes"] = 3
    if "exclusive" in parameter_names:
        pars["exclusive"] = exclusive
    if "average" in parameter_names:
        pars["average"] = "macro"
    metric = metric_class(**pars)
    results = []
    for frac_wrong in [0, 0.25, 0.5, 0.75, 1]:
        metric.reset()
        for i in range(1):
            if exclusive:
                predicted, target = generate_sample_exclusive(frac_wrong=frac_wrong)
            else:
                predicted, target = generate_sample_multilabel(frac_wrong=frac_wrong)
            if metric.needs_raw_data and exclusive:
                predicted_raw = torch.zeros((5, 3, 100))
                for i in range(5):
                    predicted_raw[i, predicted[i], range(100)] = 1
                predicted = predicted_raw
            elif metric.needs_raw_data and not exclusive:
                predicted = predicted.float()
                rand = 0.5 * torch.rand(predicted.shape)
                predicted[predicted == 1] = (0.5 + rand)[predicted == 1]
                predicted[predicted == 0] = rand[predicted == 0]
            metric.update(
                predicted=predicted,
                target=target,
                tags=None,
            )
        res = metric.calculate()
        results.append(res)
    if metric_name in options.metrics_minimize:
        assert np.sum(np.array(results[1:]) - np.array(results[:-1]) > 0) >= 3
    else:
        assert np.sum(np.array(results[1:]) - np.array(results[:-1]) < 0) >= 3


# test_metric('pku-map', True)
