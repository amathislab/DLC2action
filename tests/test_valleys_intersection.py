#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
import torch
from dlc2action.data.dataset import BehaviorDataset
import pytest

data_parameters = {
    "data_type": "dlc_track",
    "annotation_type": "boris",
    "data_path": "/home/liza/data/cricket",
    "annotation_path": "/home/liza/data/cricket",
    "behaviors": ["Grooming", "Search", "Pursuit"],
    "annotation_suffix": {".csv"},
    "data_suffix": {
        "DLC_resnet50_preycapSep30shuffle1_20000_bx_filtered.h5",
    },
    "canvas_shape": [2250, 1250],
    "default_agent_name": "mouse+single",
    "interactive": True,
    "feature_extraction_pars": {"interactive": True},
}


def test_valleys_intersection():
    """
    Test `dlc2action.data.dataset.BehaviorDataset.valleys_intersection'
    """

    dataset = BehaviorDataset(**data_parameters)
    l = len(dataset)
    f = dataset.len_segment()
    prediction = torch.zeros(l, 3, f)
    prediction[0, 1, : f // 2] = 1
    valleys1 = dataset.find_valleys(predicted=prediction, threshold=0.5, low=False)
    prediction = torch.zeros(l, 3, f)
    prediction[0, 1, f // 4 :] = 1
    valleys2 = dataset.find_valleys(predicted=prediction, threshold=0.5)
    result = dataset.valleys_intersection([valleys2, valleys1])
    coords = dataset.input_store.get_original_coordinates()[0]
    video_id = dataset.input_store.get_video_id(coords)
    assert len(result[video_id]) == 1
    prediction = torch.zeros(l, 3, f)
    prediction[0, 1, 3 * f // 4 :] = 1
    valleys2 = dataset.find_valleys(predicted=prediction, threshold=0.5, low=False)
    result = dataset.valleys_intersection([valleys2, valleys1])
    assert len(result[video_id]) == 0
