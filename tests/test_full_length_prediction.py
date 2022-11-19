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
    "default_agent_name": "mouse",
    "ignored_clips": ["single"],
}


def test_full_length_prediction():
    """
    Test the `dlc2action.data.dataset.BehaviorDataset.generate_full_length_prediction` function

    Check the shape of the predictions + make sure all ones stay all ones.
    """

    dataset = BehaviorDataset(**data_parameters)
    l = len(dataset)
    f = dataset.len_segment()
    prediction = torch.ones(l, 3, f)
    fl_prediction = dataset.generate_full_length_prediction(prediction)
    for video_id in fl_prediction:
        for clip_id in fl_prediction[video_id]:
            assert (fl_prediction[video_id][clip_id] == 1).all()
            assert fl_prediction[video_id][clip_id].shape[
                -1
            ] == dataset.input_store.get_clip_length(video_id, clip_id)
