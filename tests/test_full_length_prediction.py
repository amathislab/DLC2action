#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
import torch
from dlc2action.data.dataset import BehaviorDataset
import pytest
import os
import shutil

path = os.path.join(os.path.dirname(__file__), "data")

data_parameters = {
    "data_type": "dlc_track",
    "annotation_type": "csv",
    "data_path": path,
    "annotation_path": path,
    "annotation_suffix": {".csv"},
    "data_suffix": {
        "DeepCut_resnet50_Blockcourse1May9shuffle1_1030000.csv",
    },
    "canvas_shape": [1000, 500],
}


def test_full_length_prediction():
    """
    Test the `dlc2action.data.dataset.BehaviorDataset.generate_full_length_prediction` function

    Check the shape of the predictions + make sure all ones stay all ones.
    """

    folder = os.path.join(path, "trimmed")
    if os.path.exists(folder):
        shutil.rmtree(folder)
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
    folder = os.path.join(path, "trimmed")
    if os.path.exists(folder):
        shutil.rmtree(folder)

# test_full_length_prediction()
