#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import os
import shutil
import pytest
import torch
from dlc2action.data.dataset import BehaviorDataset
import yaml

with open("tests/config_test.yaml", "r") as f:
    config = yaml.safe_load(f)
crim_data_path = config["crim_data_path"]

data_parameters = {
    "data_type": "simba",
    "annotation_type": "simba",
    "data_path": crim_data_path,
    "annotation_path": crim_data_path,
    "behaviors": [
                    "approach",
                    "attack",
                    "copulation",
                    "chase",
                    "circle",
                    "drink",
                    "eat",
                    "clean",
                    "sniff",
                    "up",
                    "walk_away",
                ],
    "annotation_suffix": {".csv"},
    "data_suffix": {
        ".csv",
    },
    "canvas_shape": [1290, 730],
    "use_features" : False
}


def test_full_length_prediction():
    """
    Test the `dlc2action.data.dataset.BehaviorDataset.generate_full_length_prediction` function

    Check the shape of the predictions + make sure all ones stay all ones.
    """

    folder = os.path.join(crim_data_path, "trimmed")
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
    folder = os.path.join(crim_data_path, "trimmed")
    if os.path.exists(folder):
        shutil.rmtree(folder)

# test_full_length_prediction()
