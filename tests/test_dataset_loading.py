#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
from typing import Dict
from dlc2action.data.dataset import BehaviorDataset
import pytest
import os
import torch

test_data = [
    {
        "data_type": "dlc_track",
        "annotation_type": "dlc",
        "data_path": "/home/liza/data/maushaus",
        "annotation_path": "/home/liza/data/maushaus",
        "canvas_shape": [800, 600],
        "data_prefix": {"ac-", "b1-"},  # multiview
        "data_suffix": {"DLC_resnet50.h5"},
        "feature_suffix": ".pt",
        "len_segment": 512,
        "overlap": 0,
        "behaviors": [
            "grooming",
            "general locomotion",
            "running on treadmill",
            "hut",
            "water",
        ],
        "annotation_suffix": {"_lane.pickle"},
        "exclusive": False,
        "only_load_annotated": True,
    },
    # {
    #     "data_type": "pku-mmd",
    #     "annotation_type": "pku-mmd",
    #     "data_path": "/home/liza/data/pku_mmd/PKU_Skeleton_Renew/sample_tmp",
    #     "annotation_path": "/home/liza/data/pku_mmd/Train_Label_PKU_final/sample_tmp",
    #     "behavior_file": "/home/liza/data/pku_mmd/Actions.xlsx",
    # },
    {
        "data_type": "dlc_tracklet",
        "annotation_type": "dlc",
        "feature_suffix": "_pca_feat.pickle",
        "data_path": "/home/liza/data/marmoset_sample",
        "annotation_path": "/home/liza/data/marmoset_sample",
        "behaviors": ["inactive alert", "locomotion"],
        "correction": {
            "calm locomotion": "locomotion",
            "agitated_locomotion": "locomotion",
            "leaping": "locomotion",
            "chewing": "inactive alert",
            "looking around": "inactive alert",
        },
        "annotation_suffix": {"_banty.h5", "_banty.pickle"},
        "error_class": "DLC error",
        "data_suffix": {
            "_el.pickle",
        },
        "frame_limit": 2,
        "canvas_shape": [1280, 720],
        "ignored_bodyparts": {"tailend", "tail1", "tail2"},
    },
    {
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
    },
    {
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
        "feature_extraction": "heatmap",
        "canvas_shape": [2250, 1088],
        "feature_extraction_pars": {"canvas_shape": [2250, 1088]},
    },
    # {
    #     "data_type": "calms21",
    #     "annotation_type": "calms21",
    #     "task_n": 1,
    #     "data_path": "/home/liza/data/calms21",
    #     "annotation_path": "/home/liza/data/calms21",
    #     "len_segment": 256,
    #     "overlap": 100,
    #     "only_load_annotated": True,
    #     "interactive": True,
    # },
]


@pytest.mark.skip
@pytest.mark.parametrize("data_parameters", test_data)
def test_dataset_creation(data_parameters: Dict):
    """
    Check dataset creation and loading

    Make sure that with all data-annotation combinations the dataset gets created + when it's saved and re-loaded
    it is still the same data and runs `dlc2action.data.dataset.BehaviorDataset.find_valleys` without errors
    (that function calls most of the input store methods +
    `dlc2action.data.dataset.BehaviorDataset.generate_full_length_prediction`).
    """

    if (
        data_parameters.get("data_prefix") is not None
        and len(data_parameters["data_prefix"]) > 1
    ):
        prefix = data_parameters["data_prefix"]
        data_parameters["data_prefix"] = {list(prefix)[0]}
        dataset = BehaviorDataset(**data_parameters)
        shapes = dataset.features_shape()
        num_features = len(shapes)
        data_parameters["data_prefix"] = prefix
    dataset = BehaviorDataset(**data_parameters)
    if data_parameters.get("data_prefix") is not None:
        shapes = dataset.features_shape()
        assert len(shapes) > num_features
    sample_last = dataset[-1]
    dataset.save("dataset_tmp.pickle")
    dataset = BehaviorDataset(
        data_type=data_parameters["data_type"],
        annotation_type=data_parameters["annotation_type"],
        saved_data_path="dataset_tmp.pickle",
    )
    assert (sample_last["target"] == dataset[-1]["target"]).all() and all(
        [
            (value == dataset[-1]["input"][key]).all()
            for key, value in sample_last["input"].items()
        ]
    )
    os.remove("dataset_tmp.pickle")
    dataset.to_ram()
    assert (sample_last["target"] == dataset[-1]["target"]).all() and all(
        [
            (value == dataset[-1]["input"][key]).all()
            for key, value in sample_last["input"].items()
        ]
    )
    l = len(dataset)
    f = dataset.len_segment()
    prediction = torch.ones(l, 3, f)
    dataset.find_valleys(predicted=prediction)
    assert len(dataset.annotation_store) > 0


# test_dataset_creation(test_data[-1])
# for data in test_data:
#     test_dataset_creation(data)
