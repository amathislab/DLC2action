#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import os

from typing import Dict

import pytest
import torch
from dlc2action.data.dataset import BehaviorDataset
import yaml
import sys


skip_when_not_direct = pytest.mark.skipif(
    not any(f.endswith(os.path.basename(__file__)) for f in sys.argv),
    reason="Skipped when not run directly",
)

with open("tests/config_test.yaml", "r") as f:
    config = yaml.safe_load(f)

crim_data_path = config["crim_data_path"]
oft_data_path = config["oft_data_path"]

test_data = [
    {
        "data_type": "dlc_track",
        "annotation_type": "csv",
        "data_path": oft_data_path,
        "annotation_path": oft_data_path,
        "canvas_shape": [928, 576],
        "data_suffix": "DeepCut_resnet50_Blockcourse1May9shuffle1_1030000.csv",
        "annotation_suffix": ".csv",
        "behaviors": ["Grooming", "Supported", "Unsupported"],
        "len_segment": 30,
        "overlap": 0,
        "exclusive": True,
        "only_load_annotated": True,
        "feature_extraction": "heatmap",
        "feature_extraction_pars": {"canvas_shape": [928, 576]},
    },
    {
        "data_type": "dlc_track",
        "annotation_type": "csv",
        "data_path": oft_data_path,
        "annotation_path": oft_data_path,
        "canvas_shape": [928, 576],
        "data_suffix": "DeepCut_resnet50_Blockcourse1May9shuffle1_1030000.csv",
        "annotation_suffix": ".csv",
        "behaviors": ["Grooming", "Supported", "Unsupported"],
        "len_segment": 30,
        "overlap": 0,
        "exclusive": True,
        "only_load_annotated": True,
    },
]

@skip_when_not_direct
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


if __name__ == "__main__":
    test_dataset_creation(test_data[0])
    test_dataset_creation(test_data[1])
