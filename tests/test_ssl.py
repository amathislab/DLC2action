#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
from typing import List
import os
import pytest
from dlc2action import options
from dlc2action.project import Project
import yaml
import sys

device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

skip_when_not_direct = pytest.mark.skipif(
    not any(f.endswith(os.path.basename(__file__)) for f in sys.argv),
    reason="Skipped when not run directly"
)

constructors = list(options.ssl_constructors.keys())
with open("tests/config_test.yaml", "r") as f:
    config = yaml.safe_load(f)
oft_data_path = config["oft_data_path"]

@skip_when_not_direct
@pytest.mark.parametrize(
    "ssl_names", [[x] for x in constructors] + [constructors[:2], constructors[:3]]
)
def test_ssl(ssl_names: List):
    """
    Test SSL constructors

    Run one-epoch episodes with each SSL constructor + combinations of two and three.
    """

    Project.remove_project("test_ssl")
    project = Project(
        "test_ssl",
        data_type="dlc_track",
        annotation_type="csv",
        data_path=oft_data_path,
        annotation_path=oft_data_path,
    )
    project.update_parameters(
        {
            "data": {
                "data_suffix": "DeepCut_resnet50_Blockcourse1May9shuffle1_1030000.csv",  # set; the data files should have the format of {video_id}{data_suffix}, e.g. video1_suffix.pickle, where video1 is the video is and _suffix.pickle is the suffix
                "canvas_shape": [928, 576],
                "annotation_suffix": ".csv",  # str | set, optional the suffix or the set of suffices such that the annotation files are named {video_id}{annotation_suffix}, e.g, video1_suffix.pickle where video1 is the video id and _suffix.pickle is the suffix
                "behaviors": ["Grooming", "Supported", "Unsupported"],
                "ignored_bodyparts": {"tl", "tr", "br", "bl", "centre"},
                "fps": 25,
                "filter_background": False,
                "filter_annotated": False,
                "clip_frames": 0,
                "normalize": True,
                "use_features": False
            },
            "general": {
                "model_name": "c2f_tcn",
                "len_segment": 512,
                "overlap": 5,
                "exclusive": True,  # bool; if true, single-label classification is used; otherwise multi-label
                "only_load_annotated": True,
                "metric_functions": {"f1"},
                "ssl": ssl_names,
            },
            "training": {
                "partition_method": "random",
                "val_frac": 0.5,
                "normalize": False,
                "num_epochs": 1,
                "ssl_on": True,
                "ssl_weight": {s: 0.2 for s in ssl_names},
                "augment_train": 1,
                "skip_normalization_keys": ["speed_direction", "coord_diff"],
                "temporal_subsampling_size": 1,
                "batch_size": 16,
                "device": device,
            },
            "features": {
                "egocentric": True,
                "distance_pairs": None,
                "keys": {
                    "coords",
                    "angle_speeds",
                    "areas",
                    "acc_joints",
                    "center",
                    "speed_direction",
                    "speed_value",
                },
                "angle_pairs": [
                    ["tailbase", "tailcentre", "tailcentre", "tailtip"],
                    ["hipr", "tailbase", "tailbase", "hipl"],
                    ["tailbase", "bodycentre", "bodycentre", "neck"],
                    ["bcr", "bodycentre", "bodycentre", "bcl"],
                    ["bodycentre", "neck", "neck", "headcentre"],
                    ["tailbase", "bodycentre", "neck", "headcentre"],
                ],
                "area_vertices": [
                    ["tailbase", "hipr", "hipl"],
                    ["hipr", "hipl", "bcl", "bcr"],
                    ["bcr", "earr", "earl", "bcl"],
                    ["earr", "nose", "earl"],
                ],
                "neighboring_frames": 0,
                "zone_vertices": {"arena": ["tl", "tr", "br", "bl"]},
                "zone_bools": [["arena", "nose"], ["arena", "headcentre"]],
            },
            "model": {"num_f_maps": 128, "feature_dim": 256},
            "ssl": {
                m: {
                    "num_f_maps": "model_features",
                }
                for m in ssl_names
            },
        }
    )
    project.run_episode("test")
    Project.remove_project("test_ssl")

if __name__ == "__main__":
    test_ssl([constructors[0]])
    test_ssl(constructors[:2])
    test_ssl(constructors[:3])
    test_ssl(constructors[3:4])