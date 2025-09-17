#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm


def get_start_stop(nums):
    # https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


def add_confusion(list_a):
    """Add a confusion of 0 at the end of the list"""
    list_a = list(list_a)
    list_a.append(0)
    return list_a


def convert_data(
    pose_data: dict, action_data: dict, output_path: str, mode: str = "test"
):
    """Convert SHOT data to DLC2Action format
    both for the behaviors and the pose"""

    pose_suffix = "_3D_poses.h5"
    behavior_suffix = "_annotations.pickle"
    individuals = ["individual0"]

    frame_map = action_data["frame_number_map"]
    label_array = action_data["label_array"]
    behavior_names = action_data["vocabulary"]
    start_episode = 0 if mode == "test" else 68
    for sequence in tqdm(list(pose_data["sequences"]["keypoints"].keys())):

        # Load data
        prefix = "_".join(
            sequence.split("_")[:-1]
            + [str(int(sequence.split("_")[-1]) + start_episode)]
        )
        pose_seq = pose_data["sequences"]["keypoints"][sequence]
        start, stop = frame_map[sequence]
        action_seq = label_array[:, start:stop]

        # Convert pose data
        pose_seq = np.concatenate(
            [
                pose_seq,
                np.ones((pose_seq.shape[0], pose_seq.shape[1], pose_seq.shape[2], 1)),
            ],
            axis=-1,
        )
        pose_seq = np.reshape(
            pose_seq, (-1, pose_seq.shape[1] * pose_seq.shape[2] * pose_seq.shape[3])
        )
        columnindex = pd.MultiIndex.from_product(
            [
                ["SHOT72"],
                individuals,
                keypoints_names,
                ["x", "y", "z", "likelihood"],
            ],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )
        df_list = pd.DataFrame(pose_seq, columns=columnindex)
        output_filename = os.path.join(output_path, prefix + pose_suffix)
        df_list.to_hdf(output_filename, key="tracks", format="table", mode="w")

        # Convert behavioral data
        video_file = prefix + ".mp4"
        infos = {
            "datetime": str(datetime.now()),
            "annotator": "SHOT72",
            "video_file": video_file,
        }
        beh_data = []

        for behavior in action_seq:
            beh1 = np.where(behavior)[0]
            st_st = get_start_stop(beh1)
            beh_data.append(np.array(list(map(add_confusion, st_st)), dtype=np.int64))
        formated_data = (infos, behavior_names, individuals, [beh_data])
        filename = os.path.join(output_path, prefix + behavior_suffix)
        pickle.dump(formated_data, open(os.path.join(output_path, filename), "wb"))

    return


if __name__ == "__main__":

    split = "split_32_68"
    path_to_shot = "path/to/original/data"

    path_to_shot32_test = os.path.join(path_to_shot, f"{split}/test")
    path_to_shot32_train = os.path.join(path_to_shot, f"{split}/train")

    test_pose = np.load(
        os.path.join(path_to_shot32_test, "test_dictionary_poses.npy"),
        allow_pickle=True,
    ).item()
    train_pose = np.load(
        os.path.join(path_to_shot32_train, "train_dictionary_poses.npy"),
        allow_pickle=True,
    ).item()

    test_actions = np.load(
        os.path.join(path_to_shot32_test, f"converted_dict_labels_binary_SHOT72.npy"),
        allow_pickle=True,
    ).item()
    train_actions = np.load(
        os.path.join(path_to_shot32_train, "train_dictionary_actions.npy"),
        allow_pickle=True,
    ).item()

    output_path = os.path.join(path_to_shot, f"SHOT72_4D2A_traintest")
    os.makedirs(output_path, exist_ok=True)

    keypoints_names = [
        "center",
        "l_hip",
        "l_knee",
        "l_ankle",
        "l_foot",
        "l_toes",
        "r_hip",
        "r_knee",
        "r_ankle",
        "r_foot",
        "r_toes",
        "lumbars",
        "low_thorax",
        "high_thorax",
        "cervicals",
        "l_shoulder_blade",
        "l_shoulder",
        "l_elbow",
        "l_wrist",
        "neck",
        "head",
        "head_top",
        "r_shoulder_blade",
        "r_shoulder",
        "r_elbow",
        "r_wrist",
    ]

    convert_data(test_pose, test_actions, output_path, mode="test")
    convert_data(train_pose, train_actions, output_path, mode="train")
