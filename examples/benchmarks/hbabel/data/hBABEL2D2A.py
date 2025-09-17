#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from tqdm import tqdm
import joblib
import json


def get_start_stop(nums):
    # https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


def add_confusion(list_a):
    list_a = list(list_a)
    list_a.append(0)
    return list_a


def load_tar(path):
    return joblib.load(path)


def load_json(path):
    with open(path, "r") as f:
        a = json.load(f)
    return a


def get_nturgb():
    joint_names = {
        0: "Pelvis",
        12: "L_Hip",
        13: "L_Knee",
        14: "L_Ankle",
        15: "L_Foot",
        16: "R_Hip",
        17: "R_Knee",
        18: "R_Ankle",
        19: "R_Foot",
        1: "Spine1",
        20: "Spine3",
        2: "Neck",
        3: "Head",
        4: "L_Shoulder",
        5: "L_Elbow",
        6: "L_Wrist",
        7: "L_Hand",
        21: "L_HandTip",  # Not in SMPL
        22: "L_Thumb",  # Not in SMPL
        8: "R_Shoulder",
        9: "R_Elbow",
        10: "R_Wrist",
        11: "R_Hand",
        23: "R_HandTip",  # Not in SMPL
        24: "R_Thumb",  # Not in SMPL
    }

    return [joint_names[i] for i in range(len(joint_names))]


def create_split_file(train_seq, val_seq, test_seq, split_file_path="split_file.txt"):
    """Create DLC2Action split file"""
    with open(split_file_path, "w") as f:
        for header, sequences in zip(
            ["Training videos:\n", "Validation videos:\n", "Test videos:\n"],
            [train_seq, val_seq, test_seq],
        ):
            f.write(header)
            for seq in sequences:
                f.write(seq + "\n")


def convert_data(
    train_pose: dict,
    val_pose: dict,
    action_data: dict,
    split_infos: dict,
    output_path: str,
):

    keypoints_names = get_nturgb()
    pose_suffix = "_3D_poses.h5"
    behavior_suffix = "_annotations.pickle"
    individuals = ["individual0"]

    train_seq = split_infos["SubmissionTrain"]
    test_seq = split_infos["publicTest"]

    for seq_list in [test_seq, train_seq]:

        frame_map = action_data["frame_number_map"]
        label_array = action_data["label_array"]
        behavior_names = action_data["vocabulary"]

        for pose_data in [train_pose, val_pose]:
            for pose_seq in tqdm(pose_data):

                sequence = pose_seq["babel_id"]
                if not (sequence in frame_map.keys() and sequence in seq_list):
                    continue

                # Load data
                prefix = sequence
                pose_seq = pose_seq["joint_positions_processed"]
                start, stop = frame_map[sequence]
                action_seq = label_array[:, start:stop]

                # Convert pose data
                pose_seq = np.concatenate(
                    [pose_seq, np.ones((pose_seq.shape[0], pose_seq.shape[1], 1))],
                    axis=-1,
                )
                pose_seq = np.reshape(
                    pose_seq, (-1, pose_seq.shape[1] * pose_seq.shape[2])
                )
                columnindex = pd.MultiIndex.from_product(
                    [
                        ["hBABEL"],
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
                    "annotator": "hBABEL",
                    "video_file": video_file,
                }
                beh_data = []

                for behavior in action_seq:
                    beh1 = np.where(behavior)[0]
                    st_st = get_start_stop(beh1)
                    beh_data.append(
                        np.array(list(map(add_confusion, st_st)), dtype=np.int64)
                    )
                formated_data = (infos, behavior_names, individuals, [beh_data])
                filename = os.path.join(output_path, prefix + behavior_suffix)
                pickle.dump(
                    formated_data, open(os.path.join(output_path, filename), "wb")
                )

    # Create split file
    create_split_file(
        train_seq, test_seq, [], os.path.join(output_path, "split_file.txt")
    )

    return


if __name__ == "__main__":

    data_path = "path/to/original/data"

    all_actions = np.load(
        os.path.join(
            data_path,
            "mabe_format_seq",
            "babel_val_test_actions_val_top_120_60_filtered.npy",
        ),
        allow_pickle=True,
    ).item()

    val_pose = load_tar(
        os.path.join(
            data_path,
            "babel-smplh-30fps-male_split",
            "val_proc_realigned_procrustes.pth.tar",
        )
    )
    train_pose = load_tar(
        os.path.join(
            data_path,
            "babel-smplh-30fps-male_split",
            "train_proc_realigned_procrustes.pth.tar",
        )
    )

    split_infos = load_json(
        "path/to/split_info_BABEL_val_filtered.json"
    )  # TODO find this from hBehaveMAE

    # Create output folder
    output_path = os.path.join(data_path, "hBABEL_4D2A_traintest")
    os.makedirs(output_path, exist_ok=True)

    convert_data(train_pose, val_pose, all_actions, split_infos, output_path)
