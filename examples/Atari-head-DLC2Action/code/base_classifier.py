#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm


def separate_split_data(split_data):

    chapters = 0
    training_list = []
    validation_list = []
    testing_list = []
    for elem in split_data:
        if chapters == 0:
            if "Validation" in elem:
                chapters += 1
                continue
            training_list.append(elem)

        elif chapters == 1:
            if "Testing" in elem:
                chapters += 1
                continue
            validation_list.append(elem)
        elif chapters == 2:
            testing_list.append(elem)

    validation_list = validation_list[:-1]
    training_list = training_list[1:-1]

    validation_list = [elem[:-2] for elem in validation_list]
    training_list = [elem[:-2] for elem in training_list]
    testing_list = [elem[:-2] for elem in testing_list]

    return training_list, validation_list, testing_list


def convert_to_frame_based(beh_data):
    max_temp = 0
    for individual in beh_data:
        frame_action = []
        for action in individual:
            max_temp = max(max_temp, np.max(action))
            frame_beh_data = np.zeros(np.max(action))
            for sequence in action:
                frame_beh_data[sequence[0] : sequence[1]] = 1
            frame_action.append(frame_beh_data)

    return frame_action, max_temp


def save_file(f1, indices, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(f1, index=indices)
    df.to_csv(output_path)


seg_len = 128
path_to_dlc2action_models = "path/to/dlc2action_models"
path_to_data = "path/to/converted_data"
for project in tqdm(
    [
        os.path.join(path_to_dlc2action_models, elem)
        for elem in os.listdir(path_to_dlc2action_models)
    ]
):
    if os.path.exists(os.path.join(project, "results", "splits")):
        for split_filename in os.listdir(os.path.join(project, "results", "splits")):
            if f"len{seg_len}" in split_filename:
                with open(
                    os.path.join(project, "results", "splits", split_filename)
                ) as f:
                    split_data = f.readlines()

                # Read the split files from DLC2Action
                training_list, validation_list, testing_list = separate_split_data(
                    split_data
                )

                # Read behavior files
                for game in os.listdir(path_to_data):
                    f1 = {str(float(i)): [] for i in range(18)}
                    indices = []
                    output_path = os.path.join(
                        path_to_dlc2action_models, "base_classifier", game
                    )
                    if os.path.basename(project).startswith(game):
                        for behavior_file in os.listdir(
                            os.path.join(path_to_data, game)
                        ):
                            for validation_file in validation_list:
                                if behavior_file.endswith(
                                    "action_behaviors.pickle"
                                ) and behavior_file.startswith(validation_file):
                                    with open(
                                        os.path.join(path_to_data, game, behavior_file),
                                        "rb",
                                    ) as f:
                                        beh_data = pickle.load(f)

                                    # Get max frame and convert to frame binary values
                                    frame_action, max_temp = convert_to_frame_based(
                                        beh_data[3]
                                    )
                                    vocab = beh_data[1]
                                    count = 0
                                    for i, key in enumerate(list(f1.keys())):
                                        # Get base classifier predictions for each action

                                        if vocab[count] == key:
                                            p = np.sum(frame_action[count]) / max_temp
                                            f1[key].append((2 * p / (1 + p)))
                                            if count < len(vocab) - 1:
                                                count += 1
                                        else:
                                            f1[key].append(np.nan)

                                    indices.append(validation_file)
                                    # Save in a file
                                    save_file(
                                        f1,
                                        indices,
                                        os.path.join(
                                            output_path, "base_classifier.csv"
                                        ),
                                    )
