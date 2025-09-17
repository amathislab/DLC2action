#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import os

import moviepy.editor as mpy
from tqdm import tqdm
from utils.data_load import Dataset
from utils.utils import *

# Load data infos
dataset_template = Dataset()
dataset_template.data_path

for game in dataset_template.game_names:
    print("Process videos for", game)
    # Create output path
    output_folder = os.path.join(dataset_template.converted_data_path, game, "videos")
    os.makedirs(output_folder, exist_ok=True)

    for subject in os.listdir(os.path.join(dataset_template.data_path, game)):
        print("Process ", subject)
        if (
            os.path.isdir(os.path.join(dataset_template.data_path, game, subject))
            and subject != "highscore"
        ):
            # Get list of frames and sort them
            list_of_frames = os.listdir(
                os.path.join(dataset_template.data_path, game, subject)
            )
            base_name = "_".join(list_of_frames[0].split("_")[:-1])
            list_of_frames_sorted = [
                os.path.join(
                    dataset_template.data_path, game, subject, base_name + f"_{i}.png"
                )
                for i in tqdm(range(1, len(list_of_frames)))
            ]

            # # Write video file
            clip = mpy.ImageSequenceClip(list_of_frames_sorted, fps=30)
            clip.write_videofile(os.path.join(output_folder, f"{subject}.mp4"), fps=30)

    break
