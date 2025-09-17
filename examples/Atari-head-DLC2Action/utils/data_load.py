#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import os

import imageio
import numpy as np
import pandas as pd


class Dataset:
    def __init__(
        self,
        game=None,
        data_path="/media1/data/andy/Atari_project/Atari-Head",
        converted_data_path="/media1/data/andy/Atari_project/Atari-Head-4DLC2Action",
        projects_path="/media1/data/andy/Atari_project/Atari-Head-D2A-projects",
    ) -> None:
        self.data_path = data_path
        self.converted_data_path = converted_data_path
        self.projects_path = projects_path

        if os.path.exists(self.data_path):
            self.game_names = [
                elem
                for elem in os.listdir(self.data_path)
                if os.path.isdir(os.path.join(self.data_path, elem))
            ]
        else:
            self.game_names = [
                elem
                for elem in os.listdir(self.converted_data_path)
                if os.path.isdir(os.path.join(self.converted_data_path, elem))
            ]
        self.choose_game(game)
        self.read_meta_data()
        self.img_size = (160, 210)

    def choose_game(self, game):
        """Set game to dataset class"""
        if game in self.game_names or game is None:
            self.game = game
            if game is not None:
                self.get_subject_list()
                self.i3d_path = os.path.join(self.converted_data_path, game, "i3d")
                self.game_path = os.path.join(self.data_path, self.game)
        else:
            raise FileNotFoundError("The game is not in the list of available games")

    def get_subject_list(self):
        """Get list of subject"""
        self.subject_list = [
            elem
            for elem in os.listdir(os.path.join(self.data_path, self.game))
            if (
                os.path.isdir(os.path.join(self.data_path, self.game, elem))
                and elem != "highscore"
            )
        ]

    def read_subject_info(self, subject_name):
        """Read data relative to a specific subject"""
        txt_filename = os.path.join(self.game_path, subject_name + ".txt")

        with open(txt_filename, "r") as f:
            for i, line in enumerate(f.readlines()):
                if not i:
                    header = line.split(",")
                    header[-1] = header[-1].split("\n")[0]
                    data_info_dict = dict.fromkeys(header)
                    for key in data_info_dict.keys():
                        data_info_dict[key] = []
                else:
                    data_info = line.split(",")
                    data_info = [
                        elem if "null" not in elem else np.nan for elem in data_info
                    ]
                    for j, key in enumerate(list(data_info_dict.keys())):
                        if not j:
                            data_info_dict[key].append(str(data_info[j]))
                        elif j == len(list(data_info_dict.keys())) - 1:
                            data_info_dict[key].append(
                                np.array(data_info[j:], dtype=np.float32)
                            )
                        else:
                            data_info_dict[key].append(float(data_info[j]))
            data_info_dict["subject_name"] = subject_name
        return data_info_dict

    def read_meta_data(self):
        """Read meta data file, returns dataframe"""
        if os.path.exists(self.data_path):
            path_to_meta = os.path.join(self.data_path, "meta_data.csv")
            self.meta_df = pd.read_csv(path_to_meta)
        else:
            print("No meta data found")

    def fetch_frame(self, subject_name, frame_name):
        img_dir = os.path.join(self.data_path, self.game, subject_name)
        if not frame_name.endswith(".png"):
            frame_name = frame_name + ".png"
        img = imageio.imread(os.path.join(img_dir, frame_name))
        return img


if __name__ == "__main__":
    dataset = Dataset(game="enduro")
    # dataset.choose_game("enduro")
    print(dataset.subject_list)
    data_info = dataset.read_subject_info(dataset.subject_list[0])

    # Think about the overall structure pre training dataset so that you can load easily the features
    # Read gaze
    # Read score
    # Read action -> Check how actions are stored
    # Make another file to convert into DLC poses -> generate once and add a loading function
    # Check synchronization of frames, poses, actions
