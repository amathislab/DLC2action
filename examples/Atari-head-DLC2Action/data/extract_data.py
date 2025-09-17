#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import os

from utils import Dataset

dataset = Dataset()

for game_folder in [
    os.path.join(dataset.data_path, elem) for elem in dataset.game_names
]:
    for filename in [
        os.path.join(game_folder, elem) for elem in os.listdir(game_folder)
    ]:
        if filename.endswith(".tar.bz2"):
            print("Extract ", filename)
            os.system(f"tar -xvf {filename} --directory {os.path.dirname(filename)}")

    rm_str = os.path.join(game_folder, "*.tar.bz2")
    os.system(f"rm {rm_str}")
