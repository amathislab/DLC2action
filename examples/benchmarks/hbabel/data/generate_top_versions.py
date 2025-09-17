#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import pickle
import os
import numpy as np

DATA_PATH = "path/to/converted_data"
SPLIT_PATH = "path/to/split_file.txt"


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def save_split_file(filename_list, hierarchy, top, output_path):
    with open(SPLIT_PATH, "r") as f:
        lines = f.readlines()
    filename_list = filename_list + ["Training videos:", "Validation videos:", "Test videos:"]
    new_lines = [line for line in lines if line.split("\n")[0] in filename_list]
    with open(os.path.join(output_path, f"split_file_{hierarchy}_top_{top}.txt"), "w") as f:
        f.writelines(new_lines)


def create_dataset(hierarchy, top):
    output_path = os.path.join(os.path.dirname(DATA_PATH),"hBABEL_4D2A_tops")
    os.makedirs(output_path, exist_ok=True)
    filename_list = []
    for filename in os.listdir(DATA_PATH):
        if not filename.endswith(".pickle"):
            continue
        data = load_pickle(os.path.join(DATA_PATH, filename))

        if hierarchy == "seg":
            assert top <= 60
            new_dat = data[3][0][60-top:60]
            vocab = data[1][60-top:60]
        elif hierarchy == "frame":
            assert top <= 120
            new_dat = data[3][0][180-top:]
            vocab = data[1][180-top:]
        if np.sum([len(x) for x in new_dat]) == 0:
            continue
        filename_list.append(filename.split("_annotations")[0])
        data[3][0] = new_dat
        data = (data[0], vocab, data[2], data[3])

        save_pickle(data, os.path.join(output_path, os.path.splitext(filename)[0] + f"_{hierarchy}_top_{top}.pickle"))
        save_split_file(filename_list, hierarchy, top, output_path)

if __name__ == "__main__":
    for hierarchy in ["seg", "frame"]:
        for top in [10,30,60]:
            create_dataset(hierarchy, top)
    create_dataset("frame", 90)
