#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd


def get_start_stop(nums):
    # https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
    # nums = sorted(set(nums))
    # gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    # edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    # return list(zip(edges, edges))

    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


def polar(x, y, img_size):
    """returns r, theta(degrees)"""
    x_new = img_size[0] / 2 - x
    y_new = img_size[1] / 2 - y
    r = (x_new**2 + y_new**2) ** 0.5
    theta = np.arctan2(y_new, x_new)
    return r, theta


def add_confusion(list_a):
    list_a = list(list_a)
    list_a.append(0)
    return list_a


def read_frame_map(path_to_frame_map):
    with open(path_to_frame_map, "rb") as f:
        a = pickle.load(f)
    return a


def write_frame_map(frame_map, path_to_frame_map):
    with open(path_to_frame_map, "wb") as f:
        pickle.dump(frame_map, f)


def remap_st_st(cum_frame_map, st_st):
    remaped_st_st = [
        (cum_frame_map[interval[0]], cum_frame_map[interval[1]]) for interval in st_st
    ]
    return remaped_st_st


def save_confusion(confusion_matrix, classes, output_path):
    data = {"confusion": confusion_matrix, "classes": classes}
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
