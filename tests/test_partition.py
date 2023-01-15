#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
from dlc2action.data.dataset import BehaviorDataset
import pytest
import os
import shutil

path = os.path.join(os.path.dirname(__file__), "data")

data_parameters = {
    "data_type": "dlc_track",
    "annotation_type": "csv",
    "data_path": path,
    "annotation_path": path,
    "annotation_suffix": {"2.csv"},
    "overlap": 0.7,
    "data_suffix": {
        "2DeepCut_resnet50_Blockcourse1May9shuffle1_1030000.csv",
    },
}


@pytest.mark.parametrize("method", ["random"])
def test_partition(method: str):
    """
    Test the `dlc2action.data.dataset.BehaviorDataset.partition_train_test_val` function

    Check the sizes of the resulting datasets + make sure they stay the same when reading from a split file
    """

    folder = os.path.join(path, "trimmed")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    dataset = BehaviorDataset(**data_parameters)
    train_dataset, test_dataset, val_dataset = dataset.partition_train_test_val(
        method=method,
        val_frac=0.3,
        test_frac=0.05,
        split_path="split_tmp.txt",
        save_split=True,
    )
    assert len(test_dataset) < len(val_dataset)
    assert len(val_dataset) < len(train_dataset)
    assert len(train_dataset) < len(dataset)
    train_len = len(train_dataset)
    val_len = len(val_dataset)
    test_len = len(test_dataset)
    train_dataset, test_dataset, val_dataset = dataset.partition_train_test_val(
        method="file", split_path="split_tmp.txt"
    )
    assert len(train_dataset) == train_len
    assert len(test_dataset) == test_len
    assert len(val_dataset) == val_len
    os.remove("split_tmp.txt")
    folder = os.path.join(path, "trimmed")
    if os.path.exists(folder):
        shutil.rmtree(folder)


# test_partition("random")
