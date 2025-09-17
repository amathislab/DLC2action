#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import os
import shutil
import pytest
from dlc2action.data.dataset import BehaviorDataset
import yaml

# Load config and set paths
with open("tests/config_test.yaml", "r") as f:
    config = yaml.safe_load(f)
oft_data_path = config["oft_data_path"]
output_path = "tests/data/tmp"


def make_temp_folders(path, data_suffix, annotation_suffix, output_path=output_path):
    """Create temporary folders for testing and copy data files into them."""
    folders = ["train", "val", "test"]
    for folder in folders:
        os.makedirs(os.path.join(output_path, folder), exist_ok=True)
    count = 0
    for filename in os.listdir(path):
        if filename.endswith(data_suffix):
            stem = filename.split(data_suffix)[0]
            annotation_file = stem + annotation_suffix
            assert os.path.exists(
                os.path.join(path, annotation_file)
            ), f"Annotation file {annotation_file} does not exist in {path}"
            for f in [filename, annotation_file]:
                shutil.copy(
                    os.path.join(path, f),
                    os.path.join(output_path, folders[count % 3], f),
                )
            count += 1


data_parameters = dict(
    data_type="dlc_track",
    annotation_type="csv",
    data_path=oft_data_path,
    annotation_path=oft_data_path,
    annotation_suffix=".csv",
    data_suffix={"DeepCut_resnet50_Blockcourse1May9shuffle1_1030000.csv"},
    behaviors=["Grooming", "Supported", "Unsupported"],
)


@pytest.mark.parametrize("method", ["random", "time", "time:strict", "folders"])
def test_partition(method: str):
    """
    Test the `dlc2action.data.dataset.BehaviorDataset.partition_train_test_val` function

    Check the sizes of the resulting datasets + make sure they stay the same when reading from a split file
    """
    path = oft_data_path
    params = data_parameters.copy()
    if method == "folders":
        splits = ["train", "val", "test"]
        params["data_path"] = [os.path.join(output_path, s) for s in splits]
        params["annotation_path"] = [os.path.join(output_path, s) for s in splits]
        path = output_path
        make_temp_folders(
            oft_data_path,
            data_suffix="DeepCut_resnet50_Blockcourse1May9shuffle1_1030000.csv",
            annotation_suffix=".csv",
            output_path=output_path,
        )
    val_frac = 0.5 if method == "random" else 0.3
    split_path = None if "time" in method else "split_tmp.txt"
    save_split = False if "time" in method else True

    dataset = BehaviorDataset(**params)
    train_dataset, test_dataset, val_dataset = dataset.partition_train_test_val(
        method=method,
        val_frac=val_frac,
        test_frac=0.05,
        split_path=split_path,
        save_split=save_split,
    )

    if "time" in method:
        print(len(dataset), len(train_dataset), len(val_dataset), len(test_dataset))
        assert len(test_dataset) < len(val_dataset) < len(train_dataset) < len(dataset)
    else:
        train_len, val_len, test_len = map(
            len, (train_dataset, val_dataset, test_dataset)
        )
        train_dataset, test_dataset, val_dataset = dataset.partition_train_test_val(
            method="file", split_path="split_tmp.txt"
        )
        assert (len(train_dataset), len(test_dataset), len(val_dataset)) == (
            train_len,
            test_len,
            val_len,
        )
        os.remove("split_tmp.txt")

    folder = os.path.join(path, "trimmed")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

if __name__ == "__main__":
    test_partition("random")
    test_partition("time")
    test_partition("time:strict")
    test_partition("folders")