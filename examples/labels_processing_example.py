#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#

# Process OFT and EPM labels CSV files to split them into individual files based on the 'DLCFile' column.

import os
import re

import pandas as pd


def split_annotations_by_column(
    csv_path,
    group_column="DLCFile",
    data_suffix="DeepCut_resnet50_Blockcourse1May9shuffle1_1030000.csv",
    output_dir="split_annotations",
):
    """
    Split a master CSV file into separate files grouped by a specified column.

    Parameters:
    -----------
    csv_path : str
        Path to the master CSV file
    group_column : str
        Column name to group by (default: "DLCFile")
    data_suffix : str
        Suffix to remove from the grouped column value when naming output files
    output_dir : str
        Directory to save the output files (will be created if it doesn't exist)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Read the CSV file (with semicolon separator)
    df = pd.read_csv(csv_path, sep=";")

    # Get unique values in the group column
    unique_values = df[group_column].unique()
    print(f"Found {len(unique_values)} unique {group_column} values")

    # Process each unique value
    for value in unique_values:
        # Extract the subset of data for this value
        subset = df[df[group_column] == value]

        # Create a clean filename by removing the data_suffix
        clean_name = value
        if data_suffix and clean_name.endswith(data_suffix):
            clean_name = clean_name[: -len(data_suffix)]

        # Remove any remaining special characters for the filename
        clean_name = re.sub(r"[^\w\-]", "", clean_name)

        # Create the output path
        output_path = os.path.join(output_dir, f"{clean_name}.csv")

        # Save the subset to a new CSV file
        subset.to_csv(output_path, sep=",", index=False)
        print(f"Saved {len(subset)} rows to {output_path}")

    print(f"Split complete! {len(unique_values)} files created in {output_dir}")


# Example usage
if __name__ == "__main__":
    split_annotations_by_column(
        csv_path="path/to/AllLabDataOFT_final.csv",
        group_column="DLCFile",
        data_suffix="DeepCut_resnet50_Blockcourse1May9shuffle1_1030000.csv",
        output_dir="examples/OFT/OFT/Labels/",
    )
