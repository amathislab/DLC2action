#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import click
from dlc2action.project import Project


@click.option("--name", "-n", required=True, type=str, help="name of the project")
@click.option("--data_type", "-d", required=True, type=str, help="data type")
@click.option(
    "--data_path",
    "-dp",
    required=True,
    type=str,
    help="path to the folder containing " "input files for the project",
    multiple=True,
)
@click.option(
    "--annotation_path",
    "-ap",
    required=True,
    type=str,
    help="path to the folder containing " "annotation files for the project",
    multiple=True,
)
@click.option(
    "--annotation_type",
    "-a",
    required=False,
    default="none",
    type=str,
    help="annotation type",
)
@click.option(
    "--projects_path",
    "-pp",
    required=False,
    type=str,
    help="path to the projects folder " "(is filled with ~/DLC2Action by default)",
)
@click.option(
    "--copy",
    is_flag=True,
    help="if the flag is used, the files from annotation_path and data_path will be "
    "copied to the projects folder",
)
@click.command()
def main(
    name, data_type, annotation_type, projects_path, data_path, annotation_path, copy
):
    """
    Start a new project
    """

    Project(
        name,
        data_type,
        annotation_type,
        projects_path,
        data_path,
        annotation_path,
        copy,
    )


if __name__ == "__main__":
    main()
