#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import click
from dlc2action.project import Project


@click.option("--name", "-n", required=True, type=str, help="name of the project")
@click.option(
    "--projects_path",
    "-pp",
    required=False,
    type=str,
    help="path to the projects folder",
)
@click.option(
    "--all",
    is_flag=True,
    help="if the flag is used, all parameters are prompted; otherwise only those "
    'that have blank ("???") values',
)
@click.command()
def main(name, projects_path, all):
    """
    Fill in project parameters interactively
    """

    project = Project(name, projects_path=projects_path)


if __name__ == "__main__":
    main()
