#
# Copyright 2025-present by A. Mathis Group and contributors. All rights reserved.
#
#   This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
"""Apply copyright headers to all code files in the repository.

This file can be called as a python script without arguments. For
configuration, see the instructions in NOTICE.yml.

Thanks @stes - https://github.com/DeepLabCut/DeepLabCut/blob/master/tools/update_license_headers.py
"""

import tempfile
import glob
import yaml
import fnmatch
import subprocess


def load_config(filename):
    with open(filename, "r") as fh:
        config = yaml.safe_load(fh)
    return config


def walk_directory(entry):
    """Talk the directory"""

    if "header" not in entry:
        raise ValueError("Current entry does not have a header.")
    if "include" not in entry:
        raise ValueError("Current entry does not have an include list.")

    def _list_include():
        """List all files specified in the include list."""
        for include_pattern in entry["include"]:
            for filename in glob.iglob(include_pattern, recursive=True):
                yield filename

    def _filter_exclude(iterable):
        """Filter filenames from an iterator by the exclude patterns."""
        for filename in iterable:
            for exclude_pattern in entry.get("exclude", []):
                if fnmatch.fnmatch(filename, exclude_pattern):
                    break
            else:
                yield filename

    files = _filter_exclude(set(_list_include()))
    return list(files)


def main(input_file="NOTICE.yml"):
    config = load_config(input_file)
    for entry in config:
        filelist = list(walk_directory(entry))
        with tempfile.NamedTemporaryFile(mode="w") as header_file:
            header_file.write(entry["header"])
            header_file.flush()
            header_file.seek(0)
            command = ["licenseheaders", "-t", str(header_file.name), "-f"] + filelist
            result = subprocess.run(command, capture_output=True)
            if result.returncode != 0:
                print(result.stdout.decode())
                print(result.stderr.decode())


if __name__ == "__main__":
    main()
