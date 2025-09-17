#!bin/bash
##
## Copyright 2025-present by A. Mathis Group and contributors. All rights reserved.
##
##   This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.


# This script launches all the benchmarks in the examples/benchmarks directory.
# It is intended to be run from the root of the repository.
# run `bash ./examples/benchmarks/launch_all_benchmarks.sh`

# Run the benchmarks
echo "Running CRIM benchmark"
python examples/benchmarks/simba_crim.py
echo "Running RAT benchmark"
python examples/benchmarks/simba_rat.py
echo "Running EPM benchmark"
python examples/benchmarks/sturman_epm.py
echo "Running OFT benchmark"
python examples/benchmarks/sturman_oft.py
echo "Running Calms21 benchmark"
python examples/benchmarks/calms21.py
