#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
"""
## Models

The `dlc2action.model.base_model.Model` abstract class inherits from `torch.nn.Module` but additionally
handles automatic integration
of SSL modules (see `dlc2action.ssl`) and enforces consistent input and output formats.
"""

__pdoc__ = {
    "ms_tcn.MS_TCN3.dump_patches": False,
    "ms_tcn.MS_TCN3.training": False,
    "ms_tcn_modules": False,
    "base_model.Model.dump_patches": False,
    "base_model.Model.training": False,
    "base_model.LoadedModel.dump_patches": False,
    "base_model.LoadedModel.training": False,
}
