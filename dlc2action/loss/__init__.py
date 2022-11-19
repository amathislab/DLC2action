#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
"""
## Losses

There is no dedicated loss class in `dlc2action`. Instead we use regular `torch.nn.Module` instances that take
prediction
and target as input and return loss value as output.
"""

from dlc2action.loss.contrastive import *
from dlc2action.loss.mse import *
from dlc2action.loss.ms_tcn import *

__pdoc__ = {
    "contrastive.NTXent.dump_patches": False,
    "contrastive.NTXent.training": False,
    "contrastive.TripletLoss.dump_patches": False,
    "contrastive.TripletLoss.training": False,
    "contrastive.CircleLoss.dump_patches": False,
    "contrastive.CircleLoss.training": False,
    "ms_tcn.MS_TCN_Loss.dump_patches": False,
    "ms_tcn.MS_TCN_Loss.training": False,
    "mse.MSE.dump_patches": False,
    "mse.MSE.training": False,
    "asymmetric_loss": False,
}
