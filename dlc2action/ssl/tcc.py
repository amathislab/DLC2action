#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
from dlc2action.ssl.base_ssl import SSLConstructor
import torch
from dlc2action.loss.tcc import _TCCLoss
from typing import Dict, Union, Tuple
from torch import nn
from dlc2action.ssl.modules import _FC


class TCCSSL(SSLConstructor):
    """
    A contrastive SSL class with an NT-Xent loss

    The SSL input and target are left empty (the SSL input is generated as an augmentation of the
    input sample at runtime).
    """

    type = "ssl_target"

    def __init__(
        self,
        num_f_maps: torch.Size,
        len_segment: int,
        projection_head_f_maps: int = None,
        loss_type: str = "regression_mse_var",
        variance_lambda: float = 0.001,
        normalize_indices: bool = True,
        normalize_embeddings: bool = False,
        similarity_type: str = "l2",
        num_cycles: int = 20,
        cycle_length: int = 2,
        temperature: float = 0.1,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        if len(num_f_maps) > 1:
            raise RuntimeError(
                "The TCC constructor expects the input data to be 2-dimensional; "
                f"got {len(num_f_maps) + 1} dimensions"
            )
        num_f_maps = int(num_f_maps[0])
        if projection_head_f_maps is None:
            projection_head_f_maps = num_f_maps
        self.len_segment = int(len_segment)
        self.loss_function = _TCCLoss(
            loss_type,
            variance_lambda,
            normalize_indices,
            normalize_embeddings,
            similarity_type,
            int(num_cycles),
            int(cycle_length),
            temperature,
            label_smoothing,
        )
        self.pars = {
            "dim": int(num_f_maps),
            "num_f_maps": int(num_f_maps),
            "num_ssl_layers": 1,
            "num_ssl_f_maps": int(projection_head_f_maps),
        }

    def transformation(self, sample_data: Dict) -> Tuple:
        """
        Empty transformation
        """

        mask = torch.ones((1, self.len_segment))
        for key, value in sample_data.items():
            mask *= (torch.sum(value, 0) == 0).unsqueeze(0)
        mask = 1 - mask
        return torch.tensor(float("nan")), {"loaded": mask}

    def loss(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        loss = self.loss_function(predicted, target)
        return loss

    def construct_module(self) -> Union[nn.Module, None]:

        if self.pars["num_ssl_f_maps"] is None:
            module = nn.Identity()
        else:
            module = _FC(**self.pars)
        return module
