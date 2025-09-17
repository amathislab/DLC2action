#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from dlc2action.model.asformer import AttLayer


class Refinement(nn.Module):
    """
    Refinement module
    """

    def __init__(
        self,
        num_layers,
        num_f_maps_input,
        num_f_maps,
        dim,
        num_classes,
        dropout_rate,
        direction,
        skip_connections,
        attention="none",
        block_size=0,
    ):
        """
        Parameters
        ----------
        num_layers : int
            the number of layers
        num_f_maps : int
            the number of feature maps
        dim : int
            the number of features in input
        num_classes : int
            the number of target classes
        dropout_rate : float
            dropout rate
        direction : [None, 'forward', 'backward']
            the direction of convolutions; if None, regular convolutions are used
        skip_connections : bool
            if `True`, skip connections are added
        block_size : int, default 0
            if not 0, skip connections are added to the prediction generation stage with this interval
        """

        super(Refinement, self).__init__()
        self.block_size = block_size
        self.direction = direction
        if skip_connections:
            self.conv_1x1 = nn.Conv1d(dim + num_f_maps_input, num_f_maps, 1)
        else:
            self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [
                copy.deepcopy(
                    DilatedResidualLayer(
                        dilation=2**i,
                        in_channels=num_f_maps,
                        out_channels=num_f_maps,
                        dropout_rate=dropout_rate,
                        causal=(direction is not None),
                    )
                )
                for i in range(num_layers)
            ]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.attention_layers = nn.ModuleList([])
        self.attention = attention
        if self.attention == "basic":
            self.attention_layers += nn.ModuleList(
                [
                    nn.Conv1d(num_f_maps, num_f_maps, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(num_f_maps, num_f_maps, 3, padding=1),
                    nn.Sigmoid(),
                ]
            )

    def forward(self, x):
        """Forward pass."""
        x = self.conv_1x1(x)
        out = copy.copy(x)
        for l_i, layer in enumerate(self.layers):
            out = layer(out, self.direction)
            if self.block_size != 0 and (l_i + 1) % self.block_size == 0:
                out = out + x
                x = copy.copy(out)
        if self.attention != "none":
            x = copy.copy(out)
            for layer in self.attention_layers:
                x = layer(x)
            out = out * x
        out = self.conv_out(out)
        return out


class Refinement_SE(nn.Module):
    """
    Refinement module
    """

    def __init__(
        self,
        num_layers,
        num_f_maps_input,
        num_f_maps,
        dim,
        num_classes,
        dropout_rate,
        direction,
        skip_connections,
        len_segment,
        block_size=0,
    ):
        """
        Parameters
        ----------
        num_layers : int
            the number of layers
        num_f_maps : int
            the number of feature maps
        dim : int
            the number of features in input
        num_classes : int
            the number of target classes
        dropout_rate : float
            dropout rate
        direction : [None, 'forward', 'backward']
            the direction of convolutions; if None, regular convolutions are used
        skip_connections : bool
            if `True`, skip connections are added
        block_size : int, default 0
            if not 0, skip connections are added to the prediction generation stage with this interval
        """

        super().__init__()
        self.block_size = block_size
        self.direction = direction
        if skip_connections:
            self.conv_1x1 = nn.Conv1d(dim + num_f_maps_input, num_f_maps, 1)
        else:
            self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [
                DilatedResidualLayer(
                    dilation=2**i,
                    in_channels=num_f_maps,
                    out_channels=num_f_maps,
                    dropout_rate=dropout_rate,
                    causal=(direction is not None),
                )
                for i in range(num_layers)
            ]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.fc1_f = nn.ModuleList(
            [nn.Linear(num_f_maps, num_f_maps // 2) for _ in range(6)]
        )
        self.fc2_f = nn.ModuleList(
            [nn.Linear(num_f_maps // 2, num_f_maps) for _ in range(6)]
        )

    def _fc1_f(self, tag):
        if tag is None:
            for i in range(1, len(self.fc1_f)):
                self.fc1_f[i].load_state_dict(self.fc1_f[0].state_dict())
            return self.fc1_f[0]
        else:
            return self.fc1_f[tag]

    def _fc2_f(self, tag):
        if tag is None:
            for i in range(1, len(self.fc2_f)):
                self.fc2_f[i].load_state_dict(self.fc2_f[0].state_dict())
            return self.fc2_f[0]
        else:
            return self.fc2_f[tag]

    def forward(self, x, tag):
        """Forward pass."""
        x = self.conv_1x1(x)
        scale = torch.mean(x, -1)
        scale = self._fc2_f(tag)(F.relu(self._fc1_f(tag)(scale)))
        scale = F.sigmoid(scale).unsqueeze(-1)
        x = scale * x
        out = copy.copy(x)
        for l_i, layer in enumerate(self.layers):
            out = layer(out, self.direction)
            if self.block_size != 0 and (l_i + 1) % self.block_size == 0:
                out = out + x
                x = copy.copy(out)
        out = self.conv_out(out)
        return out


class RefinementB(Refinement):
    """
    Bidirectional refinement module
    """

    def forward(self, x):
        """Forward pass."""
        x_f = self.conv_1x1(x)
        x_b = copy.copy(x_f)
        forward = copy.copy(x_f)
        backward = copy.copy(x_f)
        for i, layer_f in enumerate(self.layers):
            forward = layer_f(forward, "forward")
            backward = layer_f(backward, "backward")
            if self.block_size != 0 and (i + 1) % self.block_size == 0:
                forward = forward + x_f
                backward = backward + x_b
                x_f = copy.copy(forward)
                x_b = copy.copy(backward)
        out = torch.cat([forward, backward], 1)
        out = self.conv_out(out)
        return out


class SimpleResidualLayer(nn.Module):
    """
    Basic residual layer
    """

    def __init__(self, num_f_maps, dropout_rate):
        """
        Parameters
        ----------
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        dropout_rate : float
            dropout rate
        """

        super().__init__()
        self.conv_1x1_in = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.conv_1x1 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """Forward pass."""
        out = self.conv_1x1_in(x)
        out = F.relu(out)
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class DilatedResidualLayer(nn.Module):
    """
    Dilated residual layer
    """

    def __init__(self, dilation, in_channels, out_channels, dropout_rate, causal):
        """
        Parameters
        ----------
        dilation : int
            dilation
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        dropout_rate : float
            dropout rate
        causal : bool
            if `True`, causal convolutions are used
        """

        super(DilatedResidualLayer, self).__init__()
        self.padding = dilation * 2
        self.causal = causal
        if self.causal:
            padding = 0
        else:
            padding = dilation
        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, 3, padding=padding, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, direction=None):
        """Forward pass."""
        if direction is not None and not self.causal:
            raise ValueError("Cannot set direction in a non-causal layer!")
        elif direction is None and self.causal:
            direction = "forward"
        if direction == "forward":
            padding = (0, self.padding)
        elif direction == "backward":
            padding = (self.padding, 0)
        elif direction is not None:
            raise ValueError(
                f"Unrecognized direction: {direction}, please choose from"
                f'"backward", "forward" and None'
            )
        if direction is not None:
            out = self.conv_dilated(F.pad(x, padding))
        else:
            out = self.conv_dilated(x)
        out = F.relu(out)
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class DilatedResidualLayer_SE(nn.Module):
    """
    Dilated residual layer
    """

    def __init__(
        self, dilation, in_channels, out_channels, dropout_rate, causal, len_segment
    ):
        """
        Parameters
        ----------
        dilation : int
            dilation
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        dropout_rate : float
            dropout rate
        causal : bool
            if `True`, causal convolutions are used
        """

        super().__init__()
        self.padding = dilation * 2
        self.causal = causal
        if self.causal:
            padding = 0
        else:
            padding = dilation
        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, 3, padding=padding, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1_f = nn.ModuleList(
            [nn.Linear(out_channels, out_channels // 2) for _ in range(6)]
        )
        self.fc2_f = nn.ModuleList(
            [nn.Linear(out_channels // 2, out_channels) for _ in range(6)]
        )
        # self.fc1_t = nn.ModuleList([nn.Linear(len_segment, len_segment // 2) for _ in range(6)])
        # self.fc2_t = nn.ModuleList([nn.Linear(len_segment // 2, len_segment) for _ in range(6)])

    def _fc1_f(self, tag):
        if tag is None:
            for i in range(1, len(self.fc1_f)):
                self.fc1_f[i].load_state_dict(self.fc1_f[0].state_dict())
            return self.fc1_f[0]
        else:
            return self.fc1_f[tag]

    def _fc2_f(self, tag):
        if tag is None:
            for i in range(1, len(self.fc2_f)):
                self.fc2_f[i].load_state_dict(self.fc2_f[0].state_dict())
            return self.fc2_f[0]
        else:
            return self.fc2_f[tag]

    def _fc1_t(self, tag):
        if tag is None:
            for i in range(1, len(self.fc1_t)):
                self.fc1_t[i].load_state_dict(self.fc1_t[0].state_dict())
            return self.fc1_t[0]
        else:
            return self.fc1_t[tag]

    def _fc2_t(self, tag):
        if tag is None:
            for i in range(1, len(self.fc2_t)):
                self.fc2_t[i].load_state_dict(self.fc2_t[0].state_dict())
            return self.fc2_t[0]
        else:
            return self.fc2_t[tag]

    def forward(self, x, direction, tag):
        """Forward pass."""
        if direction is not None and not self.causal:
            raise ValueError("Cannot set direction in a non-causal layer!")
        elif direction is None and self.causal:
            direction = "forward"
        if direction == "forward":
            padding = (0, self.padding)
        elif direction == "backward":
            padding = (self.padding, 0)
        elif direction is not None:
            raise ValueError(
                f"Unrecognized direction: {direction}, please choose from"
                f'"backward", "forward" and None'
            )
        if direction is not None:
            out = self.conv_dilated(F.pad(x, padding))
        else:
            out = self.conv_dilated(x)
        out = F.relu(out)
        out = self.conv_1x1(out)
        out = self.dropout(out)
        scale = torch.mean(out, -1)
        scale = self._fc2_f(tag)(F.relu(self._fc1_f(tag)(scale)))
        scale = F.sigmoid(scale).unsqueeze(-1)
        # time_scale = torch.mean(out, 1)
        # time_scale = self._fc2_t(tag)(F.relu(self._fc1_t(tag)(time_scale)))
        # time_scale = F.sigmoid(time_scale).unsqueeze(1)
        out = out * scale  # * time_scale
        return x + out


class DilatedResidualLayer_SEC(nn.Module):
    """
    Dilated residual layer
    """

    def __init__(
        self, dilation, in_channels, out_channels, dropout_rate, causal, len_segment
    ):
        """
        Parameters
        ----------
        dilation : int
            dilation
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        dropout_rate : float
            dropout rate
        causal : bool
            if `True`, causal convolutions are used
        """

        super().__init__()
        self.padding = dilation * 2
        self.causal = causal
        if self.causal:
            padding = 0
        else:
            padding = dilation
        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, 3, padding=padding, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv_sq = nn.ModuleList(
            [
                nn.Conv1d(out_channels, 1, 3, padding=padding, dilation=dilation)
                for _ in range(6)
            ]
        )

    def _conv_sq(self, tag):
        if tag is None:
            for i in range(1, len(self.conv_sq)):
                self.conv_sq[i].load_state_dict(self.conv_sq[0].state_dict())
            return self.conv_sq[0]
        else:
            return self.conv_sq[tag]

    def forward(self, x, direction, tag):
        """Forward pass."""
        if direction is not None and not self.causal:
            raise ValueError("Cannot set direction in a non-causal layer!")
        elif direction is None and self.causal:
            direction = "forward"
        if direction == "forward":
            padding = (0, self.padding)
        elif direction == "backward":
            padding = (self.padding, 0)
        elif direction is not None:
            raise ValueError(
                f"Unrecognized direction: {direction}, please choose from"
                f'"backward", "forward" and None'
            )
        if direction is not None:
            out = self.conv_dilated(F.pad(x, padding))
        else:
            out = self.conv_dilated(x)
        out = F.relu(out)
        out = self.conv_1x1(out)
        out = self.dropout(out)
        scale = torch.sigmoid(self._conv_sq(tag)(out))
        out = out * scale
        return x + out


class DualDilatedResidualLayer(nn.Module):
    """
    Dual dilated residual layer
    """

    def __init__(
        self,
        dilation1,
        dilation2,
        in_channels,
        out_channels,
        dropout_rate,
        causal,
        kernel_size=3,
    ):
        """
        Parameters
        ----------
        dilation1, dilation2 : int
            dilation of one of the blocks
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        dropout_rate : float
            dropout rate
        causal : bool
            if `True`, causal convolutions are used
        kernel_size : int, default 3
            kernel size
        """

        super(DualDilatedResidualLayer, self).__init__()
        self.causal = causal
        self.padding1 = dilation1 * (kernel_size - 1) // 2
        self.padding2 = dilation2 * (kernel_size - 1) // 2
        if self.causal:
            self.padding1 *= 2
            self.padding2 *= 2
        self.conv_dilated_1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=self.padding1,
            dilation=dilation1,
        )
        self.conv_dilated_2 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=self.padding2,
            dilation=dilation2,
        )
        self.conv_fusion = nn.Conv1d(2 * out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, direction=None):
        """Forward pass."""
        if direction is not None and not self.causal:
            raise ValueError("Cannot set direction in a non-causal layer!")
        elif direction is None and self.causal:
            direction = "forward"
        if direction not in ["backward", "forward", None]:
            raise ValueError(
                f"Unrecognized direction: {direction}, please choose from"
                f'"backward", "forward" and None'
            )
        out_1 = self.conv_dilated_1(x)
        out_2 = self.conv_dilated_2(x)
        if direction == "forward":
            out_1 = out_1[:, :, : -self.padding1]
            out_2 = out_2[:, :, : -self.padding2]
        elif direction == "backward":
            out_1 = out_1[:, :, self.padding1 :]
            out_2 = out_2[:, :, self.padding2 :]

        out = self.conv_fusion(
            torch.cat(
                [
                    out_1,
                    out_2,
                ],
                1,
            )
        )
        out = F.relu(out)
        out = self.dropout(out)
        out = out + x
        return out


class MultiDilatedTCN(nn.Module):
    """
    Multiple prediction generation stages in parallel
    """

    def __init__(
        self,
        num_layers,
        num_f_maps,
        dims,
        direction,
        block_size=5,
        kernel_size=3,
        rare_dilations=False,
    ):
        super(MultiDilatedTCN, self).__init__()
        self.PGs = nn.ModuleList(
            [
                DilatedTCN(
                    num_layers,
                    num_f_maps,
                    dim,
                    direction,
                    block_size,
                    kernel_size,
                    rare_dilations,
                )
                for dim in dims
            ]
        )
        self.conv_out = nn.Conv1d(num_f_maps * len(dims), num_f_maps, 1)

    def forward(self, x, tag=None):
        """Forward pass."""
        out = []
        for arr, PG in zip(x, self.PGs):
            out.append(PG(arr))
        out = torch.cat(out, 1)
        out = self.conv_out(out)
        return out


class DilatedTCN(nn.Module):
    """
    Prediction generation stage
    """

    def __init__(
        self,
        num_layers,
        num_f_maps,
        dim,
        direction,
        num_bp=None,
        block_size=5,
        kernel_size=3,
        rare_dilations=False,
        attention="none",
        multihead=False,
    ):
        """
        Parameters
        ----------
        num_layers : int
            number of layers
        num_f_maps : int
            number of feature maps
        dim : int
            number of features in input
        direction : [None, 'forward', 'backward']
            the direction of convolutions; if None, regular convolutions are used
        block_size : int, default 0
            if not 0, skip connections are added to the prediction generation stage with this interval
        kernel_size : int, default 3
            kernel size
        rare_dilations : bool, default False
            if `False`, dilation increases every layer, otherwise every second layer
        """

        super().__init__()
        self.num_layers = num_layers
        self.block_size = block_size
        self.direction = direction
        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)
        pars = {
            "in_channels": num_f_maps,
            "out_channels": num_f_maps,
            "dropout_rate": 0.5,
            "causal": (direction is not None),
            "kernel_size": kernel_size,
        }
        module = DualDilatedResidualLayer
        if not rare_dilations:
            self.layers = nn.ModuleList(
                [
                    module(dilation1=2 ** (num_layers - 1 - i), dilation2=2**i, **pars)
                    for i in range(num_layers)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    module(
                        dilation1=2 ** (num_layers // 2 - 1 - i // 2),
                        dilation2=2 ** (i // 2),
                        **pars,
                    )
                    for i in range(num_layers)
                ]
            )
        self.attention_layers = nn.ModuleList([])
        self.attention = attention
        if self.attention in ["basic"]:
            self.attention_layers += nn.ModuleList(
                [
                    nn.Conv1d(num_f_maps, num_f_maps, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(num_f_maps, num_f_maps, 3, padding=1),
                    nn.Sigmoid(),
                ]
            )
        elif isinstance(self.attention, str) and self.attention != "none":
            self.attention_layers = AttLayer(
                num_f_maps,
                num_f_maps,
                num_f_maps,
                4,
                4,
                2,
                64,
                att_type=self.attention,
                stage="encoder",
            )
        self.multihead = multihead

    def forward(self, x, tag=None):
        """Forward pass."""
        x = self.conv_1x1_in(x)
        f = copy.copy(x)
        for i, layer in enumerate(self.layers):
            f = layer(f, self.direction)
            if self.block_size != 0 and (i + 1) % self.block_size == 0:
                f = f + x
                x = copy.copy(f)
        if isinstance(self.attention, str) and self.attention != "none":
            x = copy.copy(f)
            if self.attention != "basic":
                f = self.attention_layers(x)
            elif not self.multihead:
                for layer in self.attention_layers:
                    x = layer(x)
                f = f * x
            else:
                outputs = []
                for layers in self.attention_layers:
                    y = copy.copy(x)
                    for layer in layers:
                        y = layer(y)
                    outputs.append(copy.copy(f * y))
                outputs = torch.cat(outputs, dim=1)
                f = self.conv_att(self.dropout(outputs))
        return f


class DilatedTCNB(nn.Module):
    """
    Bidirectional prediction generation stage
    """

    def __init__(
        self,
        num_layers,
        num_f_maps,
        dim,
        block_size=5,
        kernel_size=3,
        rare_dilations=False,
    ):
        """
        Parameters
        ----------
        num_layers : int
            number of layers
        num_f_maps : int
            number of feature maps
        dim : int
            number of features in input
        block_size : int, default 0
            if not 0, skip connections are added to the prediction generation stage with this interval
        kernel_size : int, default 3
            kernel size
        rare_dilations : bool, default False
            if `False`, dilation increases every layer, otherwise every second layer
        """

        super().__init__()
        self.num_layers = num_layers
        self.block_size = block_size
        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)
        self.conv_1x1_out = nn.Conv1d(num_f_maps * 2, num_f_maps, 1)
        if not rare_dilations:
            self.layers = nn.ModuleList(
                [
                    DualDilatedResidualLayer(
                        dilation1=2 ** (num_layers - 1 - i),
                        dilation2=2**i,
                        in_channels=num_f_maps,
                        out_channels=num_f_maps,
                        dropout_rate=0.5,
                        causal=True,
                        kernel_size=kernel_size,
                    )
                    for i in range(num_layers)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    DualDilatedResidualLayer(
                        dilation1=2 ** (num_layers // 2 - 1 - i // 2),
                        dilation2=2 ** (i // 2),
                        in_channels=num_f_maps,
                        out_channels=num_f_maps,
                        dropout_rate=0.5,
                        causal=True,
                        kernel_size=kernel_size,
                    )
                    for i in range(num_layers)
                ]
            )

    def forward(self, x, tag=None):
        """Forward pass."""
        x_f = self.conv_1x1_in(x)
        x_b = copy.copy(x_f)
        forward = copy.copy(x_f)
        backward = copy.copy(x_f)
        for i, layer_f in enumerate(self.layers):
            forward = layer_f(forward, "forward")
            backward = layer_f(backward, "backward")
            if self.block_size != 0 and (i + 1) % self.block_size == 0:
                forward = forward + x_f
                backward = backward + x_b
                x_f = copy.copy(forward)
                x_b = copy.copy(backward)
        out = torch.cat([forward, backward], 1)
        out = self.conv_1x1_out(out)
        return out


class SpatialFeatures(nn.Module):
    """
    Spatial features extraction stage
    """

    def __init__(
        self,
        num_layers,
        num_f_maps,
        dim,
        block_size=5,
        graph_edges=None,
        num_nodes=None,
        denom: int = 8,
    ):
        """
        Parameters
        ----------
        num_layers : int
            number of layers
        num_f_maps : int
            number of feature maps
        dim : int
            number of features in input
        block_size : int, default 5
            if not 0, skip connections are added to the prediction generation stage with this interval
        """

        super().__init__()
        self.num_nodes = num_nodes
        if graph_edges is None:
            module = SimpleResidualLayer
            self.graph = False
            pars = {"num_f_maps": num_f_maps, "dropout_rate": 0.5}
            self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)
        else:
            raise NotImplementedError("Graph not implemented")
        
        self.num_layers = num_layers
        self.block_size = block_size
        self.layers = nn.ModuleList([module(**pars) for _ in range(num_layers)])

    def forward(self, x):
        """Forward pass."""
        if self.graph:
            B, _, L = x.shape
            x = x.transpose(-1, -2)
            x = x.reshape((-1, x.shape[-1]))
            x = x.reshape((x.shape[0], self.num_nodes, -1))
            x = x.transpose(-1, -2)
        x = self.conv_1x1_in(x)
        f = copy.copy(x)
        for i, layer in enumerate(self.layers):
            f = layer(f)
            if self.block_size != 0 and (i + 1) % self.block_size == 0:
                f = f + x
                x = copy.copy(f)
        if self.graph:
            f = f.reshape((B, L, -1))
            f = f.transpose(-1, -2)
            f = self.conv_1x1_out(f)
        return f


class MSRefinement(nn.Module):
    """
    Refinement stage
    """

    def __init__(
        self,
        num_layers_R,
        num_R,
        num_f_maps_input,
        num_f_maps,
        num_classes,
        dropout_rate,
        exclusive,
        skip_connections,
        direction,
        block_size=0,
        num_heads=1,
        attention="none",
    ):
        """
        Parameters
        ----------
        num_layers_R : int
            number of layers in refinement modules
        num_R : int
            number of refinement modules
        num_f_maps : int
            number of feature maps
        num_classes : int
            number of target classes
        dropout_rate : float
            dropout rate
        exclusive : bool
            set `False` for multi-label classification
        skip_connections : bool
            if `True`, skip connections are added
        direction : [None, 'bidirectional', 'forward', 'backward']
            the direction of convolutions; if None, regular convolutions are used
        block_size : int, default 0
            if not 0, skip connections are added to the prediction generation stage with this interval
        num_heads : int, default 1
            number of parallel refinement stages
        """

        super().__init__()
        self.skip_connections = skip_connections
        self.num_heads = num_heads
        if exclusive:
            self.nl = lambda x: F.softmax(x, dim=1)
        else:
            self.nl = lambda x: torch.sigmoid(x)
        if direction == "bidirectional":
            refinement_module = RefinementB
        else:
            refinement_module = Refinement
        self.Rs = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        refinement_module(
                            num_layers=num_layers_R,
                            num_f_maps=num_f_maps,
                            num_f_maps_input=num_f_maps_input,
                            dim=num_classes,
                            num_classes=num_classes,
                            dropout_rate=dropout_rate,
                            direction=direction,
                            skip_connections=skip_connections,
                            block_size=block_size,
                            attention=attention,
                        )
                        for s in range(num_R)
                    ]
                )
                for _ in range(self.num_heads)
            ]
        )
        self.conv_out = nn.ModuleList(
            [nn.Conv1d(num_f_maps_input, num_classes, 1) for _ in range(self.num_heads)]
        )
        if self.num_heads == 1:
            self.Rs = self.Rs[0]
            self.conv_out = self.conv_out[0]

    def _Rs(self, tag):
        if self.num_heads == 1:
            return self.Rs
        if tag is None:
            tag = 0
            for i in range(1, self.num_heads):
                self.Rs[i].load_state_dict(self.Rs[0].state_dict())
        return self.Rs[tag]

    def _conv_out(self, tag):
        if self.num_heads == 1:
            return self.conv_out
        if tag is None:
            tag = 0
            for i in range(1, self.num_heads):
                self.conv_out[i].load_state_dict(self.conv_out[0].state_dict())
        return self.conv_out[tag]

    def forward(self, x, tag=None):
        """Forward pass."""
        if tag is not None:
            tag = tag[0]
        out = self._conv_out(tag)(x)
        outputs = out.unsqueeze(0)
        for R in self._Rs(tag):
            if self.skip_connections:
                out = R(torch.cat([self.nl(out), x], axis=1))
                # out = R(torch.cat([out, x], axis=1))
            else:
                out = R(self.nl(out))
                # out = R(out)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs


class MSRefinementShared(nn.Module):
    """
    Refinement stage with shared weights across modules
    """

    def __init__(
        self,
        num_layers_R,
        num_R,
        num_f_maps_input,
        num_f_maps,
        num_classes,
        dropout_rate,
        exclusive,
        skip_connections,
        direction,
        block_size=0,
        num_heads=1,
        attention="none",
    ):
        """
        Parameters
        ----------
        num_layers_R : int
            number of layers in refinement modules
        num_R : int
            number of refinement modules
        num_f_maps : int
            number of feature maps
        num_classes : int
            number of target classes
        dropout_rate : float
            dropout rate
        exclusive : bool
            set `False` for multi-label classification
        skip_connections : bool
            if `True`, skip connections are added
        direction : [None, 'bidirectional', 'forward', 'backward']
            the direction of convolutions; if None, regular convolutions are used
        block_size : int, default 0
            if not 0, skip connections are added to the prediction generation stage with this interval
        num_heads : int, default 1
            number of parallel refinement stages
        """

        super().__init__()
        if exclusive:
            self.nl = lambda x: F.softmax(x, dim=1)
        else:
            self.nl = lambda x: torch.sigmoid(x)
        if direction == "bidirectional":
            refinement_module = RefinementB
        else:
            refinement_module = Refinement
        self.num_heads = num_heads
        self.R = nn.ModuleList(
            [
                refinement_module(
                    num_layers=num_layers_R,
                    num_f_maps_input=num_f_maps_input,
                    num_f_maps=num_f_maps,
                    dim=num_classes,
                    num_classes=num_classes,
                    dropout_rate=dropout_rate,
                    direction=direction,
                    skip_connections=skip_connections,
                    block_size=block_size,
                    attention=attention,
                )
                for _ in range(self.num_heads)
            ]
        )
        self.num_R = num_R
        self.conv_out = nn.ModuleList(
            [nn.Conv1d(num_f_maps_input, num_classes, 1) for _ in range(self.num_heads)]
        )
        self.skip_connections = skip_connections
        if self.num_heads == 1:
            self.R = self.R[0]
            self.conv_out = self.conv_out[0]

    def _R(self, tag):
        if self.num_heads == 1:
            return self.R
        if tag is None:
            tag = 0
            for i in range(1, self.num_heads):
                self.R[i].load_state_dict(self.R[0].state_dict())
        return self.R[tag]

    def _conv_out(self, tag):
        if self.num_heads == 1:
            return self.conv_out
        if tag is None:
            tag = 0
            for i in range(1, self.num_heads):
                self.conv_out[i].load_state_dict(self.conv_out[0].state_dict())
        return self.conv_out[tag]

    def forward(self, x, tag=None):
        """Forward pass."""
        if tag is not None:
            tag = tag[0]
        out = self._conv_out(tag)(x)
        outputs = out.unsqueeze(0)
        for _ in range(self.num_R):
            if self.skip_connections:
                # out = self._R(tag)(torch.cat([self.nl(out), x], axis=1))
                out = self._R(tag)(torch.cat([out, x], axis=1))
            else:
                # out = self._R(tag)(self.nl(out))
                out = self._R(tag)(out)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs


class MSRefinementAttention(nn.Module):
    """
    Refinement stage
    """

    def __init__(
        self,
        num_layers_R,
        num_R,
        num_f_maps_input,
        num_f_maps,
        num_classes,
        dropout_rate,
        exclusive,
        skip_connections,
        len_segment,
        block_size=0,
    ):
        """
        Parameters
        ----------
        num_layers_R : int
            number of layers in refinement modules
        num_R : int
            number of refinement modules
        num_f_maps : int
            number of feature maps
        num_classes : int
            number of target classes
        dropout_rate : float
            dropout rate
        exclusive : bool
            set `False` for multi-label classification
        skip_connections : bool
            if `True`, skip connections are added
        direction : [None, 'bidirectional', 'forward', 'backward']
            the direction of convolutions; if None, regular convolutions are used
        block_size : int, default 0
            if not 0, skip connections are added to the prediction generation stage with this interval
        num_heads : int, default 1
            number of parallel refinement stages
        """

        super().__init__()
        self.skip_connections = skip_connections
        if exclusive:
            self.nl = lambda x: F.softmax(x, dim=1)
        else:
            self.nl = lambda x: torch.sigmoid(x)
        refinement_module = Refinement_SE
        self.Rs = nn.ModuleList(
            [
                refinement_module(
                    num_layers=num_layers_R,
                    num_f_maps=num_f_maps,
                    num_f_maps_input=num_f_maps_input,
                    dim=num_classes,
                    num_classes=num_classes,
                    dropout_rate=dropout_rate,
                    direction=None,
                    skip_connections=skip_connections,
                    block_size=block_size,
                    len_segment=len_segment,
                )
                for s in range(num_R)
            ]
        )
        self.conv_out = nn.Conv1d(num_f_maps_input, num_classes, 1)

    def forward(self, x, tag=None):
        """Forward pass."""
        if tag is not None:
            tag = tag[0]
        out = self.conv_out(x)
        outputs = out.unsqueeze(0)
        for R in self.Rs:
            if self.skip_connections:
                out = R(torch.cat([self.nl(out), x], axis=1), tag)
                # out = R(torch.cat([out, x], axis=1), tag)
            else:
                out = R(self.nl(out), tag)
                # out = R(out, tag)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs


class DilatedTCNC(nn.Module):
    def __init__(
        self,
        num_f_maps,
        num_layers_PG,
        len_segment,
        block_size_prediction=5,
        kernel_size_prediction=3,
        direction_PG=None,
    ):
        super(DilatedTCNC, self).__init__()
        if direction_PG == "bidirectional":
            self.PG_S = DilatedTCNB(
                num_layers=num_layers_PG,
                num_f_maps=num_f_maps,
                dim=num_f_maps,
                block_size=block_size_prediction,
            )
            self.PG_T = DilatedTCNB(
                num_layers=num_layers_PG,
                num_f_maps=len_segment,
                dim=len_segment,
                block_size=block_size_prediction,
            )
        else:
            self.PG_S = DilatedTCN(
                num_layers=num_layers_PG,
                num_f_maps=num_f_maps,
                dim=num_f_maps,
                direction=direction_PG,
                block_size=block_size_prediction,
                kernel_size=kernel_size_prediction,
            )
            self.PG_T = DilatedTCN(
                num_layers=num_layers_PG,
                num_f_maps=len_segment,
                dim=len_segment,
                direction=direction_PG,
                block_size=block_size_prediction,
                kernel_size=kernel_size_prediction,
            )

    def forward(self, x):
        """Forward pass."""
        x = self.PG_S(x)
        x = torch.transpose(x, 1, 2)
        x = self.PG_T(x)
        x = torch.transpose(x, 1, 2)
        return x
