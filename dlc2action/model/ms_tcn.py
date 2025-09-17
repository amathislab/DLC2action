#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
# Incorporates code adapted from MS-TCN++ by yabufarha
# Original work Copyright (c) 2019 June01
# Source: https://github.com/sj-li/MS-TCN2
# Originally licensed under MIT License
# Combined work licensed under GNU AGPLv3
#
"""
MS-TCN++ (multi-stage temporal convolutional network) variations
"""

from dlc2action.model.base_model import Model
from dlc2action.model.ms_tcn_modules import *


class Compiled(nn.Module):
    def __init__(self, modules):
        super(Compiled, self).__init__()
        self.module_list = nn.ModuleList(modules)

    def forward(self, x, tag=None):
        """Forward pass."""
        for m in self.module_list:
            x = m(x, tag)
        return x


class MS_TCN3(Model):
    """
    A modification of MS-TCN++ model with additional options
    """

    def __init__(
        self,
        num_f_maps,
        num_classes,
        exclusive,
        dims,
        num_layers_R,
        num_R,
        num_layers_PG,
        num_f_maps_R=None,
        num_layers_S=0,
        dropout_rate=0.5,
        shared_weights=False,
        skip_connections_refinement=True,
        block_size_prediction=0,
        block_size_refinement=0,
        kernel_size_prediction=3,
        direction_PG=None,
        direction_R=None,
        PG_in_FE=False,
        rare_dilations=False,
        num_heads=1,
        R_attention="none",
        PG_attention="none",
        state_dict_path=None,
        ssl_constructors=None,
        ssl_types=None,
        ssl_modules=None,
        multihead=False,
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------
        num_f_maps : int
            number of feature maps
        num_classes : int
            number of classes to predict
        exclusive : bool
            if `True`, single-label predictions are made; otherwise multi-label
        dims : torch.Size
            shape of features in the input data
        num_layers_R : int
            number of layers in the refinement stages
        num_R : int
            number of refinement stages
        num_layers_PG : int
            number of layers in the prediction generation stage
        num_layers_S : int, default 0
            number of layers in the spatial feature extraction stage
        dropout_rate : float, default 0.5
            dropout rate
        shared_weights : bool, default False
            if `True`, weights are shared across refinement stages
        skip_connections_refinement : bool, default False
            if `True`, skip connections are added to the refinement stages
        block_size_prediction : int, default 0
            if not 0, skip connections are added to the prediction generation stage with this interval
        block_size_refinement : int, default 0
            if not 0, skip connections are added to the refinement stage with this interval
        direction_PG : [None, 'bidirectional', 'forward', 'backward']
            if not `None`, a combination of causal and anticausal convolutions are used in the
            prediction generation stage
        direction_R : [None, 'bidirectional', 'forward', 'backward']
            if not `None`, a combination of causal and anticausal convolutions are used in the refinement stages
        PG_in_FE : bool, default True
            if `True`, the prediction generation stage is included in the feature extractor and otherwise in the
            predictor (the output of the feature extractor is used in SSL tasks)
        rare_dilations : bool, default False
            if `False`, dilation increases every layer, otherwise every second layer in
            the prediction generation stage
        num_heads : int, default 1
            the number of parallel refinement stages
        PG_attention : bool, default False
            if `True`, an attention layer is added to the prediction generation stage
        R_attention : bool, default False
            if `True`, an attention layer is added to the refinement stages
        state_dict_path : str, optional
            if not `None`, the model state dictionary will be loaded from this path
        ssl_constructors : list, optional
            a list of `dlc2action.ssl.base_ssl.SSLConstructor` instances to integrate
        ssl_types : list, optional
            a list of types of the SSL modules to integrate (used alternatively to `ssl_constructors`)
        ssl_modules : list, optional
            a list of SSL modules to integrate (used alternatively to `ssl_constructors`)
        """

        self.num_layers_R = int(float(num_layers_R))
        self.num_R = int(float(num_R))
        self.num_f_maps = int(float(num_f_maps))
        self.num_classes = int(float(num_classes))
        self.dropout_rate = float(dropout_rate)
        self.exclusive = bool(exclusive)
        self.num_layers_PG = int(float(num_layers_PG))
        self.num_layers_S = int(float(num_layers_S))
        self.dim = self._get_dims(dims)
        self.shared_weights = bool(shared_weights)
        self.skip_connections_ref = bool(skip_connections_refinement)
        self.block_size_prediction = int(float(block_size_prediction))
        self.block_size_refinement = int(float(block_size_refinement))
        self.direction_R = direction_R
        self.direction_PG = direction_PG
        self.kernel_size_prediction = int(float(kernel_size_prediction))
        self.PG_in_FE = PG_in_FE
        self.rare_dilations = rare_dilations
        self.num_heads = int(float(num_heads))
        self.PG_attention = PG_attention
        self.R_attention = R_attention
        self.multihead = multihead
        if num_f_maps_R is None:
            num_f_maps_R = self.num_f_maps
        self.num_f_maps_R = num_f_maps_R
        super().__init__(ssl_constructors, ssl_modules, ssl_types, state_dict_path)

    def _get_dims(self, dims):
        return int(sum([s[0] for s in dims.values()]))

    def _PG(self):
        if self.num_layers_S == 0:
            dim = self.dim
        else:
            dim = self.num_f_maps
        if self.direction_PG == "bidirectional":
            PG = DilatedTCNB(
                num_layers=self.num_layers_PG,
                num_f_maps=self.num_f_maps,
                dim=dim,
                block_size=self.block_size_prediction,
                kernel_size=self.kernel_size_prediction,
                rare_dilations=self.rare_dilations,
            )
        else:
            PG = DilatedTCN(
                num_layers=self.num_layers_PG,
                num_f_maps=self.num_f_maps,
                dim=dim,
                direction=self.direction_PG,
                block_size=self.block_size_prediction,
                kernel_size=self.kernel_size_prediction,
                rare_dilations=self.rare_dilations,
                attention=self.PG_attention,
                multihead=self.multihead,
            )
        return PG

    def _feature_extractor(self):
        if self.num_layers_S == 0:
            if self.PG_in_FE:
                print("MS-TCN using the prediction generator as a feature extractor")
                return self._PG()
            else:
                print("MS-TCN without a feature extractor -> no SSL possible!")
                return nn.Identity()

        print("MS-TCN using a spatial feature extractor")
        feature_extractor = SpatialFeatures(
            self.num_layers_S,
            self.num_f_maps,
            self.dim,
            self.block_size_prediction,
        )
        if self.PG_in_FE:
            print("  -> also has the prediction generator as a feature extractor")
            PG = self._PG()
            feature_extractor = [feature_extractor, PG]
        return feature_extractor

    def _predictor(self):
        if self.shared_weights:
            prediction_module = MSRefinementShared
        else:
            prediction_module = MSRefinement
        predictor = prediction_module(
            num_layers_R=int(self.num_layers_R),
            num_R=int(self.num_R),
            num_f_maps_input=int(self.num_f_maps),
            num_f_maps=int(self.num_f_maps_R),
            num_classes=int(self.num_classes),
            dropout_rate=self.dropout_rate,
            exclusive=self.exclusive,
            skip_connections=self.skip_connections_ref,
            direction=self.direction_R,
            block_size=self.block_size_refinement,
            num_heads=self.num_heads,
            attention=self.R_attention,
        )
        if not self.PG_in_FE:
            PG = self._PG()
            predictor = Compiled([PG, predictor])
        return predictor

    def features_shape(self) -> torch.Size:
        """
        Get the shape of feature extractor output

        Returns
        -------
        feature_shape : torch.Size
            shape of feature extractor output
        """

        return torch.Size([self.num_f_maps])


class MS_TCN_P(MS_TCN3):
    def _get_dims(self, dims):
        keys = list(dims.keys())
        values = list(dims.values())
        groups = [key.split("---")[-1] for key in keys]
        unique_groups = sorted(set(groups))
        res = []
        for group in unique_groups:
            res.append(int(sum([x[0] for x, g in zip(values, groups) if g == group])))
        if "loaded" in dims:
            res.append(int(dims["loaded"][0]))
        return res

    def _PG(self):
        PG = MultiDilatedTCN(
            self.num_layers_PG,
            self.num_f_maps,
            self.dim,
            self.direction_PG,
            self.block_size_prediction,
            self.kernel_size_prediction,
            self.rare_dilations,
        )
        return PG


# class MS_TCNC(Model):
#     """
#     Basic MS-TCN++ model with options for shared weights and added skip connections
#     """
#
#     def __init__(
#         self,
#         num_layers_R,
#         num_R,
#         num_f_maps,
#         num_classes,
#         exclusive,
#         num_layers_PG,
#         num_layers_S,
#         dims,
#         len_segment,
#         dropout_rate=0.5,
#         shared_weights=False,
#         skip_connections_refinement=True,
#         block_size_prediction=5,
#         block_size_refinement=0,
#         kernel_size_prediction=3,
#         direction_PG=None,
#         direction_R=None,
#         PG_in_FE=False,
#         state_dict_path=None,
#         ssl_constructors=None,
#         ssl_types=None,
#         ssl_modules=None,
#     ):
#         """
#         Parameters
#         ----------
#         num_layers_R : int
#             number of layers in the refinement stages
#         num_R : int
#             number of refinement stages
#         num_f_maps : int
#             number of feature maps
#         num_classes : int
#             number of classes to predict
#         exclusive : bool
#             if `True`, single-label predictions are made; otherwise multi-label
#         num_layers_PG : int
#             number of layers in the prediction generation stage
#         dims : torch.Size
#             shape of features in the input data
#         dropout_rate : float, default 0.5
#             dropout rate
#         shared_weights : bool, default False
#             if `True`, weights are shared across refinement stages
#         skip_connections_refinement : bool, default False
#             if `True`, skip connections are added to the refinement stages
#         block_size_prediction : int, optional
#             if not 'None', skip connections are added to the prediction generation stage with this interval
#         direction_PG : bool, default True
#             if True, causal convolutions are used in the prediction generation stage
#         direction_R : bool, default False
#             if True, causal convolutions are used in the refinement stages
#         state_dict_path : str, optional
#             if not `None`, the model state dictionary will be loaded from this path
#         ssl_constructors : list, optional
#             a list of `dlc2action.ssl.base_ssl.SSLConstructor` instances to integrate
#         ssl_types : list, optional
#             a list of types of the SSL modules to integrate (used alternatively to `ssl_constructors`)
#         ssl_modules : list, optional
#             a list of SSL modules to integrate (used alternatively to `ssl_constructors`)
#         """
#
#         if len(dims) > 1:
#             raise RuntimeError(
#                 "The MS-TCN++ model expects the input data to be 2-dimensional; "
#                 f"got {len(dims) + 1} dimensions"
#             )
#         self.num_layers_R = int(num_layers_R)
#         self.num_R = int(num_R)
#         self.num_f_maps = int(num_f_maps)
#         self.num_classes = int(num_classes)
#         self.dropout_rate = dropout_rate
#         self.exclusive = exclusive
#         self.num_layers_PG = int(num_layers_PG)
#         self.num_layers_S = int(num_layers_S)
#         self.dim = int(dims[0])
#         self.shared_weights = shared_weights
#         self.skip_connections_ref = skip_connections_refinement
#         self.block_size_prediction = int(block_size_prediction)
#         self.block_size_refinement = int(block_size_refinement)
#         self.direction_R = direction_R
#         self.direction_PG = direction_PG
#         self.kernel_size_prediction = int(kernel_size_prediction)
#         self.PG_in_FE = PG_in_FE
#         self.len_segment = len_segment
#         super().__init__(ssl_constructors, ssl_modules, ssl_types, state_dict_path)
#
#     def _PG(self):
#         PG = DilatedTCNC(
#             num_f_maps=self.num_f_maps,
#             num_layers_PG=self.num_layers_PG,
#             len_segment=self.len_segment,
#             block_size_prediction=self.block_size_prediction,
#             kernel_size_prediction=self.kernel_size_prediction,
#             direction_PG=self.direction_PG,
#         )
#         return PG
#
#     def _feature_extractor(self):
#         feature_extractor = SpatialFeatures(
#             num_layers=self.num_layers_S,
#             num_f_maps=self.num_f_maps,
#             dim=self.dim,
#             block_size=self.block_size_prediction,
#         )
#         if self.PG_in_FE:
#             PG = self._PG()
#             feature_extractor = [feature_extractor, PG]
#         return feature_extractor
#
#     def _predictor(self):
#
#         if self.shared_weights:
#             prediction_module = MSRefinementShared
#         else:
#             prediction_module = MSRefinement
#         predictor = prediction_module(
#             num_layers_R=int(self.num_layers_R),
#             num_R=int(self.num_R),
#             num_f_maps=int(self.num_f_maps),
#             num_classes=int(self.num_classes),
#             dropout_rate=self.dropout_rate,
#             exclusive=self.exclusive,
#             skip_connections=self.skip_connections_ref,
#             direction=self.direction_R,
#             block_size=self.block_size_refinement,
#         )
#         if not self.PG_in_FE:
#             PG = self._PG()
#             predictor = Compiled([PG, predictor])
#         return predictor
#
#     def features_shape(self) -> torch.Size:
#         """
#         Get the shape of feature extractor output
#
#         Returns
#         -------
#         feature_shape : torch.Size
#             shape of feature extractor output
#         """
#
#         return torch.Size([self.num_f_maps])
#
# class MS_TCNA(Model):
#     """
#     Basic MS-TCN++ model with additional options
#     """
#
#     def __init__(
#         self,
#         num_f_maps,
#         num_classes,
#         exclusive,
#         dims,
#         num_layers_R,
#         num_R,
#         num_layers_PG,
#         len_segment,
#         num_f_maps_R=None,
#         num_layers_S=0,
#         dropout_rate=0.5,
#         skip_connections_refinement=True,
#         block_size_prediction=0,
#         block_size_refinement=0,
#         kernel_size_prediction=3,
#         direction_PG=None,
#         direction_R=None,
#         PG_in_FE=False,
#         rare_dilations=False,
#         state_dict_path=None,
#         ssl_constructors=None,
#         ssl_types=None,
#         ssl_modules=None,
#         *args, **kwargs
#     ):
#         """
#         Parameters
#         ----------
#         num_f_maps : int
#             number of feature maps
#         num_classes : int
#             number of classes to predict
#         exclusive : bool
#             if `True`, single-label predictions are made; otherwise multi-label
#         dims : torch.Size
#             shape of features in the input data
#         num_layers_R : int
#             number of layers in the refinement stages
#         num_R : int
#             number of refinement stages
#         num_layers_PG : int
#             number of layers in the prediction generation stage
#         num_layers_S : int, default 0
#             number of layers in the spatial feature extraction stage
#         dropout_rate : float, default 0.5
#             dropout rate
#         shared_weights : bool, default False
#             if `True`, weights are shared across refinement stages
#         skip_connections_refinement : bool, default False
#             if `True`, skip connections are added to the refinement stages
#         block_size_prediction : int, default 0
#             if not 0, skip connections are added to the prediction generation stage with this interval
#         block_size_refinement : int, default 0
#             if not 0, skip connections are added to the refinement stage with this interval
#         direction_PG : [None, 'bidirectional', 'forward', 'backward']
#             if not `None`, a combination of causal and anticausal convolutions are used in the
#             prediction generation stage
#         direction_R : [None, 'bidirectional', 'forward', 'backward']
#             if not `None`, a combination of causal and anticausal convolutions are used in the refinement stages
#         PG_in_FE : bool, default True
#             if `True`, the prediction generation stage is included in the feature extractor and otherwise in the
#             predictor (the output of the feature extractor is used in SSL tasks)
#         rare_dilations : bool, default False
#             if `False`, dilation increases every layer, otherwise every second layer in
#             the prediction generation stage
#         num_heads : int, default 1
#             the number of parallel refinement stages
#         state_dict_path : str, optional
#             if not `None`, the model state dictionary will be loaded from this path
#         ssl_constructors : list, optional
#             a list of `dlc2action.ssl.base_ssl.SSLConstructor` instances to integrate
#         ssl_types : list, optional
#             a list of types of the SSL modules to integrate (used alternatively to `ssl_constructors`)
#         ssl_modules : list, optional
#             a list of SSL modules to integrate (used alternatively to `ssl_constructors`)
#         """
#
#         if len(dims) > 1:
#             raise RuntimeError(
#                 "The MS-TCN++ model expects the input data to be 2-dimensional; "
#                 f"got {len(dims) + 1} dimensions"
#             )
#         self.num_layers_R = int(num_layers_R)
#         self.num_R = int(num_R)
#         self.num_f_maps = int(num_f_maps)
#         self.num_classes = int(num_classes)
#         self.dropout_rate = dropout_rate
#         self.exclusive = exclusive
#         self.num_layers_PG = int(num_layers_PG)
#         self.num_layers_S = int(num_layers_S)
#         self.dim = int(dims[0])
#         self.skip_connections_ref = skip_connections_refinement
#         self.block_size_prediction = int(block_size_prediction)
#         self.block_size_refinement = int(block_size_refinement)
#         self.direction_R = direction_R
#         self.direction_PG = direction_PG
#         self.kernel_size_prediction = int(kernel_size_prediction)
#         self.PG_in_FE = PG_in_FE
#         self.rare_dilations = rare_dilations
#         self.len_segment = len_segment
#         if num_f_maps_R is None:
#             num_f_maps_R = num_f_maps
#         self.num_f_maps_R = num_f_maps_R
#         super().__init__(ssl_constructors, ssl_modules, ssl_types, state_dict_path)
#
#     def _PG(self):
#         if self.num_layers_S == 0:
#             dim = self.dim
#         else:
#             dim = self.num_f_maps
#         if self.direction_PG == "bidirectional":
#             PG = DilatedTCNB(
#                 num_layers=self.num_layers_PG,
#                 num_f_maps=self.num_f_maps,
#                 dim=dim,
#                 block_size=self.block_size_prediction,
#                 kernel_size=self.kernel_size_prediction,
#                 rare_dilations=self.rare_dilations,
#             )
#         else:
#             PG = DilatedTCN(
#                 num_layers=self.num_layers_PG,
#                 num_f_maps=self.num_f_maps,
#                 dim=dim,
#                 direction=self.direction_PG,
#                 block_size=self.block_size_prediction,
#                 kernel_size=self.kernel_size_prediction,
#                 rare_dilations=self.rare_dilations,
#             )
#         return PG
#
#     def _feature_extractor(self):
#         if self.num_layers_S == 0:
#             if self.PG_in_FE:
#                 return self._PG()
#             else:
#                 return nn.Identity()
#         feature_extractor = SpatialFeatures(
#             self.num_layers_S,
#             self.num_f_maps,
#             self.dim,
#             self.block_size_prediction,
#         )
#         if self.PG_in_FE:
#             PG = self._PG()
#             feature_extractor = [feature_extractor, PG]
#         return feature_extractor
#
#     def _predictor(self):
#         predictor = MSRefinementAttention(
#             num_layers_R=int(self.num_layers_R),
#             num_R=int(self.num_R),
#             num_f_maps_input=int(self.num_f_maps),
#             num_f_maps=int(self.num_f_maps_R),
#             num_classes=int(self.num_classes),
#             dropout_rate=self.dropout_rate,
#             exclusive=self.exclusive,
#             skip_connections=self.skip_connections_ref,
#             block_size=self.block_size_refinement,
#             len_segment=self.len_segment,
#         )
#         if not self.PG_in_FE:
#             PG = self._PG()
#             predictor = Compiled([PG, predictor])
#         return predictor
#
#     def features_shape(self) -> torch.Size:
#         """
#         Get the shape of feature extractor output
#
#         Returns
#         -------
#         feature_shape : torch.Size
#             shape of feature extractor output
#         """
#
#         return torch.Size([self.num_f_maps])
