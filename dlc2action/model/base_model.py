#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
"""
Abstract parent class for models used in `dlc2action.task.universal_task.Task`
"""

from typing import Dict, Union, List, Callable, Tuple
from torch import nn
from abc import ABC, abstractmethod
import torch
import warnings
from collections.abc import Iterable
import copy

available_ssl_types = [
    "ssl_input",
    "ssl_target",
    "contrastive",
    "none",
    "contrastive_2layers",
]


class Model(nn.Module, ABC):
    """
    Base class for all models

    Manages interaction of base model and SSL modules + ensures consistent input and output format
    """

    process_labels = False

    def __init__(
        self,
        ssl_constructors: List = None,
        ssl_modules: List = None,
        ssl_types: List = None,
        state_dict_path: str = None,
        strict: bool = False,
        prompt_function: Callable = None,
    ) -> None:
        """
        Parameters
        ----------
        ssl_constructors : list, optional
            a list of SSL constructors that build the necessary SSL modules
        ssl_modules : list, optional
            a list of torch.nn.Module instances that will serve as SSL modules
        ssl_types : list, optional
            a list of string SSL types
        state_dict_path : str, optional
            path to the model state dictionary to load
        strict : bool, default False
            when True, the state dictionary will only be loaded if the current and the loaded architecture are the same;
            otherwise missing or extra keys, as well as shaoe inconsistencies, are ignored
        """

        super(Model, self).__init__()
        feature_extractors = self._feature_extractor()
        if not isinstance(feature_extractors, list):
            feature_extractors = [feature_extractors]
        self.feature_extractor = feature_extractors[0]
        self.feature_extractors = nn.ModuleList(feature_extractors[1:])
        self.predictor = self._predictor()
        self.set_ssl(ssl_constructors, ssl_types, ssl_modules)
        self.ssl_active = True
        self.main_task_active = True
        self.prompt_function = prompt_function
        self.class_tensors = None
        if state_dict_path is not None:
            self.load_state_dict(torch.load(state_dict_path), strict=strict)
        # self.feature_extractors = nn.ModuleList([nn.DataParallel(x) for x in self.feature_extractors])
        # self.predictor = nn.DataParallel(self.predictor)
        # if self.ssl != [None]:
        #     self.ssl = nn.ModuleList([nn.DataParallel(x) for x in self.ssl])

    # def to(self, device, *args, **kwargs):
    #     if self.class_tensors is not None:
    #         self.class_tensors = {
    #             k: v.to(device) for k, v in self.class_tensors.items()
    #         }
    #     return super().to(device, *args, **kwargs)

    def freeze_feature_extractor(self) -> None:
        """
        Freeze the parameters of the feature extraction module
        """

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze_feature_extractor(self) -> None:
        """
        Unfreeze the parameters of the feature extraction module
        """

        for param in self.feature_extractor.parameters():
            param.requires_grad = True

    def load_state_dict(self, state_dict: str, strict: bool = True) -> None:
        """
        Load a model state dictionary

        Parameters
        ----------
        state_dict : str
            the path to the saved state dictionary
        strict : bool, default True
            when True, the state dictionary will only be loaded if the current and the loaded architecture are the same;
            otherwise missing or extra keys, as well as shaoe inconsistencies, are ignored
        """

        try:
            super().load_state_dict(state_dict, strict)
        except RuntimeError as e:
            if strict:
                raise e
            else:
                warnings.warn(
                    "Some of the layer shapes do not match the loaded state dictionary, skipping those"
                )
                own_state = self.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    if isinstance(param, nn.Parameter):
                        # backwards compatibility for serialized parameters
                        param = param.data
                    try:
                        own_state[name].copy_(param)
                    except:
                        pass

    def ssl_off(self) -> None:
        """
        Turn SSL off (SSL output will not be computed by the forward function)
        """

        self.ssl_active = False

    def ssl_on(self) -> None:
        """
        Turn SSL on (SSL output will be computed by the forward function)
        """

        self.ssl_active = True

    def main_task_on(self) -> None:
        """
        Turn main task training on
        """

        self.main_task_active = True

    def main_task_off(self) -> None:
        """
        Turn main task training on
        """

        self.main_task_active = False

    def set_ssl(
        self,
        ssl_constructors: List = None,
        ssl_types: List = None,
        ssl_modules: List = None,
    ) -> None:
        """
        Set the SSL modules
        """

        if ssl_constructors is None and ssl_types is None:
            self.ssl_type = ["none"]
            self.ssl = [None]
        else:
            if ssl_constructors is not None:
                ssl_types = [
                    ssl_constructor.type for ssl_constructor in ssl_constructors
                ]
                ssl_modules = [
                    ssl_constructor.construct_module()
                    for ssl_constructor in ssl_constructors
                ]
            if not isinstance(ssl_types, Iterable):
                ssl_types = [ssl_types]
                ssl_modules = [ssl_modules]
            for t in ssl_types:
                if t not in available_ssl_types:
                    raise ValueError(
                        f"SSL type {t} is not implemented yet, please choose from {available_ssl_types}"
                    )
            self.ssl_type = ssl_types
            self.ssl = nn.ModuleList(ssl_modules)

    @abstractmethod
    def _feature_extractor(self) -> Union[torch.nn.Module, List]:
        """
        Construct the feature extractor module

        Returns
        -------
        feature_extractor : torch.nn.Module
            an instance of torch.nn.Module that has a forward method receiving input and
            returning features that can be passed to an SSL module and to a prediction module
        """

    @abstractmethod
    def _predictor(self) -> torch.nn.Module:
        """
        Construct the predictor module

        Returns
        -------
        predictor : torch.nn.Module
            an instance of torch.nn.Module that has a forward method receiving features
            exctracted by self.feature_extractor and returning a prediction
        """

    @abstractmethod
    def features_shape(self) -> torch.Size:
        """
        Get the shape of feature extractor output

        Returns
        -------
        feature_shape : torch.Size
            shape of feature extractor output
        """

    def extract_features(self, x, start=0):
        """
        Apply the feature extraction modules consecutively

        Parameters
        ----------
        x : torch.Tensor
            the input tensor
        start : int, default 0
            the index of the feature extraction module to start with

        Returns
        -------
        output : torch.Tensor
            the output tensor
        """

        if start == 0:
            x = self.feature_extractor(x)
        for extractor in self.feature_extractors[max(0, start - 1) :]:
            x = extractor(x)
        return x

    def _extract_features_first(self, x):
        return self.feature_extractor(x)

    def forward(
        self,
        x: torch.Tensor,
        ssl_xs: list,
        tag: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Generate a prediction for x

        Parameters
        ----------
        x : torch.Tensor
            the main input
        ssl_xs : list
            a list of SSL input tensors
        tag : any, optional
            a meta information tag

        Returns
        -------
        prediction : torch.Tensor
            prediction for the main input
        ssl_out : list
            a list of SSL prediction tensors
        """

        ssl_out = None
        features_0 = self._extract_features_first(x)
        if len(self.feature_extractors) > 1:
            features = copy.copy(features_0)
            features = self.extract_features(features, start=1)
        else:
            features = features_0
        if self.ssl_active:
            ssl_out = []
            for ssl, ssl_x, ssl_type in zip(self.ssl, ssl_xs, self.ssl_type):
                if ssl_type in ["contrastive", "ssl_input", "contrastive_2layers"]:
                    ssl_features = self.extract_features(ssl_x)
                if ssl_type == "ssl_input":
                    ssl_out.append(ssl(ssl_features))
                elif ssl_type == "contrastive_2layers":
                    ssl_out.append(
                        (ssl(features_0, extract_features=False), ssl(ssl_features))
                    )
                elif ssl_type == "contrastive":
                    ssl_out.append((ssl(features), ssl(ssl_features)))
                elif ssl_type == "ssl_target":
                    ssl_out.append(ssl(features))
        args = [features]
        if tag is not None:
            args.append(tag)
        if self.main_task_active:
            x = self.predictor(*args)
        else:
            x = None
        return x, ssl_out


class LoadedModel(Model):
    """
    A class to generate a Model instance from a torch.nn.Module
    """

    ssl_types = ["none"]

    def __init__(self, model: nn.Module, **kwargs) -> None:
        """
        Parameters
        ----------
        model : torch.nn.Module
            a model with a forward function that takes a single tensor as input and returns a single tensor as output
        """

        super(LoadedModel, self).__init__()
        self.ssl_active = False
        self.feature_extractor = model

    def _feature_extractor(self) -> None:
        """
        Set feature extractor
        """

        pass

    def _predictor(self) -> None:
        """
        Set predictor
        """

        self.predictor = nn.Identity()

    def ssl_on(self):
        """
        Turn SSL on (SSL output will be computed by the forward function)
        """

        pass
