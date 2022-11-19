#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
"""
Abstract parent class for transformers
"""

from typing import Dict, List, Callable, Union, Tuple
import torch
from abc import ABC, abstractmethod
from dlc2action.utils import TensorList
from copy import deepcopy
from matplotlib import pyplot as plt


class Transformer(ABC):
    """
    A base class for all transformers

    A transformer should apply augmentations and generate model input and training target tensors.

    All augmentation functions need to take `(main_input: dict, ssl_inputs: list, ssl_targets: list)`
    as input and return an output of the same format. Here `main_input` is a feature dictionary of the sample
    data, `ssl_inputs` is a list of SSL input feature dictionaries and `ssl_targets` is a list of SSL target
    feature dictionaries. The same augmentations are applied to all inputs and then `None` values are replaced
    according to the rules set by `keep_target_none` and `generate_ssl_input` parameters and the feature
    dictionaries are compiled into tensors.
    """

    def __init__(
        self,
        model_name: str,
        augmentations: List = None,
        use_default_augmentations: bool = False,
        generate_ssl_input: List = None,
        keep_target_none: List = None,
        ssl_augmentations: List = None,
        graph_features: bool = False,
        bodyparts_order: List = None,
    ) -> None:
        """
        Parameters
        ----------
        augmentations : list, optional
            a list of string names of augmentations to use (if not provided, either no augmentations are applied or
            (if use_default_augmentations is True) a default list is used
        use_default_augmentations : bool, default False
            if True and augmentations are not passed, default augmentations will be applied; otherwise no augmentations
        generate_ssl_input : list, optional
            a list of bool values of the length of the number of SSL modules being used; if the corresponding bool value
            is `True`, the ssl input will be generated as a new augmentation of main input (if not provided defaults to
            `False` for each module)
        keep_target_none : list, optional
            a list of bool values of the length of the number of SSL modules being used; if the corresponding bool value
            is `False` and the SSL target is `None`, the target is set to augmented main input (if not provided defaults
            to `True` for each module)
        ssl_augmentations : list, optional
            a list of augmentation names to be applied with generating SSL input (when `generate_ssl_input` is True)
            (if not provided, defaults to the main augmentations list)
        graph_features : bool, default False
            if `True`, all features in each frame can be meaningfully reshaped to `(#bodyparts, #features)`
        bodyparts_order : list, optional
            a list of bodypart names, optional
        """

        if augmentations is None:
            augmentations = []
        if generate_ssl_input is None:
            generate_ssl_input = [None]
        if keep_target_none is None:
            keep_target_none = [None]
        self.model_name = model_name
        self.augmentations = augmentations
        self.generate_ssl_input = generate_ssl_input
        self.keep_target_none = keep_target_none
        if len(self.augmentations) == 0 and use_default_augmentations:
            self.augmentations = self._default_augmentations()
        if ssl_augmentations is None:
            ssl_augmentations = self.augmentations
        self.ssl_augmentations = ssl_augmentations
        self.graph_features = graph_features
        self.num_graph_nodes = (
            len(bodyparts_order) if bodyparts_order is not None else None
        )
        self._check_augmentations(self.augmentations)
        self._check_augmentations(self.ssl_augmentations)

    def transform(
        self,
        main_input: Dict,
        ssl_inputs: List = None,
        ssl_targets: List = None,
        augment: bool = False,
        subsample: List = None,
    ) -> Tuple:
        """
        Apply augmentations and generate tensors from feature dictionaries

        The same augmentations are applied to all the inputs (if they have the features known to the transformer).

        If `generate_ssl_input` is set to True for some of the SSL pairs, those SSL inputs will be generated as
        another augmentation of main_input.
        Unless `keep_target_none` is set to True, `None` SSL targets will be replaced with augmented `main_input`.
        All features are stacked together to form a tensor of shape `(#features, #frames)` that can be passed to
        a model.

        Parameters
        ----------
        main_input : dict
            the feature dictionary of the main input
        ssl_inputs : list, optional
            a list of feature dictionaries of SSL inputs (some or all can be None)
        ssl_targets : list, optional
            a list of feature dictionaries of SSL targets (some or all can be None)
        augment : bool, default True

        Returns
        -------
        main_input : torch.Tensor
            the augmented tensor of the main input
        ssl_inputs : list, optional
            a list of augmented tensors of SSL inputs (some or all can be None)
        ssl_targets : list, optional
            a list of augmented tensors of SSL targets (some or all can be None)
        """

        if subsample is not None:
            original_len = list(main_input.values())[0].shape[-1]
            for key in main_input:
                main_input[key] = main_input[key][..., subsample]
            # subsample_ssl = sorted(random.sample(range(original_len), len(subsample)))
            for x in ssl_inputs + ssl_targets:
                if x is not None:
                    for key in x:
                        if len(x[key].shape) == 3 and x[key].shape[-1] == original_len:
                            x[key] = x[key][..., subsample]
        main_input, ssl_inputs, ssl_targets = self._apply_augmentations(
            main_input, ssl_inputs, ssl_targets, augment
        )
        meta = [None for _ in ssl_inputs]
        for i, x in enumerate(ssl_inputs):
            if type(x) is tuple:
                x, meta_x = x
                meta[i] = meta_x
                ssl_inputs[i] = x
        for (i, ssl_x), generate in zip(enumerate(ssl_inputs), self.generate_ssl_input):
            if ssl_x is None and generate:
                ssl_inputs[i] = self._apply_augmentations(
                    deepcopy(main_input), None, None, augment=True, ssl=True
                )[0]
        output = []
        num_ssl = len(ssl_inputs)
        dicts = [main_input] + ssl_inputs + ssl_targets

        for x in dicts:
            if x is None:
                output.append(None)
            else:
                output.append(self._make_tensor(x, self.model_name))
                # output.append(
                #     torch.cat([x[key] for key in sorted(list(x.keys()))], dim=1)
                # )
        main_input, ssl_inputs, ssl_targets = (
            output[0],
            output[1 : num_ssl + 1],
            output[num_ssl + 1 :],
        )
        for (i, ssl_x), keep in zip(enumerate(ssl_targets), self.keep_target_none):
            if not keep and ssl_x is None:
                ssl_targets[i] = main_input
        for i, meta_x in enumerate(meta):
            if meta_x is not None:
                ssl_inputs[i] = (ssl_inputs[i], meta_x)
        return main_input, ssl_inputs, ssl_targets

    @abstractmethod
    def _augmentations_dict(self) -> Dict:
        """
        Return a dictionary of possible augmentations

        The keys are augmentation names and the values are the corresponding functions

        Returns
        -------
        augmentations_dict : dict
            a dictionary of augmentation functions (each function needs to take
            `(main_input: dict, ssl_inputs: list, ssl_targets: list)` as input and return an output
            of the same format)
        """

    @abstractmethod
    def _default_augmentations(self) -> List:
        """
        Return a list of default augmentation names

        In case an augmentation list is not provided to the class constructor and use_default_augmentations is `True`,
        this function is called to set the augmentations parameter. The elements of the list have to be keys of the
        dictionary returned by `self._augmentations_dict()`

        Returns
        -------
        default_augmentations : list
            a list of string names of the default augmentation functions
        """

    def _check_augmentations(self, augmentations: List) -> None:
        """
        Check the validity of an augmentations list
        """

        for aug_name in augmentations:
            if aug_name not in self._augmentations_dict():
                raise ValueError(
                    f"The {aug_name} augmentation is not possible in this augmentor! Please choose from {list(self._augmentations_dict().keys())}"
                )

    def _get_augmentation(self, aug_name: str) -> Callable:
        """
        Return the augmentation specified by `aug_name`
        """

        return self._augmentations_dict()[aug_name]

    def _visualize(self, main_input, title):
        coord_keys = [x for x in main_input.keys() if x.split("---")[0] == "coords"]
        if len(coord_keys) > 0:
            coords = [main_input[key][0, :, :, 0].detach().cpu() for key in coord_keys]
            centers = [None]
        else:
            coord_keys = [
                x for x in main_input.keys() if x.split("---")[0] == "coord_diff"
            ]
            coords = []
            centers = []
            for coord_diff_key in coord_keys:
                if len(coord_diff_key.split("---")) == 2:
                    ind = coord_diff_key.split("---")[1]
                    center_key = f"center---{ind}"
                else:
                    center_key = "center"
                    ind = ""
                if center_key in main_input.keys():
                    coords.append(
                        main_input[center_key][0, :, :, 0].detach().cpu()
                        + main_input[coord_diff_key][0, :, :, 0].detach().cpu()
                    )
                    title += f", {ind}: {main_input[center_key][0, 0, :, 0].data}"
                    center = main_input[center_key][0, :, :, 0].detach().cpu()
                else:
                    coords.append(main_input[coord_diff_key][0, :, :, 0].detach().cpu())
                    center = None
                centers.append(center)
        colors = ["blue", "orange", "green", "purple", "pink"]
        if coords[0].shape[1] == 2:
            plt.figure(figsize=(15, 15))
        else:
            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(projection="3d")
        for i, coord in enumerate(coords):
            if coord.shape[1] == 2:
                plt.scatter(coord[:, 0], coord[:, 1], color=colors[i])
                plt.xlim((-0.5, 0.5))
                plt.ylim((-0.5, 0.5))
            else:
                ax.scatter(
                    coord[:, 0].detach().cpu(),
                    coord[:, 1].detach().cpu(),
                    coord[:, 2].detach().cpu(),
                    color=colors[i],
                )
                if centers[i] is not None:
                    ax.scatter(
                        centers[i][:, 0].detach().cpu(),
                        centers[i][:, 1].detach().cpu(),
                        0,
                        color="red",
                        s=30,
                    )
                    center = centers[i][0].detach().cpu()
                    ax.text(
                        center[0], center[1], 0, f"({center[0]:.2f}, {center[1]:.2f})"
                    )
                for i in [1, 8]:
                    ax.scatter(
                        coord[i : i + 1, 0].detach().cpu(),
                        coord[i : i + 1, 1].detach().cpu(),
                        coord[i : i + 1, 2].detach().cpu(),
                        color="purple",
                    )
                    ax.text(
                        coord[i, 0],
                        coord[i, 1],
                        coord[i, 2],
                        f"({coord[i, 0]:.2f}, {coord[i, 1]:.2f}, {coord[i, 2]:.2f})",
                    )

        plt.xlim((-3, 3))
        plt.ylim((-3, 3))
        ax.set_zlim(-2, 4)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.title(title)
        plt.show()

    def _apply_augmentations(
        self,
        main_input: Dict,
        ssl_inputs: List = None,
        ssl_targets: List = None,
        augment: bool = False,
        ssl: bool = False,
    ) -> Tuple:
        """
        Apply the augmentations

        The same augmentations are applied to all inputs
        """

        visualize = False
        if ssl:
            augmentations = self.ssl_augmentations
        else:
            augmentations = self.augmentations
        if visualize:
            self._visualize(main_input, "before")
        if ssl_inputs is None:
            ssl_inputs = [None]
        if ssl_targets is None:
            ssl_targets = [None]
        if augment:
            for aug_name in augmentations:
                augment_func = self._get_augmentation(aug_name)
                main_input, ssl_inputs, ssl_targets = augment_func(
                    main_input, ssl_inputs, ssl_targets
                )
                if visualize:
                    self._visualize(main_input, aug_name)
        return main_input, ssl_inputs, ssl_targets

    def _make_tensor(self, x: Dict, model_name: str) -> Union[torch.Tensor, TensorList]:
        """
        Turn a feature dictionary into a tensor or a `dlc2action.utils.TensorList` object
        """

        if model_name == "ms_tcn_p":
            keys = sorted(list(x.keys()))
            groups = [key.split("---")[-1] for key in keys]
            unique_groups = sorted(set(groups))
            tensor = TensorList()
            for group in unique_groups:
                if not self.graph_features:
                    tensor.append(
                        torch.cat(
                            [x[key] for key, g in zip(keys, groups) if g == group],
                            dim=2,
                        )
                    )
                else:
                    tensor.append(
                        torch.cat(
                            [
                                x[key].reshape(
                                    (
                                        x[key].shape[0],
                                        self.num_graph_nodes,
                                        -1,
                                        x[key].shape[-1],
                                    )
                                )
                                for key, g in zip(keys, groups)
                                if g == group
                            ],
                            dim=2,
                        )
                    )
                    tensor[-1] = tensor[-1].reshape(
                        (tensor[-1].shape[0], -1, tensor[-1].shape[-1])
                    )
            if "loaded" in x:
                tensor.append(x["loaded"])
        elif model_name == "c2f_tcn_p":
            keys = sorted(
                [
                    key
                    for key in x.keys()
                    if len(key.split("---")) != 1
                    and len(key.split("---")[-1].split("+")) != 2
                ]
            )
            inds = [key.split("---")[-1] for key in keys]
            unique_inds = sorted(set(inds))
            tensor = TensorList()
            for ind in unique_inds:
                if not self.graph_features:
                    tensor.append(
                        torch.cat(
                            [x[key] for key, g in zip(keys, inds) if g == ind],
                            dim=1,
                        )
                    )
                else:
                    tensor.append(
                        torch.cat(
                            [
                                x[key].reshape(
                                    (
                                        x[key].shape[0],
                                        self.num_graph_nodes,
                                        -1,
                                        x[key].shape[-1],
                                    )
                                )
                                for key, g in zip(keys, inds)
                                if g == ind
                            ],
                            dim=1,
                        )
                    )
                    tensor[-1] = tensor[-1].reshape(
                        (tensor[-1].shape[0], -1, tensor[-1].shape[-1])
                    )
        elif model_name == "c3d_a":
            tensor = torch.cat([x[key] for key in sorted(list(x.keys()))], dim=1)
        else:
            if not self.graph_features:
                tensor = torch.cat([x[key] for key in sorted(list(x.keys()))], dim=1)
            else:
                tensor = torch.cat(
                    [
                        x[key].reshape(
                            (
                                x[key].shape[0],
                                self.num_graph_nodes,
                                -1,
                                x[key].shape[-1],
                            )
                        )
                        for key in sorted(list(x.keys()))
                    ],
                    dim=2,
                )
                tensor = tensor.reshape((tensor.shape[0], -1, tensor.shape[-1]))
        return tensor


class EmptyTransformer(Transformer):
    """
    Empty transformer class that does not apply augmentations
    """

    def _augmentations_dict(self) -> Dict:
        """
        Return a dictionary of possible augmentations

        The keys are augmentation names and the values are the corresponding functions

        Returns
        -------
        augmentations_dict : dict
            a dictionary of augmentation functions
        """

        return {}

    def _default_augmentations(self) -> List:
        """
        Return a list of default augmentation names

        In case an augmentation list is not provided to the class constructor and use_default_augmentations is True,
        this function is called to set the augmentations parameter. The elements of the list have to be keys of the
        dictionary returned by `self._augmentations_dict()`

        Returns
        -------
        default_augmentations : list
            a list of string names of the default augmentation functions
        """

        return []
