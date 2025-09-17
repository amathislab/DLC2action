#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
"""
Kinematic transformer
"""

from typing import Dict, Tuple, List
from torchvision import transforms as tr
from dlc2action.transformer.base_transformer import Transformer


class HeatmapTransformer(Transformer):
    """
    A transformer that augments the output of the Heatmap feature extractor

    The available augmentations are `'rotate'`, `'horizontal_flip'`, `'vertical_flip'`.
    """

    def __init__(
        self,
        model_name: str,
        augmentations: List = None,
        use_default_augmentations: bool = False,
        rotation_degree_limits: List = None,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        model_name : str
            the name of the model used
        augmentations : list, optional
            list of augmentation names to use ("rotate", "mirror", "shift")
        use_default_augmentations : bool, default False
            if `True` and augmentations are not passed, default augmentations will be applied; otherwise no augmentations
        rotation_degree_limits : list, default [-90, 90]
            list of float rotation angle limits (`[low, high]`)
        **kwargs : dict
            other parameters for the base transformer class
        """

        if augmentations is None:
            augmentations = []
        if rotation_degree_limits is None:
            rotation_degree_limits = [-90, 90]
        super().__init__(
            model_name,
            augmentations,
            use_default_augmentations,
            graph_features=False,
            bodyparts_order=None,
        )
        self.rotation_limits = rotation_degree_limits

    def _apply_transform(
        self, transformation, main_input: Dict, ssl_inputs: List, ssl_targets: List
    ) -> (Dict, List, List):
        """
        Apply a `torchvision.transforms` transformation to the data
        """

        dicts = [main_input] + ssl_inputs + ssl_targets
        for x in dicts:
            if x is not None:
                keys = self._get_keys(
                    ("coords_heatmap", "motion_heatmap"),
                    x,
                )
                for key in keys:
                    if key in x:
                        x[key] = transformation(x[key])
        return main_input, ssl_inputs, ssl_targets

    def _horizontal_flip(
        self, main_input: Dict, ssl_inputs: List, ssl_targets: List
    ) -> (Dict, List, List):
        """
        Apply a random horizontal flip
        """

        transform = tr.RandomHorizontalFlip()
        return self._apply_transform(transform, main_input, ssl_inputs, ssl_targets)

    def _vertical_flip(
        self, main_input: Dict, ssl_inputs: List, ssl_targets: List
    ) -> (Dict, List, List):
        """
        Apply a random vertical flip
        """

        transform = tr.RandomVerticalFlip()
        return self._apply_transform(transform, main_input, ssl_inputs, ssl_targets)

    def _rotate(
        self, main_input: Dict, ssl_inputs: List, ssl_targets: List
    ) -> (Dict, List, List):
        """
        Apply a random rotation
        """

        transform = tr.RandomRotation(self.rotation_limits)
        return self._apply_transform(transform, main_input, ssl_inputs, ssl_targets)

    def _augmentations_dict(self) -> Dict:
        """
        Get the mapping from augmentation names to functions
        """

        return {
            "rotate": self._rotate,
            "horizontal_flip": self._horizontal_flip,
            "vertical_flip": self._vertical_flip,
        }

    def _default_augmentations(self) -> List:
        """
        Get the list of default augmentation names
        """

        return ["horizontal_flip"]

    def _get_bodyparts(self, shape: Tuple) -> None:
        """
        Set the number of bodyparts from the data if it is not known
        """

        if self.n_bodyparts is None:
            N, B, F = shape
            self.n_bodyparts = B // 2

    def _get_keys(self, key_bases: Tuple, x: Dict) -> List:
        """
        Get the keys of x that start with one of the strings from key_bases
        """

        keys = []
        for key in x:
            if key.startswith(key_bases):
                keys.append(key)
        return keys

    def _apply_augmentations(
        self,
        main_input: Dict,
        ssl_inputs: List = None,
        ssl_targets: List = None,
        augment: bool = False,
        ssl: bool = False,
    ) -> Tuple:
        dicts = [main_input] + ssl_inputs + ssl_targets
        key = self._get_keys(
            ("coords_heatmap", "motion_heatmap"),
            main_input,
        )[0]
        self.original_shape = main_input[key].shape
        for x in dicts:
            if x is not None:
                keys = self._get_keys(
                    ("coords_heatmap", "motion_heatmap"),
                    x,
                )
                for key in keys:
                    x[key] = x[key].reshape((-1, x[key].shape[-2], x[key].shape[-1]))
        main_input, ssl_inputs, ssl_targets = super()._apply_augmentations(
            main_input, ssl_inputs, ssl_targets, augment=augment, ssl=ssl
        )
        dicts = [main_input] + ssl_inputs + ssl_targets
        for x in dicts:
            if x is not None:
                keys = self._get_keys(
                    ("coords_heatmap", "motion_heatmap"),
                    x,
                )
                for key in keys:
                    x[key] = x[key].reshape(self.original_shape)
        return main_input, ssl_inputs, ssl_targets
