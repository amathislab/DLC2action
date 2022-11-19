#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
"""
Kinematic transformer
"""

from typing import Dict, Tuple, List, Set, Iterable
import torch
import numpy as np
from copy import copy, deepcopy
from random import getrandbits
from collections import defaultdict
from dlc2action.transformer.base_transformer import Transformer
from dlc2action.utils import rotation_matrix_2d, rotation_matrix_3d


class KinematicTransformer(Transformer):
    """
    A transformer that augments the output of the Kinematic feature extractor

    The available augmentations are `'rotate'`, `'mirror'`, `'shift'`, `'add_noise'` and `'zoom'`
    """

    def __init__(
        self,
        model_name: str,
        augmentations: List = None,
        use_default_augmentations: bool = False,
        rotation_limits: List = None,
        mirror_dim: Set = None,
        noise_std: float = 0.05,
        zoom_limits: List = None,
        masking_probability: float = 0.05,
        dim: int = 2,
        graph_features: bool = False,
        bodyparts_order: List = None,
        canvas_shape: List = None,
        move_around_image_center: bool = True,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        augmentations : list, optional
            list of augmentation names to use ("rotate", "mirror", "shift", "add_noise", "zoom")
        use_default_augmentations : bool, default False
            if `True` and augmentations are not passed, default augmentations will be applied; otherwise no augmentations
        rotation_limits : list, default [-pi/2, pi/2]
            list of float rotation angle limits (`[low, high]``, or `[[low_x, high_x], [low_y, high_y], [low_z, high_z]]`
            for 3D data)
        mirror_dim : set, default {0}
            set of integer indices of dimensions that can be mirrored
        noise_std : float, default 0.05
            standard deviation of noise
        zoom_limits : list, default [0.5, 1.5]
            list of float zoom limits ([low, high])
        masking_probability : float, default 0.1
            the probability of masking a joint
        dim : int, default 2
            the dimensionality of the input data
        **kwargs : dict
            other parameters for the base transformer class
        """

        if augmentations is None:
            augmentations = []
        if canvas_shape is None:
            canvas_shape = [1, 1]
        self.dim = int(dim)

        self.offset = [0 for _ in range(self.dim)]
        self.scale = canvas_shape[1] / canvas_shape[0]
        self.image_center = move_around_image_center
        # if canvas_shape is None:
        #     self.offset = [0.5 for _ in range(self.dim)]
        # else:
        #     self.offset = [0.5 * canvas_shape[i] / canvas_shape[0] for i in range(self.dim)]

        self.blank = 0  # the value that nan values are set to (shouldn't be changed in augmentations)
        super().__init__(
            model_name,
            augmentations,
            use_default_augmentations,
            graph_features=graph_features,
            bodyparts_order=bodyparts_order,
            **kwargs,
        )
        if rotation_limits is None:
            rotation_limits = [-np.pi / 2, np.pi / 2]
        if mirror_dim is None:
            mirror_dim = [0]
        if zoom_limits is None:
            zoom_limits = [0.5, 1.5]
        self.rotation_limits = rotation_limits
        self.n_bodyparts = None
        self.mirror_dim = mirror_dim
        self.noise_std = noise_std
        self.zoom_limits = zoom_limits
        self.masking_probability = masking_probability

    def _apply_augmentations(
        self,
        main_input: Dict,
        ssl_inputs: List = None,
        ssl_targets: List = None,
        augment: bool = False,
        ssl: bool = False,
    ) -> Tuple:

        if ssl_targets is None:
            ssl_targets = [None]
        if ssl_inputs is None:
            ssl_inputs = [None]

        keys_main = self._get_keys(
            (
                "coords",
                "speed_joints",
                "speed_direction",
                "speed_bones",
                "acc_bones",
                "bones",
                "coord_diff",
            ),
            main_input,
        )
        if len(keys_main) > 0:
            key = keys_main[0]
            final_shape = main_input[key].shape
            # self._get_bodyparts(final_shape)
            batch = final_shape[0]
            s = main_input[key].reshape((batch, -1, self.dim, final_shape[-1]))
            x_shape = s.shape
            # print(f'{x_shape=}, {self.dim=}')
            self.n_bodyparts = x_shape[1]
            if self.dim == 3:
                if len(self.rotation_limits) == 2:
                    self.rotation_limits = [[0, 0], [0, 0], self.rotation_limits]
            dicts = [main_input] + ssl_inputs + ssl_targets
            for x in dicts:
                if x is not None:
                    keys = self._get_keys(
                        (
                            "coords",
                            "speed_joints",
                            "speed_direction",
                            "speed_bones",
                            "acc_bones",
                            "bones",
                            "coord_diff",
                        ),
                        x,
                    )
                    for key in keys:
                        x[key] = x[key].reshape(x_shape)
        if len(self._get_keys(("center"), main_input)) > 0:
            dicts = [main_input] + ssl_inputs + ssl_targets
            for x in dicts:
                if x is not None:
                    key_bases = ["center"]
                    keys = self._get_keys(
                        key_bases,
                        x,
                    )
                    for key in keys:
                        x[key] = x[key].reshape(
                            (x[key].shape[0], 1, -1, x[key].shape[-1])
                        )
        main_input, ssl_inputs, ssl_targets = super()._apply_augmentations(
            main_input, ssl_inputs, ssl_targets, augment
        )
        if len(keys_main) > 0:
            dicts = [main_input] + ssl_inputs + ssl_targets
            for x in dicts:
                if x is not None:
                    keys = self._get_keys(
                        (
                            "coords",
                            "speed_joints",
                            "speed_direction",
                            "speed_bones",
                            "acc_bones",
                            "bones",
                            "coord_diff",
                        ),
                        x,
                    )
                    for key in keys:
                        x[key] = x[key].reshape(final_shape)
        if len(self._get_keys(("center"), main_input)) > 0:
            dicts = [main_input] + ssl_inputs + ssl_targets
            for x in dicts:
                if x is not None:
                    key_bases = ["center"]
                    keys = self._get_keys(
                        key_bases,
                        x,
                    )
                    for key in keys:
                        x[key] = x[key].reshape((x[key].shape[0], -1, x[key].shape[-1]))
        return main_input, ssl_inputs, ssl_targets

    def _rotate(
        self, main_input: Dict, ssl_inputs: List, ssl_targets: List
    ) -> Tuple[Dict, List, List]:
        """
        Rotate the "coords" and "speed_joints" features of the input to a random degree
        """

        keys = self._get_keys(
            (
                "coords",
                "bones",
                "coord_diff",
                "center",
                "speed_joints",
                "speed_direction",
                "center",
            ),
            main_input,
        )
        if len(keys) == 0:
            return main_input, ssl_inputs, ssl_targets
        batch = main_input[keys[0]].shape[0]
        if self.dim == 2:
            angles = torch.FloatTensor(batch).uniform_(*self.rotation_limits)
            R = rotation_matrix_2d(angles).to(main_input[keys[0]].device)
        else:
            angles_x = torch.FloatTensor(batch).uniform_(*self.rotation_limits[0])
            angles_y = torch.FloatTensor(batch).uniform_(*self.rotation_limits[1])
            angles_z = torch.FloatTensor(batch).uniform_(*self.rotation_limits[2])
            R = rotation_matrix_3d(angles_x, angles_y, angles_z).to(
                main_input[keys[0]].device
            )
        dicts = [main_input] + ssl_inputs + ssl_targets
        for x in dicts:
            if x is not None:
                keys = self._get_keys(
                    (
                        "coords",
                        "speed_joints",
                        "speed_direction",
                        "speed_bones",
                        "acc_bones",
                        "bones",
                        "coord_diff",
                        "center",
                    ),
                    x,
                )
                for key in keys:
                    if key in x:
                        mask = x[key] == self.blank
                        if (
                            any([key.startswith(x) for x in ["coords", "center"]])
                            and not self.image_center
                        ):
                            offset = x[key].mean(1).unsqueeze(1)
                        else:
                            offset = 0
                        x[key] = (
                            torch.einsum(
                                "abjd,aij->abid",
                                x[key] - offset,
                                R,
                            )
                            + offset
                        )
                        x[key][mask] = self.blank
        return main_input, ssl_inputs, ssl_targets

    def _mirror(
        self, main_input: Dict, ssl_inputs: List, ssl_targets: List
    ) -> Tuple[Dict, List, List]:
        """
        Mirror the "coords" and "speed_joints" features of the input randomly
        """

        mirror = []
        for i in range(3):
            if i in self.mirror_dim and bool(getrandbits(1)):
                mirror.append(i)
        if len(mirror) == 0:
            return main_input, ssl_inputs, ssl_targets
        dicts = [main_input] + ssl_inputs + ssl_targets
        for x in dicts:
            if x is not None:
                keys = self._get_keys(
                    (
                        "coords",
                        "speed_joints",
                        "speed_direction",
                        "speed_bones",
                        "acc_bones",
                        "bones",
                        "center",
                        "coord_diff",
                    ),
                    x,
                )
                for key in keys:
                    if key in x:
                        mask = x[key] == self.blank
                        y = deepcopy(x[key])
                        if not self.image_center and any(
                            [key.startswith(x) for x in ["coords", "center"]]
                        ):
                            mean = y.mean(1).unsqueeze(1)
                            for dim in mirror:
                                y[:, :, dim, :] = (
                                    2 * mean[:, :, dim, :] - y[:, :, dim, :]
                                )
                        else:
                            for dim in mirror:
                                y[:, :, dim, :] *= -1
                        x[key] = y
                        x[key][mask] = self.blank
        return main_input, ssl_inputs, ssl_targets

    def _shift(
        self, main_input: Dict, ssl_inputs: List, ssl_targets: List
    ) -> Tuple[Dict, List, List]:
        """
        Shift the "coords" features of the input randomly
        """

        keys = self._get_keys(("coords", "center"), main_input)
        if len(keys) == 0:
            return main_input, ssl_inputs, ssl_targets
        batch = main_input[keys[0]].shape[0]
        dim = main_input[keys[0]].shape[2]
        device = main_input[keys[0]].device
        coords = torch.cat(
            [main_input[key].transpose(-1, -2).reshape(batch, -1, dim) for key in keys],
            dim=1,
        )
        minimask = coords[:, :, 0] != self.blank
        min_x = -0.5 - torch.min(coords[:, :, 0], dim=-1)[0]
        min_y = -0.5 * self.scale - torch.min(coords[:, :, 1], dim=-1)[0]
        max_x = 0.5 - torch.max(coords[:, :, 0][minimask], dim=-1)[0]
        max_y = 0.5 * self.scale - torch.max(coords[:, :, 1][minimask], dim=-1)[0]
        del coords
        shift_x = min_x + torch.FloatTensor(batch).uniform_().to(device) * (
            max_x - min_x
        )
        shift_y = min_y + torch.FloatTensor(batch).uniform_().to(device) * (
            max_y - min_y
        )
        shift_x = shift_x.unsqueeze(-1).unsqueeze(-1)
        shift_y = shift_y.unsqueeze(-1).unsqueeze(-1)
        dicts = [main_input] + ssl_inputs + ssl_targets
        for x in dicts:
            if x is not None:
                keys = self._get_keys(("coords", "center"), x)
                for key in keys:
                    y = deepcopy(x[key])
                    mask = y == self.blank
                    y[:, :, 0, :] += shift_x
                    y[:, :, 1, :] += shift_y
                    x[key] = y
                    x[key][mask] = self.blank
        return main_input, ssl_inputs, ssl_targets

    def _add_noise(
        self, main_input: Dict, ssl_inputs: List, ssl_targets: List
    ) -> Tuple[Dict, List, List]:
        """
        Add normal noise to all features of the input
        """

        dicts = [main_input] + ssl_inputs + ssl_targets
        for x in dicts:
            if x is not None:
                keys = self._get_keys(
                    (
                        "coords",
                        "speed_joints",
                        "speed_direction",
                        "intra_distance",
                        "acc_joints",
                        "angle_speeds",
                        "inter_distance",
                        "speed_bones",
                        "acc_bones",
                        "bones",
                        "coord_diff",
                        "center",
                    ),
                    x,
                )
                for key in keys:
                    mask = x[key] == self.blank
                    x[key] = x[key] + torch.FloatTensor(x[key].shape).normal_(
                        std=self.noise_std
                    ).to(x[key].device)
                    x[key][mask] = self.blank
        return main_input, ssl_inputs, ssl_targets

    def _zoom(
        self, main_input: Dict, ssl_inputs: List, ssl_targets: List
    ) -> Tuple[Dict, List, List]:
        """
        Add random zoom to all features of the input
        """

        key = list(main_input.keys())[0]
        batch = main_input[key].shape[0]
        device = main_input[key].device
        zoom = torch.FloatTensor(batch).uniform_(*self.zoom_limits).to(device)
        dicts = [main_input] + ssl_inputs + ssl_targets
        for x in dicts:
            if x is not None:
                keys = self._get_keys(
                    (
                        "speed_joints",
                        "intra_distance",
                        "acc_joints",
                        "inter_distance",
                        "speed_bones",
                        "acc_bones",
                        "bones",
                        "coord_diff",
                        "center",
                        "speed_value",
                    ),
                    x,
                )
                for key in keys:
                    mask = x[key] == self.blank
                    shape = np.array(x[key].shape)
                    shape[1:] = 1
                    y = deepcopy(x[key])
                    y *= zoom.reshape(list(shape))
                    x[key] = y
                    x[key][mask] = self.blank
                keys = self._get_keys("coords", x)
                for key in keys:
                    mask = x[key] == self.blank
                    shape = np.array(x[key].shape)
                    shape[1:] = 1
                    zoom = zoom.reshape(list(shape))
                    x[key][mask] = 10
                    min_x = x[key][:, :, 0, :].min(1)[0].unsqueeze(1)
                    min_y = x[key][:, :, 1, :].min(1)[0].unsqueeze(1)
                    center = torch.stack([min_x, min_y], dim=2)
                    coords = (x[key] - center) * zoom + center
                    x[key] = coords
                    x[key][mask] = self.blank

        return main_input, ssl_inputs, ssl_targets

    def _mask_joints(
        self, main_input: Dict, ssl_inputs: List, ssl_targets: List
    ) -> Tuple[Dict, List, List]:
        """
        Mask joints randomly
        """

        key = list(main_input.keys())[0]
        batch, *_, frames = main_input[key].shape
        masked_joints = (
            torch.FloatTensor(batch, self.n_bodyparts).uniform_()
            < self.masking_probability
        )
        dicts = [main_input] + ssl_inputs + ssl_targets
        for x in [y for y in dicts if y is not None]:
            keys = self._get_keys(("intra_distance", "inter_distance"), x)
            for key in keys:
                mask = (
                    masked_joints.repeat(self.n_bodyparts, frames, 1, 1)
                    .transpose(0, 2)
                    .transpose(1, 3)
                )
                indices = torch.triu_indices(self.n_bodyparts, self.n_bodyparts, 1)

                X = torch.zeros((batch, self.n_bodyparts, self.n_bodyparts, frames)).to(
                    x[key].device
                )
                X[:, indices[0], indices[1], :] = x[key]
                X[mask] = self.blank
                X[mask.transpose(1, 2)] = self.blank
                x[key] = X[:, indices[0], indices[1], :].reshape(batch, -1, frames)
            keys = self._get_keys(
                (
                    "speed_joints",
                    "speed_direction",
                    "coords",
                    "acc_joints",
                    "speed_bones",
                    "acc_bones",
                    "bones",
                    "coord_diff",
                ),
                x,
            )
            for key in keys:
                mask = (
                    masked_joints.repeat(self.dim, frames, 1, 1)
                    .transpose(0, 2)
                    .transpose(1, 3)
                )
                x[key][mask] = (
                    x[key].mean(1).unsqueeze(1).repeat(1, x[key].shape[1], 1, 1)[mask]
                )
            keys = self._get_keys("angle_speeds", x)
            for key in keys:
                mask = (
                    masked_joints.repeat(frames, 1, 1).transpose(0, 1).transpose(1, 2)
                )
                x[key][mask] = (
                    x[key].mean(1).unsqueeze(1).repeat(1, x[key].shape[1], 1)[mask]
                )

        return main_input, ssl_inputs, ssl_targets

    def _switch(
        self, main_input: Dict, ssl_inputs: List, ssl_targets: List
    ) -> Tuple[Dict, List, List]:
        if bool(getrandbits(1)):
            return main_input, ssl_inputs, ssl_targets
        individuals = set()
        ind_dict = defaultdict(lambda: set())
        for key in main_input:
            if len(key.split("---")) != 2:
                continue
            ind = key.split("---")[1]
            if "+" in ind:
                continue
            individuals.add(ind)
            ind_dict[ind].add(key.split("---")[0])
        individuals = list(individuals)
        if len(individuals) < 2:
            return main_input, ssl_inputs, ssl_targets
        for x, y in zip(individuals[::2], individuals[1::2]):
            for key in ind_dict[x]:
                if key in ind_dict[y]:
                    main_input[f"{key}---{x}"], main_input[f"{key}---{y}"] = (
                        main_input[f"{key}---{y}"],
                        main_input[f"{key}---{x}"],
                    )
                for d in ssl_inputs + ssl_targets:
                    if d is None:
                        continue
                    if f"{key}---{x}" in d and f"{key}---{y}" in d:
                        d[f"{key}---{x}"], d[f"{key}---{y}"] = (
                            d[f"{key}---{y}"],
                            d[f"{key}---{x}"],
                        )
        return main_input, ssl_inputs, ssl_targets

    def _get_keys(self, key_bases: Iterable, x: Dict) -> List:
        """
        Get the keys of x that start with one of the strings from key_bases
        """

        keys = []
        if isinstance(key_bases, str):
            key_bases = [key_bases]
        for key in x:
            if any([x == key.split("---")[0] for x in key_bases]):
                keys.append(key)
        return keys

    def _augmentations_dict(self) -> Dict:
        """
        Get the mapping from augmentation names to functions
        """

        return {
            "mirror": self._mirror,
            "shift": self._shift,
            "add_noise": self._add_noise,
            "zoom": self._zoom,
            "rotate": self._rotate,
            "mask": self._mask_joints,
            "switch": self._switch,
        }

    def _default_augmentations(self) -> List:
        """
        Get the list of default augmentation names
        """

        return ["mirror", "shift", "add_noise"]

    def _get_bodyparts(self, shape: Tuple) -> None:
        """
        Set the number of bodyparts from the data if it is not known
        """

        if self.n_bodyparts is None:
            N, B, F = shape
            self.n_bodyparts = B // 2
