#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
"""
## Feature extraction

Feature extractors generate feature dictionaries that are then passed to SSL transformations
(see `dlc2action.ssl`) and finally to
transformers that perform augmentations and merge all features into a tensor (see `dlc2action.transformer`).
The keys of the dictionaries are the feature names (`'coords'`, `'speeds'` and so on) and the values are the
feature tensors. It is generally assumed that the tensors have shape `(F, ..., L)` where `F` is the variable
number of features (per frame, keypoint, pixel...) and `L` is the length of the segment in frames. The `F`
value can be different for every tensor in the dictionary and the rest of the shape should be constant.
"""
import copy
from typing import Dict, Tuple, List, Set
import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial.distance import cdist
import math
from itertools import combinations
from matplotlib.cm import get_cmap
from dlc2action.data.base_store import PoseInputStore
from scipy.ndimage.filters import gaussian_filter


class FeatureExtractor(ABC):
    """
    The base class for feature extractors

    The `extract_features` method receives a data dictionary as input.
    We do not assume a specific
    structure in the values and all necessary information (coordinates of a bodypart, number
    of frames, list of bodyparts) is inferred using input store methods. Therefore, each child class
    of `FeatureExtractor` is written for a specific subclass of `dlc2action.data.base_Store.InputStore`
    with the data inference
    functions defined (i.e. `dlc2action.data.base_store.PoseInputStore`).
    """

    input_store_class = None
    """
    The `dlc2action.data.base_Store.InputStore` child class paired with this feature extractor
    """

    @abstractmethod
    def __init__(self, ignored_clips: List = None, **kwargs):
        """
        Parameters
        ----------
        ignored_clips : list
            a list of string names of clip ids to ignore
        """

    @abstractmethod
    def extract_features(
        self, data_dict: Dict, video_id: str, one_clip: bool = False
    ) -> Dict:
        """
        Extract features from a data dictionary

        An input store will call this method while pre-computing a dataset. The data dictionary has to relate to one
        video id and have clip ids as keys. Read the documentation at `dlc2action.data` to find out more about video
        and clip ids. We do not assume a specific
        structure in the values, so all necessary information (coordinates of a bodypart, number
        of frames, list of bodyparts) is inferred using input store methods.

        Parameters
        ----------
        data_dict : dict
            the data dictionary
        video_id : str
            the id of the video associated with the data dictionary
        one_clip : bool, default False
            if `True`, all features will be concatenated and assigned to one clip named `'all'`

        Returns
        -------
        features : dict
            a features dictionary where the keys are the feature names (e.g. 'coords', 'distances') and the
            values are numpy arrays of shape `(#features, ..., #frames)`
        """


class PoseFeatureExtractor(FeatureExtractor):
    """
    The base class for pose feature extractors

    Pose feature extractors work with `dlc2action.data.base_store.InputStore` instances
    that inherit from `dlc2action.data.base_store.PoseInputStore`.
    """

    input_store_class = PoseInputStore

    def __init__(self, input_store: PoseInputStore, *args, **kwargs):
        """
        Parameters
        ----------
        input_store : PoseInputStore
            the input store object
        """

        self.get_bodyparts = input_store.get_bodyparts
        self.get_coords = input_store.get_coords
        self.get_n_frames = input_store.get_n_frames
        self.get_likelihood = input_store.get_likelihood


# class KinematicBones(PoseFeatures):
#
#     def __init__(self, dataset, bone_pairs, *args, **kwargs):
#         self.bone_starts, self.bone_ends = zip(*bone_pairs)
#         self.keys = ["bones", "speed_bones", "acc_bones"]
#         super().__init__(dataset)
#
#     def extract_features(self, data_dict: Dict, clip_id: str, name: str) -> Dict:
#         if isinstance(clip_id, list):
#             clip_id = clip_id[0]
#         bodyparts = np.array(self.get_bodyparts(data_dict, clip_id))
#         bone_starts = np.where(
#             np.array(self.bone_starts)[:, None] == bodyparts[None, :]
#         )[1]
#         bone_ends = np.where(np.array(self.bone_ends)[:, None] == bodyparts[None, :])[1]
#         coords = np.stack(
#             [self.get_coords(data_dict, clip_id, bp) for bp in bodyparts], axis=1
#         )
#         bones = coords[:, bone_ends, :] - coords[:, bone_starts, :]
#         speeds = bones[1:] - bones[:-1]
#         speeds = np.concatenate([speeds[:1], speeds], axis=0)
#         acc = speeds[1:] - speeds[:-1]
#         acc = np.concatenate([acc[:1], acc], axis=0)
#         n_frames = bones.shape[0]
#         features = {
#             "bones": bones.reshape((n_frames, -1)),
#             "speed_bones": speeds.reshape((n_frames, -1)),
#             "acc_bones": acc.reshape((n_frames, -1)),
#         }
#         return features


class KinematicExtractor(PoseFeatureExtractor):
    """
    A feature extractor for basic kinematic features: speeds, accelerations, distances.

    The available keys are:
        - coords: the allocentric bodypart coordinates,
        - coord_diff: the egocentric bodypart coordinates,
        - center: the body center (mean of bodyparts) coordinates,
        - intra_distance: distances between bodyparts (pairs set in `distance_pairs` or all combinations by default),
        - inter_distance: computed in interactive mode (for pairs of animals); distances from each bodypart of each animal to the centroid between them,
        - speed_direction: unit vector of speed approximation for each bodypart,
        - speed_value: l2 norm of the speed approximation vector for each bodypart,
        - acc_joints: l2 norm of the acceleration approximation vector for each bodypart,
        - angle_speeds: vector of angle speed approximation for each bodypart,
        - angles: cosines of angles set in `angle_pairs`,
        - areas: areas of polygons set in `area_vertices`,
        - zone_bools: binary identifier of zone visitation, defined in `zone_bools`,
        - zone_distances: distance to zone boundary, defined in `zone_distances'`,
        - likelihood: pose estimation likelihood (if known).

    The default set is `{coord_diff, center, intra_distance, inter_distance, speed_direction, speed_value, acc_joints, angle_speeds}`
    """

    def __init__(
        self,
        input_store: PoseInputStore,
        keys: List = None,
        ignored_clips: List = None,
        interactive: bool = False,
        averaging_window: int = 1,
        distance_pairs: List = None,
        angle_pairs: List = None,
        neighboring_frames: int = 0,
        area_vertices: List = None,
        zone_vertices: Dict = None,
        zone_bools: List = None,
        zone_distances: List = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        input_store : PoseInputStore
            the input store object
        keys : list, optional
            a list of names of the features to extract
        ignored_clips : list, optional
            a list of clip ids to ignore
        interactive : bool, default False
            if `True`, features for pairs of clips will be computed
        averaging_window : int, default 1
            if >1, features are averaged with a moving window of this size (in frames)
        distance_pairs : list, optional
            a list of bodypart name tuples (e.g. `[("tail", "nose")]`) to compute distances for when `"intra_distance"`
            is in `keys` (by default all distances are computed)
        angle_pairs : list, optional
            a list of bodypart name tuples (e.g. `[("ear1", "nose", "ear2")]`) for the angle between `"ear1"--"nose"` and
            `"nose"--"ear2"` lines) to compute angle cosines for when `"angles"` is in `keys` (by default no angles are computed)
        neighboring_frames : int, default 0
            if >0, this number of neighboring frames is aggregated in the center frame features (generally not recommended)
        area_vertices : list, optional
            a list of bodypart name tuples of any length >= 3 (e.g. `[("ear1", "nose", "ear2", "spine1")]`) that define polygons
            to compute areas for when `"areas"` is in `keys` (by default no areas are computed)
        zone_vertices : dict, optional
            a dictionary of bodypart name tuples of any length >= 3 that define zones for `"zone_bools"`and `"zone_distances"`
            featyres; keys should be zone names and values should be tuples that define the polygons (e.g.
            `{"main_area": ("x_min", "x_max", "y_max", "y_min"))}`)
        zone_bools : list, optional
            a list of zone and bodypart name tuples to compute binary identifiers for (1 if an animal is within the polygon or
            0 if it's outside) (e.g. `[("main_area", "nose")]`); the zones should be defined in the `zone_vertices` parameter;
            this is only computed if `"zone_bools"` is in `keys`
        zone_distances : list, optional
            a list of zone and bodypart name tuples to compute distances for (distance from the bodypart to the closest of the
            boundaries) (e.g. `[("main_area", "nose")]`); the zones should be defined in the `zone_vertices` parameter;
            this is only computed if `"zone_distances"` is in `keys`
        """

        if keys is None:
            keys = [
                "coord_diff",
                "center",
                "intra_distance",
                "speed_direction",
                "speed_value",
                "angle_speeds",
                "acc_joints",
                "inter_distance",
            ]
        if ignored_clips is None:
            ignored_clips = []
        if zone_vertices is None:
            zone_vertices = {}
        if zone_bools is None:
            zone_bools = []
        if zone_distances is None:
            zone_distances = []
        self.keys = keys
        self.ignored_clips = ignored_clips
        self.interactive = interactive
        self.w = averaging_window
        self.distance_pairs = distance_pairs
        self.angle_pairs = angle_pairs
        self.area_vertices = area_vertices
        self.neighboring_frames = int(neighboring_frames)
        self.zone_vertices = zone_vertices
        self.zone_bools = zone_bools
        self.zone_distances = zone_distances
        super().__init__(input_store)

    def _angle_speed(self, xy_coord_joint: np.array, n_frames: int) -> np.array:
        """
        Compute the angle speed
        """

        if xy_coord_joint.shape[1] == 2:
            x_diff = np.diff(xy_coord_joint[:, 0])
            y_diff = np.diff(xy_coord_joint[:, 1])
            x_diff[xy_coord_joint[:-1, 0] == 0] = 0
            y_diff[xy_coord_joint[:-1, 1] == 0] = 0
            angle_dir_radians = [
                math.atan2(y_diff[i], x_diff[i]) for i in range(n_frames - 1)
            ]
            angle_dir_radians = np.insert(
                angle_dir_radians, 0, angle_dir_radians[0], axis=0
            )
        else:
            x_diff = np.diff(xy_coord_joint[:, 0])
            y_diff = np.diff(xy_coord_joint[:, 1])
            z_diff = np.diff(xy_coord_joint[:, 2])
            x_diff[xy_coord_joint[:-1, 0] == 0] = 0
            y_diff[xy_coord_joint[:-1, 1] == 0] = 0
            y_diff[xy_coord_joint[:-1, 2] == 0] = 0
            angle_dir_radians = []
            for x, y in combinations([x_diff, y_diff, z_diff], 2):
                radians = [math.atan2(x[i], y[i]) for i in range(n_frames - 1)]
                radians = np.insert(radians, 0, radians[0], axis=0)
                angle_dir_radians.append(radians)
            angle_dir_radians = np.concatenate(angle_dir_radians)

        return angle_dir_radians

    def _poly_area(self, x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _cdist_keep_zeros(self, a: np.array, b: np.array) -> np.array:
        """
        Compute all distance combinations while setting the distance to zero if at least one of the elements is at zero
        """

        dist = cdist(a, b, "euclidean")
        a_zero = np.sum(a == 0, axis=1) > 0
        b_zero = np.sum(b == 0, axis=1) > 0
        dist[a_zero, :] = 0
        dist[:, a_zero] = 0
        dist[b_zero, :] = 0
        dist[:, b_zero] = 0
        return dist

    def _distance(
        self, data_dict: Dict, clip1: str, clip2: str, name: str, centroid: bool = False
    ) -> Tuple:
        """
        Compute the distances between all keypoints
        """

        if not isinstance(clip1, list):
            body_parts_1 = self.get_bodyparts()
        else:
            body_parts_1 = clip1
        n_body_parts = len(body_parts_1)
        body_parts_2 = self.get_bodyparts()
        n_frames = self.get_n_frames(data_dict, clip1)
        if n_frames != self.get_n_frames(data_dict, clip2):
            raise RuntimeError(
                f"The numbers of frames for {clip1} and {clip2} are not equal at {name}!"
            )

        # joint distances for single agent
        upper_indices = np.triu_indices(n_body_parts, 1)

        xy_coord_joints_1 = np.stack(
            [self.get_coords(data_dict, clip1, bp) for bp in body_parts_1], axis=1
        )
        if self.w > 1:
            for i in range(xy_coord_joints_1.shape[0]):
                for j in range(xy_coord_joints_1.shape[1]):
                    xy_coord_joints_1[i, j, :] = np.convolve(
                        xy_coord_joints_1[i, j, :], (1 / self.w) * np.ones(self.w)
                    )[self.w // 2 : -self.w // 2 + (self.w + 1) % 2]
        if clip1 != clip2:
            xy_coord_joints_2 = np.stack(
                [self.get_coords(data_dict, clip2, bp) for bp in body_parts_2], axis=1
            )
            if self.w > 1:
                for i in range(xy_coord_joints_2.shape[0]):
                    for j in range(xy_coord_joints_2.shape[1]):
                        xy_coord_joints_2[i, j, :] = np.convolve(
                            xy_coord_joints_2[i, j, :], (1 / self.w) * np.ones(self.w)
                        )[self.w // 2 : -self.w // 2 + (self.w + 1) % 2]
        else:
            xy_coord_joints_2 = copy.copy(xy_coord_joints_1)

        if clip1 != clip2 and centroid:
            centroid_1 = np.expand_dims(np.mean(xy_coord_joints_1, axis=1), 1)
            distance_1 = np.linalg.norm(xy_coord_joints_2 - centroid_1, axis=-1)
            centroid_2 = np.expand_dims(np.mean(xy_coord_joints_2, axis=1), 1)
            distance_2 = np.linalg.norm(xy_coord_joints_1 - centroid_2, axis=-1)
            intra_distance = np.concatenate([distance_1, distance_2], axis=-1)
        else:
            if self.distance_pairs is None:
                n_distances = n_body_parts * (n_body_parts - 1) // 2
                intra_distance = np.asarray(
                    [
                        self._cdist_keep_zeros(
                            xy_coord_joints_1[i], xy_coord_joints_2[i]
                        )[upper_indices].reshape(-1, n_distances)
                        for i in range(n_frames)
                    ]
                ).reshape(n_frames, n_distances)
            else:
                intra_distance = []
                for x, y in self.distance_pairs:
                    x_ind = body_parts_1.index(x)
                    y_ind = body_parts_1.index(y)
                    intra_distance.append(
                        np.sqrt(
                            np.sum(
                                (
                                    xy_coord_joints_1[:, x_ind, :]
                                    - xy_coord_joints_1[:, y_ind, :]
                                )
                                ** 2,
                                axis=1,
                            )
                        )
                    )
                intra_distance = np.stack(intra_distance, axis=1)

        if clip1 == clip2:
            angle_joints_radian = np.stack(
                [
                    self._angle_speed(xy_coord_joints_1[:, i, :], n_frames)
                    for i in range(xy_coord_joints_1.shape[1])
                ],
                axis=1,
            )
            if self.angle_pairs is None:
                angles = None
            else:
                angles = []
                for x0, x1, y0, y1 in self.angle_pairs:
                    x0_ind = body_parts_1.index(x0)
                    x1_ind = body_parts_1.index(x1)
                    y0_ind = body_parts_1.index(y0)
                    y1_ind = body_parts_1.index(y1)
                    diff_x = (
                        xy_coord_joints_1[:, x0_ind, :]
                        - xy_coord_joints_1[:, x1_ind, :]
                    )
                    diff_y = (
                        xy_coord_joints_1[:, y0_ind, :]
                        - xy_coord_joints_1[:, y1_ind, :]
                    )
                    dist_x = np.linalg.norm(diff_x, axis=-1)
                    dist_y = np.linalg.norm(diff_y, axis=-1)
                    denom = dist_x * dist_y + 1e-7
                    mult = np.einsum("ij,ij->i", diff_x, diff_y)
                    angles.append(mult / denom)
                angles = np.stack(angles, axis=1)
            if self.area_vertices is not None:
                areas = []
                for points in self.area_vertices:
                    point_areas = []
                    inds = [body_parts_1.index(x) for x in points]
                    for f_i in range(xy_coord_joints_1.shape[0]):
                        x = xy_coord_joints_1[f_i, inds, 0]
                        y = xy_coord_joints_1[f_i, inds, 1]
                        point_areas.append(self._poly_area(x, y))
                    areas.append(np.array(point_areas))
                areas = np.stack(areas, axis=-1)
            else:
                areas = None

            zone_bools = []
            for zone, vertex in self.zone_bools:
                if zone not in self.zone_vertices:
                    raise ValueError(f"The {zone} zone is not in zone_vertices!")
                if vertex not in body_parts_1:
                    raise ValueError(f"The {vertex} bodypart not in bodyparts!")
                zone_bool = np.ones((xy_coord_joints_1.shape[0], 1))
                vertex_coords = self.get_coords(data_dict, clip1, vertex)
                for i, x in enumerate(self.zone_vertices[zone]):
                    v1 = self.get_coords(data_dict, clip1, x)
                    next_i = (i + 1) % len(self.zone_vertices[zone])
                    next_next_i = (i + 2) % len(self.zone_vertices[zone])
                    v2 = self.get_coords(
                        data_dict, clip1, self.zone_vertices[zone][next_i]
                    )
                    v3 = self.get_coords(
                        data_dict, clip1, self.zone_vertices[zone][next_next_i]
                    )
                    v3_above = (
                        v1[:, 1]
                        + ((v3[:, 0] - v1[:, 0]) / (v2[:, 0] - v1[:, 0] + 1e-7))
                        * (v2[:, 1] - v1[:, 1])
                        > v3[:, 1]
                    )
                    vertex_above = (
                        v1[:, 1]
                        + (
                            (vertex_coords[:, 0] - v1[:, 0])
                            / (v2[:, 0] - v1[:, 0] + 1e-7)
                        )
                        * (v2[:, 1] - v1[:, 1])
                        > vertex_coords[:, 1]
                    )
                    edge_bool = v3_above == vertex_above
                    edge_bool[v2[:, 0] == v1[:, 0]] = (
                        (vertex_coords[:, 0] > v2[:, 0]) == (v3[:, 0] > v2[:, 0])
                    )[v2[:, 0] == v1[:, 0]]
                    zone_bool *= np.expand_dims(edge_bool, 1)
                zone_bools.append(zone_bool)
            if len(zone_bools) == 0:
                zone_bools = None
            else:
                zone_bools = np.concatenate(zone_bools, axis=1)

            distances = []
            for zone, vertex in self.zone_distances:
                if zone not in self.zone_vertices:
                    raise ValueError(f"The {zone} zone is not in zone_vertices!")
                if vertex not in body_parts_1:
                    raise ValueError(f"The {vertex} bodypart not in bodyparts!")
                v0 = self.get_coords(data_dict, clip1, vertex)
                dd = []
                for i, x in enumerate(self.zone_vertices[zone]):
                    v1 = self.get_coords(data_dict, clip1, x)
                    next_i = (i + 1) % len(self.zone_vertices[zone])
                    v2 = self.get_coords(
                        data_dict, clip1, self.zone_vertices[zone][next_i]
                    )
                    d = np.abs(
                        (v2[:, 0] - v2[:, 0]) * (v1[:, 1] - v0[:, 1])
                        - (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v1[:, 1])
                    ) / np.sqrt(
                        (v2[:, 0] - v1[:, 0]) ** 2 + (v2[:, 1] - v1[:, 1]) ** 2 + 1e-7
                    )
                    d[(v2[:, 0] == v1[:, 0]) * (v2[:, 1] == v1[:, 1])] = 0
                    dd.append(d)
                dd = np.stack(dd, axis=0)
                dd = np.min(dd, 0)
                distances.append(dd)
            if len(distances) == 0:
                distances = None
            else:
                distances = np.stack(distances, axis=1)

        if clip1 != clip2:
            return intra_distance, xy_coord_joints_1, xy_coord_joints_2, n_frames
        else:
            return (
                intra_distance,
                xy_coord_joints_1,
                n_frames,
                angle_joints_radian,
                areas,
                angles,
                zone_bools,
                distances,
            )

    def _kinematic_features_pair(
        self, data_dict: Dict, clip1: str, clip2: str, name: str
    ) -> Dict:
        """
        Compute features for a pair of clips
        """

        if clip1 == clip2:
            (
                intra_distance,
                xy_coord_joints,
                n_frames,
                angle_joints_radian,
                areas,
                angles,
                zone_bools,
                zone_distances,
            ) = self._distance(data_dict, clip1, clip2, name)
        else:
            (
                intra_distance,
                xy_coord_joints_1,
                xy_coord_joints_2,
                n_frames,
            ) = self._distance(data_dict, clip1, clip2, name)
            xy_coord_joints = xy_coord_joints_2 - xy_coord_joints_1

        xy_coord_joints = xy_coord_joints.transpose((1, 2, 0))

        speed_joints = np.diff(xy_coord_joints, axis=-1)
        speed_joints[xy_coord_joints[..., :-1] == 0] = 0
        speed_joints = np.insert(speed_joints, 0, speed_joints[:, :, 0], axis=-1)

        # acceleration
        acc_joints = np.asarray([np.diff(speed_joint) for speed_joint in speed_joints])
        acc_joints = np.insert(acc_joints, 0, acc_joints[:, :, 0], axis=-1)
        acc_joints = np.linalg.norm(acc_joints, axis=1)

        # from matplotlib import pyplot as plt
        # print(f'{xy_coord_joints.shape=}')
        # plt.scatter(xy_coord_joints[:, 0, 0],
        #             xy_coord_joints[:, 1, 0])
        # plt.xlim(-0.5, 0.5)
        # plt.ylim(-0.5, 0.5)
        # plt.show()

        features = {}
        if "coords" in self.keys:
            features["coords"] = copy.copy(xy_coord_joints).reshape((-1, n_frames)).T
        if "center" in self.keys:
            features["center"] = xy_coord_joints.mean(0).T
        if "coord_diff" in self.keys:
            features["coord_diff"] = (
                (xy_coord_joints - np.expand_dims(xy_coord_joints.mean(0), 0))
                .reshape((-1, n_frames))
                .T
            )
        if "intra_distance" in self.keys:
            features["intra_distance"] = intra_distance
        if "speed_joints" in self.keys:
            features["speed_joints"] = speed_joints.reshape((-1, n_frames)).T
        if "speed_direction" in self.keys or "speed_value" in self.keys:
            values = np.expand_dims(np.linalg.norm(speed_joints, axis=1), 1) + 1e-7
            directions = speed_joints / values
            if "speed_direction" in self.keys:
                features["speed_direction"] = directions.reshape((-1, n_frames)).T
            if "speed_value" in self.keys:
                features["speed_value"] = values.reshape((-1, n_frames)).T
        if (
            "angle_speeds" in self.keys or "angle_joints_radian" in self.keys
        ) and clip1 == clip2:
            features["angle_speeds"] = angle_joints_radian
        if "angles" in self.keys and clip1 == clip2 and self.angle_pairs is not None:
            features["angles"] = angles
        if "acc_joints" in self.keys:
            features["acc_joints"] = acc_joints.T
        if "areas" in self.keys and clip1 == clip2 and areas is not None:
            features["areas"] = areas * 10
        if "zone_bools" in self.keys and clip1 == clip2 and zone_bools is not None:
            features["zone_bools"] = zone_bools
        if (
            "zone_distances" in self.keys
            and clip1 == clip2
            and zone_distances is not None
        ):
            features["zone_distances"] = zone_distances
        if clip1 == clip2 and "likelihood" in self.keys:
            likelihood = [
                self.get_likelihood(data_dict, clip1, bp) for bp in self.get_bodyparts()
            ]
            if likelihood[0] is not None:
                likelihood = np.stack(likelihood, 1)
                features["likelihood"] = likelihood
        return features

    def extract_features(
        self,
        data_dict: Dict,
        video_id: str,
        prefix: str = None,
        one_clip: bool = False,
    ) -> Dict:
        """
        Extract features from a data dictionary

        An input store will call this method while pre-computing a dataset. We do not assume a specific
        structure in the data dictionary, so all necessary information (coordinates of a bodypart, number
        of frames, list of bodyparts) is inferred using input store methods.

        Parameters
        ----------
        data_dict : dict
            the data dictionary
        video_id : str
            the id of the video associated with the data dictionary
        prefix : str, optional
            a prefix for the feature names
        one_clip : bool, default False
            if `True`, all features will be concatenated and assigned to one clip named `'all'`

        Returns
        -------
        features : dict
            a features dictionary where the keys are the feature names (e.g. 'coords', 'distances') and the
            values are numpy arrays of shape `(#features, #frames)`
        """

        features = {}
        keys = [x for x in data_dict.keys() if x not in self.ignored_clips]
        if self.interactive:
            if one_clip:
                agents = [keys]
            else:
                agents = combinations(keys, 2)
        else:
            agents = [[x] for x in keys]
        for clip_ids in agents:
            clip_features = {}
            for clip in clip_ids:
                single_features = self._kinematic_features_pair(
                    data_dict, clip, clip, video_id
                )
                for key, value in single_features.items():
                    name = key
                    if prefix is not None or len(clip_ids) > 1:
                        name += "---"
                        if prefix is not None:
                            name += prefix
                        if len(clip_ids) > 1:
                            name += clip
                    clip_features[name] = single_features[key]
            if len(clip_ids) > 1 and "inter_distance" in self.keys:
                for clip1, clip2 in combinations(clip_ids, 2):
                    distance, *_ = self._distance(
                        data_dict, clip1, clip2, video_id, centroid=True
                    )
                    name = "inter_distance---"
                    if prefix is not None:
                        name += prefix
                    name += f"{clip1}+{clip2}"
                    clip_features[name] = distance
            if one_clip:
                combo_name = "all"
            else:
                combo_name = "+".join(map(str, clip_ids))
            features[video_id + "---" + combo_name] = clip_features
        if self.neighboring_frames != 0:
            for key in features.keys():
                for clip_key in features[key].keys():
                    new_feature = []
                    for i in range(
                        self.neighboring_frames + 1,
                        features[key][clip_key].shape[0] - self.neighboring_frames,
                    ):
                        new_feature.append(
                            features[key][clip_key][
                                i
                                - self.neighboring_frames : i
                                + self.neighboring_frames,
                                :,
                            ].flatten()
                        )
                    features[key][clip_key] = np.stack(new_feature, axis=0)
        return features
