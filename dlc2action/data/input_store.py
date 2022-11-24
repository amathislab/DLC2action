#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
"""
Specific realisations of `dlc2action.data.base_store.InputStore` are defined here
"""

from dlc2action.data.base_store import PoseInputStore
from typing import Dict, List, Tuple, Union, Set, Optional, Iterable
from dlc2action.utils import TensorDict, strip_suffix, strip_prefix
import numpy as np
from collections import defaultdict
import torch
import os
import pickle
from abc import abstractmethod
import pandas as pd
from p_tqdm import p_map
from dlc2action import options
from PIL import Image
from tqdm import tqdm
import mimetypes



class GeneralInputStore(PoseInputStore):
    """
    A generalized realization of a PoseInputStore

    Assumes the following file structure:
    ```
    data_path
    ├── video1DLC1000.pickle
    ├── video2DLC400.pickle
    ├── video1_features.pt
    └── video2_features.pt
    ```
    Here `data_suffix` is `{'DLC1000.pickle', 'DLC400.pickle'}` and `feature_suffix` (optional) is `'_features.pt'`.
    """

    data_suffix = None

    def __init__(
        self,
        video_order: List = None,
        data_path: Union[Set, str] = None,
        file_paths: Set = None,
        data_suffix: Union[Set, str] = None,
        data_prefix: Union[Set, str] = None,
        feature_suffix: str = None,
        convert_int_indices: bool = True,
        feature_save_path: str = None,
        canvas_shape: List = None,
        len_segment: int = 128,
        overlap: int = 0,
        feature_extraction: str = "kinematic",
        ignored_clips: List = None,
        ignored_bodyparts: List = None,
        default_agent_name: str = "ind0",
        key_objects: Tuple = None,
        likelihood_threshold: float = 0,
        num_cpus: int = None,
        frame_limit: int = 1,
        normalize: bool = False,
        feature_extraction_pars: Dict = None,
        centered: bool = False,
        transpose_features: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        video_order : list, optional
            a list of video ids that should be processed in the same order (not passed if creating from key objects
        data_path : str | set, optional
            the path to the folder where the pose and feature files are stored or a set of such paths
            (not passed if creating from key objects or from `file_paths`)
        file_paths : set, optional
            a set of string paths to the pose and feature files
            (not passed if creating from key objects or from `data_path`)
        data_suffix : str | set, optional
            the suffix or the set of suffices such that the pose files are named {video_id}{data_suffix}
            (not passed if creating from key objects or if irrelevant for the dataset)
        data_prefix : str | set, optional
            the prefix or the set of prefixes such that the pose files for different video views of the same
            clip are named {prefix}{sep}{video_id}{data_suffix} (not passed if creating from key objects
            or if irrelevant for the dataset)
        feature_suffix : str | set, optional
            the suffix or the set of suffices such that the additional feature files are named
            {video_id}{feature_suffix} (and placed at the data_path folder)
        convert_int_indices : bool, default True
            if `True`, convert any integer key `i` in feature files to `'ind{i}'`
        feature_save_path : str, optional
            the path to the folder where pre-processed files are stored (not passed if creating from key objects)
        canvas_shape : List, default [1, 1]
            the canvas size where the pose is defined
        len_segment : int, default 128
            the length of the segments in which the data should be cut (in frames)
        overlap : int, default 0
            the length of the overlap between neighboring segments (in frames)
        feature_extraction : str, default 'kinematic'
            the feature extraction method (see options.feature_extractors for available options)
        ignored_clips : list, optional
            list of strings of clip ids to ignore
        ignored_bodyparts : list, optional
            list of strings of bodypart names to ignore
        default_agent_name : str, default 'ind0'
            the agent name used as default in the pose files for a single agent
        key_objects : tuple, optional
            a tuple of key objects
        likelihood_threshold : float, default 0
            coordinate values with likelihoods less than this value will be set to 'unknown'
        num_cpus : int, optional
            the number of cpus to use in data processing
        frame_limit : int, default 1
            clips shorter than this number of frames will be ignored
        feature_extraction_pars : dict, optional
            parameters of the feature extractor
        """

        super().__init__()
        self.loaded_max = 0
        if feature_extraction_pars is None:
            feature_extraction_pars = {}
        if ignored_clips is None:
            ignored_clips = []
        self.bodyparts = []
        self.visibility = None
        self.normalize = normalize

        if canvas_shape is None:
            canvas_shape = [1, 1]
        if isinstance(data_suffix, str):
            data_suffix = [data_suffix]
        if isinstance(data_prefix, str):
            data_prefix = [data_prefix]
        if isinstance(data_path, str):
            data_path = [data_path]
        if isinstance(feature_suffix, str):
            feature_suffix = [feature_suffix]

        self.video_order = video_order
        self.centered = centered
        self.feature_extraction = feature_extraction
        self.len_segment = int(len_segment)
        self.data_suffices = data_suffix
        self.data_prefixes = data_prefix
        self.feature_suffix = feature_suffix
        self.convert_int_indices = convert_int_indices
        if overlap < 1:
            overlap = overlap * len_segment
        self.overlap = int(overlap)
        self.canvas_shape = canvas_shape
        self.default_agent_name = default_agent_name
        self.feature_save_path = feature_save_path
        self.data_suffices = data_suffix
        self.data_prefixes = data_prefix
        self.likelihood_threshold = likelihood_threshold
        self.num_cpus = num_cpus
        self.frame_limit = frame_limit
        self.transpose = transpose_features

        self.ram = False
        self.min_frames = {}
        self.original_coordinates = np.array([])

        self.file_paths = self._get_file_paths(file_paths, data_path)

        self.extractor = options.feature_extractors[self.feature_extraction](
            self,
            **feature_extraction_pars,
        )

        self.canvas_center = np.array(canvas_shape) // 2

        if ignored_clips is not None:
            self.ignored_clips = ignored_clips
        else:
            self.ignored_clips = []
        if ignored_bodyparts is not None:
            self.ignored_bodyparts = ignored_bodyparts
        else:
            self.ignored_bodyparts = []

        self.step = self.len_segment - self.overlap
        if self.step < 0:
            raise ValueError(
                f"The overlap value ({self.overlap}) cannot be larger than len_segment ({self.len_segment}"
            )

        if self.feature_save_path is None and data_path is not None:
            self.feature_save_path = os.path.join(data_path[0], "trimmed")

        if key_objects is None and self.video_order is not None:
            print("Computing input features...")
            self.data = self._load_data()
        elif key_objects is not None:
            self.load_from_key_objects(key_objects)

    def __getitem__(self, ind: int) -> Dict:
        prompt = self.data[ind]
        if not self.ram:
            with open(prompt, "rb") as f:
                prompt = pickle.load(f)
        return prompt

    def __len__(self) -> int:
        if self.data is None:
            raise RuntimeError("The input store data has not been initialized!")
        return len(self.data)

    @classmethod
    def _get_file_paths(cls, file_paths: Set, data_path: Union[str, Set]) -> List:
        """
        Get a set of relevant files

        Parameters
        ----------
        file_paths : set
            a set of filepaths to include
        data_path : str | set
            the path to a folder that contains relevant files (a single path or a set)

        Returns
        -------
        file_paths : list
            a list of relevant file paths (input and feature files that follow the dataset naming pattern)
        """

        if file_paths is None:
            file_paths = []
        file_paths = list(file_paths)
        if data_path is not None:
            if isinstance(data_path, str):
                data_path = [data_path]
            for folder in data_path:
                file_paths += [os.path.join(folder, x) for x in os.listdir(folder)]
        return file_paths

    def get_folder(self, video_id: str) -> str:
        """
        Get the input folder that the file with this video id was read from

        Parameters
        ----------
        video_id : str
            the video id

        Returns
        -------
        folder : str
            the path to the directory that contains the input file associated with the video id
        """

        for file in self.file_paths:
            if (
                strip_prefix(
                    strip_suffix(os.path.basename(file), self.data_suffices),
                    self.data_prefixes,
                )
                == video_id
            ):
                return os.path.dirname(file)

    def remove(self, indices: List) -> None:
        """
        Remove the samples corresponding to indices

        Parameters
        ----------
        indices : int
            a list of integer indices to remove
        """

        if len(indices) > 0:
            mask = np.ones(len(self.original_coordinates))
            mask[indices] = 0
            mask = mask.astype(bool)
            for file in self.data[~mask]:
                os.remove(file)
            self.original_coordinates = self.original_coordinates[mask]
            self.data = self.data[mask]
            if self.metadata is not None:
                self.metadata = self.metadata[mask]

    def key_objects(self) -> Tuple:
        """
        Return a tuple of the key objects necessary to re-create the Store

        Returns
        -------
        key_objects : tuple
            a tuple of key objects
        """

        for k, v in self.min_frames.items():
            self.min_frames[k] = dict(v)
        for k, v in self.max_frames.items():
            self.max_frames[k] = dict(v)
        return (
            self.original_coordinates,
            dict(self.min_frames),
            dict(self.max_frames),
            self.data,
            self.visibility,
            self.step,
            self.file_paths,
            self.len_segment,
            self.metadata,
        )

    def load_from_key_objects(self, key_objects: Tuple) -> None:
        """
        Load the information from a tuple of key objects

        Parameters
        ----------
        key_objects : tuple
            a tuple of key objects
        """

        (
            self.original_coordinates,
            self.min_frames,
            self.max_frames,
            self.data,
            self.visibility,
            self.step,
            self.file_paths,
            self.len_segment,
            self.metadata,
        ) = key_objects

    def to_ram(self) -> None:
        """
        Transfer the data samples to RAM if they were previously stored as file paths
        """

        if self.ram:
            return

        if os.name != "nt":
            data = p_map(lambda x: self[x], list(range(len(self))), num_cpus=self.num_cpus)
        else:
            print("Multiprocessing is not supported on Windows, loading files sequentially.")
            data = [load(x) for x in tqdm(self.data)]
        self.data = TensorDict(data)
        self.ram = True

    def get_original_coordinates(self) -> np.ndarray:
        """
        Return the original coordinates array

        Returns
        -------
        np.ndarray
            an array that contains the coordinates of the data samples in original input data (video id, clip id,
            start frame)
        """

        return self.original_coordinates

    def create_subsample(self, indices: List, ssl_indices: List = None):
        """
        Create a new store that contains a subsample of the data

        Parameters
        ----------
        indices : list
            the indices to be included in the subsample
        ssl_indices : list, optional
            the indices to be included in the subsample without the annotation data
        """

        if ssl_indices is None:
            ssl_indices = []
        new = self.new()
        new.original_coordinates = self.original_coordinates[indices + ssl_indices]
        new.min_frames = self.min_frames
        new.max_frames = self.max_frames
        new.data = self.data[indices + ssl_indices]
        new.visibility = self.visibility
        new.step = self.step
        new.file_paths = self.file_paths
        new.len_segment = self.len_segment
        if self.metadata is None:
            new.metadata = None
        else:
            new.metadata = self.metadata[indices + ssl_indices]
        return new

    def get_video_id(self, coords: Tuple) -> str:
        """
        Get the video id from an element of original coordinates

        Parameters
        ----------
        coords : tuple
            an element of the original coordinates array

        Returns
        -------
        video_id: str
            the id of the video that the coordinates point to
        """

        video_name = coords[0].split("---")[0]
        return video_name

    def get_clip_id(self, coords: Tuple) -> str:
        """
        Get the clip id from an element of original coordinates

        Parameters
        ----------
        coords : tuple
            an element of the original coordinates array

        Returns
        -------
        clip_id : str
            the id of the clip that the coordinates point to
        """

        clip_id = coords[0].split("---")[1]
        return clip_id

    def get_clip_length(self, video_id: str, clip_id: str) -> int:
        """
        Get the clip length from the id

        Parameters
        ----------
        video_id : str
            the video id
        clip_id : str
            the clip id

        Returns
        -------
        clip_length : int
            the length of the clip
        """

        inds = clip_id.split("+")
        max_frame = min([self.max_frames[video_id][x] for x in inds])
        min_frame = max([self.min_frames[video_id][x] for x in inds])
        return max_frame - min_frame + 1

    def get_clip_start_end(self, coords: Tuple) -> Tuple[int, int]:
        """
        Get the clip start and end frames from an element of original coordinates

        Parameters
        ----------
        coords : tuple
            an element of original coordinates array

        Returns
        -------
        start : int
            the start frame of the clip that the coordinates point to
        end : int
            the end frame of the clip that the coordinates point to
        """

        l = self.get_clip_length_from_coords(coords)
        i = coords[1]
        start = int(i) * self.step
        end = min(start + self.len_segment, l)
        return start, end

    def get_clip_start(self, video_name: str, clip_id: str) -> int:
        """
        Get the clip start frame from the video id and the clip id

        Parameters
        ----------
        video_name : str
            the video id
        clip_id : str
            the clip id

        Returns
        -------
        clip_start : int
            the start frame of the clip
        """

        return max(
            [self.min_frames[video_name][clip_id_k] for clip_id_k in clip_id.split("+")]
        )

    def get_visibility(
        self, video_id: str, clip_id: str, start: int, end: int, score: int
    ) -> float:
        """
        Get the fraction of the frames in that have a visibility score better than a hard_threshold

        For example, in the case of keypoint data the visibility score can be the number of identified keypoints.

        Parameters
        ----------
        video_id : str
            the video id of the frames
        clip_id : str
            the clip id of the frames
        start : int
            the start frame
        end : int
            the end frame
        score : float
            the visibility score hard_threshold

        Returns
        -------
        frac_visible: float
            the fraction of frames with visibility above the hard_threshold
        """

        s = 0
        for ind_k in clip_id.split("+"):
            s += np.sum(self.visibility[video_id][ind_k][start:end] > score) / (
                end - start
            )
        return s / len(clip_id.split("+"))

    def get_annotation_objects(self) -> Dict:
        """
        Get a dictionary of objects necessary to create an AnnotationStore

        Returns
        -------
        annotation_objects : dict
            a dictionary of objects to be passed to the AnnotationStore constructor where the keys are the names of
            the objects
        """

        min_frames = self.min_frames
        max_frames = self.max_frames
        num_bp = self.visibility
        return {
            "min_frames": min_frames,
            "max_frames": max_frames,
            "visibility": num_bp,
        }

    @classmethod
    def get_file_ids(
        cls,
        data_suffix: Union[Set, str] = None,
        data_path: Union[Set, str] = None,
        data_prefix: Union[Set, str] = None,
        file_paths: Set = None,
        feature_suffix: Set = None,
        *args,
        **kwargs,
    ) -> List:
        """
        Process data parameters and return a list of ids  of the videos that should
        be processed by the __init__ function

        Parameters
        ----------
        data_suffix : set | str, optional
            the suffix (or a set of suffixes) of the input data files
        data_path : set | str, optional
            the path to the folder where the pose and feature files are stored or a set of such paths
            (not passed if creating from key objects or from `file_paths`)
        data_prefix : set | str, optional
            the prefix or the set of prefixes such that the pose files for different video views of the same
            clip are named {prefix}{sep}{video_id}{data_suffix} (not passed if creating from key objects
            or if irrelevant for the dataset)
        file_paths : set, optional
            a set of string paths to the pose and feature files
        feature_suffix : str | set, optional
            the suffix or the set of suffices such that the additional feature files are named
            {video_id}{feature_suffix} (and placed at the `data_path` folder or at `file_paths`)

        Returns
        -------
        video_ids : list
            a list of video file ids
        """

        if data_suffix is None:
            if cls.data_suffix is not None:
                data_suffix = cls.data_suffix
            else:
                raise ValueError("Cannot get video ids without the data suffix!")
        if feature_suffix is None:
            feature_suffix = []
        if data_prefix is None:
            data_prefix = ""
        if isinstance(data_suffix, str):
            data_suffix = [data_suffix]
        else:
            data_suffix = [x for x in data_suffix]
        data_suffix = tuple(data_suffix)
        if isinstance(data_prefix, str):
            data_prefix = data_prefix
        else:
            data_prefix = tuple([x for x in data_prefix])
        if isinstance(feature_suffix, str):
            feature_suffix = [feature_suffix]
        if file_paths is None:
            file_paths = []
        if data_path is not None:
            if isinstance(data_path, str):
                data_path = [data_path]
            file_paths = []
            for folder in data_path:
                file_paths += [os.path.join(folder, x) for x in os.listdir(folder)]
        basenames = [os.path.basename(f) for f in file_paths]
        ids = set()
        for f in file_paths:
            if f.endswith(data_suffix) and os.path.basename(f).startswith(data_prefix):
                bn = os.path.basename(f)
                video_id = strip_prefix(strip_suffix(bn, data_suffix), data_prefix)
                if all([video_id + s in basenames for s in feature_suffix]):
                    ids.add(video_id)
        ids = sorted(ids)
        return ids

    def get_bodyparts(self) -> List:
        """
        Get a list of bodypart names

        Parameters
        ----------
        data_dict : dict
            the data dictionary (passed to feature extractor)
        clip_id : str
            the clip id

        Returns
        -------
        bodyparts : list
            a list of string or integer body part names
        """

        return [x for x in self.bodyparts if x not in self.ignored_bodyparts]

    def get_coords(self, data_dict: Dict, clip_id: str, bodypart: str) -> np.ndarray:
        """
        Get the coordinates array of a specific bodypart in a specific clip

        Parameters
        ----------
        data_dict : dict
            the data dictionary (passed to feature extractor)
        clip_id : str
            the clip id
        bodypart : str
            the name of the body part

        Returns
        -------
        coords : np.ndarray
            the coordinates array of shape (#timesteps, #coordinates)
        """

        columns = [x for x in data_dict[clip_id].columns if x != "likelihood"]
        xy_coord = (
            data_dict[clip_id]
            .xs(bodypart, axis=0, level=1, drop_level=False)[columns]
            .values
        )
        return xy_coord

    def get_n_frames(self, data_dict: Dict, clip_id: str) -> int:
        """
        Get the length of the clip

        Parameters
        ----------
        data_dict : dict
            the data dictionary (passed to feature extractor)
        clip_id : str
            the clip id

        Returns
        -------
        n_frames : int
            the length of the clip
        """

        if clip_id in data_dict:
            return len(data_dict[clip_id].groupby(level=0))
        else:
            return min(
                [len(data_dict[ind_k].groupby(level=0)) for ind_k in clip_id.split("+")]
            )

    def _filter(self, data_dict: Dict) -> Tuple[Dict, Dict, Dict]:
        """
        Apply filters to a data dictionary + normalize the values and generate frame index dictionaries

        The filters include filling nan values, applying length and likelihood thresholds and removing
        ignored clip ids.
        """

        new_data_dict = {}
        keys = list(data_dict.keys())
        for key in keys:
            if key == "loaded":
                continue
            coord = data_dict.pop(key)
            if key in self.ignored_clips:
                continue
            num_frames = len(coord.index.unique(level=0))
            if num_frames < self.frame_limit:
                continue
            if "likelihood" in coord.columns:
                columns = list(coord.columns)
                columns.remove("likelihood")
                coord.loc[
                    coord["likelihood"] < self.likelihood_threshold, columns
                ] = np.nan
            if not isinstance(self.centered, Iterable):
                self.centered = [
                    bool(self.centered)
                    for dim in ["x", "y", "z"]
                    if dim in coord.columns
                ]
            for i, dim in enumerate(["x", "y", "z"]):
                if dim in coord.columns:
                    if self.centered[i]:
                        coord[dim] = coord[dim] + self.canvas_shape[i] // 2
                    # coord.loc[coord[dim] < -self.canvas_shape[i] * 3 // 2, dim] = np.nan
                    # coord.loc[coord[dim] > self.canvas_shape[i] * 3 // 2, dim] = np.nan
            coord = coord.sort_index(level=0)
            for bp in coord.index.unique(level=1):
                coord.loc[coord.index.isin([bp], level=1)] = coord[
                    coord.index.isin([bp], level=1)
                ].interpolate()
            dims = [x for x in coord.columns if x != "likelihood"]
            mask = ~coord[dims[0]].isna()
            for dim in dims[1:]:
                mask = mask & (~coord[dim].isna())
            mean = coord.loc[mask].groupby(level=0).mean()
            for frame in set(coord.index.get_level_values(0)):
                if frame not in mean.index:
                    mean.loc[frame] = [np.nan for _ in mean.columns]
            mean = mean.interpolate()
            mean[mean.isna()] = 0
            for dim in coord.columns:
                if dim == "likelihood":
                    continue
                coord.loc[coord[dim].isna(), dim] = mean.loc[
                    coord.loc[coord[dim].isna()].index.get_level_values(0)
                ][dim].to_numpy()
            if np.sum(self.canvas_shape) > 0:
                for i, dim in enumerate(["x", "y", "z"]):
                    if dim in coord.columns:
                        coord[dim] = (
                            coord[dim] - self.canvas_shape[i] // 2
                        ) / self.canvas_shape[0]
            new_data_dict[key] = coord
        max_frames = {}
        min_frames = {}
        for key, value in new_data_dict.items():
            max_frames[key] = max(value.index.unique(0))
            min_frames[key] = min(value.index.unique(0))
        if "loaded" in data_dict:
            new_data_dict["loaded"] = data_dict["loaded"]
        return new_data_dict, min_frames, max_frames

    def _get_files_from_ids(self):
        files = defaultdict(lambda: [])
        used_prefixes = defaultdict(lambda: [])
        for f in self.file_paths:
            if f.endswith(tuple([x for x in self.data_suffices])):
                bn = os.path.basename(f)
                video_id = strip_prefix(
                    strip_suffix(bn, self.data_suffices), self.data_prefixes
                )
                ok = True
                if self.data_prefixes is not None:
                    for p in self.data_prefixes:
                        if bn.startswith(p):
                            if p not in used_prefixes[video_id]:
                                used_prefixes[video_id].append(p)
                            else:
                                ok = False
                            break
                if not ok:
                    continue
                files[video_id].append(f)
        files = [files[x] for x in self.video_order]
        return files

    def _make_trimmed_data(self, keypoint_dict: Dict) -> Tuple[List, Dict, List]:
        """
        Cut a keypoint dictionary into overlapping pieces of equal length
        """

        X = []
        original_coordinates = []
        lengths = defaultdict(lambda: {})
        if not os.path.exists(self.feature_save_path):
            try:
                os.mkdir(self.feature_save_path)
            except FileExistsError:
                pass
        order = sorted(list(keypoint_dict.keys()))
        for v_id in order:
            keypoints = keypoint_dict[v_id]
            v_len = min([len(x) for x in keypoints.values()])
            sp = np.arange(0, v_len, self.step)
            pad = sp[-1] + self.len_segment - v_len
            video_id, clip_id = v_id.split("---")
            for key in keypoints:
                if len(keypoints[key]) > v_len:
                    keypoints[key] = keypoints[key][:v_len]
                if len(keypoints[key].shape) == 2:
                    keypoints[key] = np.pad(keypoints[key], ((0, pad), (0, 0)))
                else:
                    keypoints[key] = np.pad(
                        keypoints[key], ((0, pad), (0, 0), (0, 0), (0, 0))
                    )
            for i, start in enumerate(sp):
                sample_dict = {}
                original_coordinates.append((v_id, i))
                for key in keypoints:
                    sample_dict[key] = keypoints[key][start : start + self.len_segment]
                    sample_dict[key] = torch.tensor(np.array(sample_dict[key])).float()
                    sample_dict[key] = sample_dict[key].permute(
                        (*range(1, len(sample_dict[key].shape)), 0)
                    )
                name = os.path.join(self.feature_save_path, f"{v_id}_{start}.pickle")
                X.append(name)
                lengths[video_id][clip_id] = v_len
                with open(name, "wb") as f:
                    pickle.dump(sample_dict, f)
        return X, dict(lengths), original_coordinates

    def _load_saved_features(self, video_id: str):
        """
        Load saved features file `(#frames, #features)`
        """

        basenames = [os.path.basename(x) for x in self.file_paths]
        loaded_features_cat = []
        self.feature_suffix = sorted(self.feature_suffix)
        for feature_suffix in self.feature_suffix:
            i = basenames.index(os.path.basename(video_id) + feature_suffix)
            path = self.file_paths[i]
            if not os.path.exists(path):
                raise RuntimeError(f"Did not find a feature file for {video_id}!")
            extension = feature_suffix.split(".")[-1]
            if extension in ["pickle", "pkl"]:
                with open(path, "rb") as f:
                    loaded_features = pickle.load(f)
            elif extension in ["pt", "pth"]:
                loaded_features = torch.load(path)
            elif extension == "npy":
                loaded_features = np.load(path, allow_pickle=True).item()
            else:
                raise ValueError(
                    f"Found feature file in an unrecognized format: .{extension}. \n "
                    "Please save with torch (as .pt or .pth), numpy (as .npy) or pickle (as .pickle or .pkl)."
                )
            loaded_features_cat.append(loaded_features)
        keys = list(loaded_features_cat[0].keys())
        loaded_features = {}
        for k in keys:
            if k in ["min_frames", "max_frames", "video_tag"]:
                loaded_features[k] = loaded_features_cat[0][k]
                continue
            features = []
            for x in loaded_features_cat:
                if not isinstance(x[k], torch.Tensor):
                    features.append(torch.from_numpy(x[k]))
                else:
                    features.append(x[k])
            a = torch.cat(features)
            if self.transpose:
                a = a.T
            loaded_features[k] = a
        return loaded_features

    def get_likelihood(
        self, data_dict: Dict, clip_id: str, bodypart: str
    ) -> Union[np.ndarray, None]:
        """
        Get the likelihood values

        Parameters
        ----------
        data_dict : dict
            the data dictionary
        clip_id : str
            the clip id
        bodypart : str
            the name of the body part

        Returns
        -------
        likelihoods: np.ndarrray | None
            `None` if the dataset doesn't have likelihoods or an array of shape (#timestamps)
        """

        if "likelihood" in data_dict[clip_id].columns:
            likelihood = (
                data_dict[clip_id]
                .xs(bodypart, axis=0, level=1, drop_level=False)
                .values[:, -1]
            )
            return likelihood
        else:
            return None

    def _get_video_metadata(self, metadata_list: Optional[List]):
        """
        Make a single metadata dictionary from a list of dictionaries recieved from different data prefixes
        """

        if metadata_list is None:
            return None
        else:
            return metadata_list[0]

    def get_indices(self, tag: int) -> List:
        """
        Get a list of indices of samples that have a specific meta tag

        Parameters
        ----------
        tag : int
            the meta tag for the subsample (`None` for the whole dataset)

        Returns
        -------
        indices : list
            a list of indices that meet the criteria
        """

        if tag is None:
            return list(range(len(self.data)))
        else:
            return list(np.where(self.metadata == tag)[0])

    def get_tags(self) -> List:
        """
        Get a list of all meta tags

        Returns
        -------
        tags: List
            a list of unique meta tag values
        """

        if self.metadata is None:
            return [None]
        else:
            return list(np.unique(self.metadata))

    def get_tag(self, idx: int) -> Union[int, None]:
        """
        Return a tag object corresponding to an index

        Tags can carry meta information (like annotator id) and are accepted by models that require
        that information. When a tag is `None`, it is not passed to the model.

        Parameters
        ----------
        idx : int
            the index

        Returns
        -------
        tag : int
            the tag object
        """

        if self.metadata is None or idx is None:
            return None
        else:
            return self.metadata[idx]

    @abstractmethod
    def _load_data(self) -> None:
        """
        Load input data and generate data prompts
        """


class FileInputStore(GeneralInputStore):
    """
    An implementation of `dlc2action.data.InputStore` for datasets where each input data file corresponds to one video
    """

    def _count_bodyparts(
        self, data: Dict, stripped_name: str, max_frames: Dict
    ) -> Dict:
        """
        Create a visibility score dictionary (with a score from 0 to 1 assigned to each frame of each clip)
        """

        result = {stripped_name: {}}
        prefixes = list(data.keys())
        for ind in data[prefixes[0]]:
            res = 0
            for _, data_dict in data.items():
                num_bp = len(data_dict[ind].index.unique(level=1))
                coords = (
                    data_dict[ind].values.reshape(
                        -1, num_bp, len(data_dict[ind].columns)
                    )[: max_frames[ind], :, 0]
                    != 0
                )
                res = np.sum(coords, axis=1) + res
            result[stripped_name][ind] = (res / len(prefixes)) / coords.shape[1]
        return result

    def _generate_features(self, data: Dict, video_id: str) -> Dict:
        """
        Generate features from the raw coordinates
        """

        features = defaultdict(lambda: {})
        loaded_common = []
        for prefix, data_dict in data.items():
            if prefix == "":
                prefix = None
            if "loaded" in data_dict:
                loaded_common.append(torch.tensor(data_dict.pop("loaded")))

            key_features = self.extractor.extract_features(
                data_dict, video_id, prefix=prefix
            )
            for f_key in key_features:
                features[f_key].update(key_features[f_key])
        if len(loaded_common) > 0:
            loaded_common = torch.cat(loaded_common, dim=1)
        else:
            loaded_common = None
        if self.feature_suffix is not None:
            loaded_features = self._load_saved_features(video_id)
            for clip_id, feature_tensor in loaded_features.items():
                if not isinstance(feature_tensor, torch.Tensor):
                    feature_tensor = torch.tensor(feature_tensor)
                if self.convert_int_indices and (
                    isinstance(clip_id, int) or isinstance(clip_id, np.integer)
                ):
                    clip_id = f"ind{clip_id}"
                key1 = f"{os.path.basename(video_id)}---{clip_id}"
                if key1 in features:
                    try:
                        key2 = list(features[key1].keys())[0]
                        n_frames = features[key1][key2].shape[0]
                        if feature_tensor.shape[0] != n_frames:
                            n = feature_tensor.shape[0] - n_frames
                            if (
                                abs(n) > 2
                                and abs(feature_tensor.shape[1] - n_frames) <= 2
                            ):
                                feature_tensor = feature_tensor.T
                            # If off by <=2 frames, just clip the end
                            elif n > 0 and n <= 2:
                                feature_tensor = feature_tensor[:n_frames, :]
                            elif n < 0 and n >= -2:
                                filler = feature_tensor[-2:-1, :]
                                for i in range(n_frames - feature_tensor.shape[0]):
                                    feature_tensor = torch.cat(
                                        [feature_tensor, filler], 0
                                    )
                            else:
                                raise RuntimeError(
                                    f"Number of frames in precomputed features with shape"
                                    f" {feature_tensor.shape} is inconsistent with generated features!"
                                )
                        if loaded_common is not None:
                            if feature_tensor.shape[0] == loaded_common.shape[0]:
                                feature_tensor = torch.cat(
                                    [feature_tensor, loaded_common], dim=1
                                )
                            elif feature_tensor.shape[0] == loaded_common.shape[1]:
                                feature_tensor = torch.cat(
                                    [feature_tensor.T, loaded_common], dim=1
                                )
                            else:
                                raise ValueError(
                                    "The features from the data file and from the feature file have a different number of frames!"
                                )
                        features[key1]["loaded"] = feature_tensor
                    except ValueError:
                        raise RuntimeError(
                            "Individuals in precomputed features are inconsistent "
                            "with generated features"
                        )
        elif loaded_common is not None:
            for key in features:
                features[key]["loaded"] = loaded_common
        return features

    def _load_data(self) -> np.array:
        """
        Load input data and generate data prompts
        """

        if self.video_order is None:
            return None

        files = defaultdict(lambda: [])
        for f in self.file_paths:
            if f.endswith(tuple([x for x in self.data_suffices])):
                bn = os.path.basename(f)
                video_id = strip_prefix(
                    strip_suffix(bn, self.data_suffices), self.data_prefixes
                )
                files[video_id].append(f)
        files = [files[x] for x in self.video_order]

        def make_data_dictionary(filenames):
            data = {}
            stored_maxes = defaultdict(lambda: [])
            min_frames, max_frames = {}, {}
            name = strip_suffix(filenames[0], self.data_suffices)
            name = os.path.basename(name)
            stripped_name = strip_prefix(name, self.data_prefixes)
            metadata_list = []
            for filename in filenames:
                name = strip_suffix(filename, self.data_suffices)
                name = os.path.basename(name)
                prefix = strip_suffix(name, [stripped_name])
                data_new, tag = self._open_data(filename, self.default_agent_name)
                data_new, min_frames, max_frames = self._filter(data_new)
                data[prefix] = data_new
                for key, val in max_frames.items():
                    stored_maxes[key].append(val)
                metadata_list.append(tag)
            video_tag = self._get_video_metadata(metadata_list)
            sample_df = list(list(data.values())[0].values())[0]
            self.bodyparts = sorted(list(sample_df.index.unique(1)))
            smallest_maxes = dict.fromkeys(stored_maxes)
            for key, val in stored_maxes.items():
                smallest_maxes[key] = np.amin(val)
            data_dict = self._generate_features(data, stripped_name)
            bp_dict = self._count_bodyparts(
                data=data, stripped_name=stripped_name, max_frames=smallest_maxes
            )
            min_frames = {stripped_name: min_frames}  # name is e.g. 20190707T1126-1226
            max_frames = {stripped_name: max_frames}
            names, lengths, coords = self._make_trimmed_data(data_dict)
            return names, lengths, coords, bp_dict, min_frames, max_frames, video_tag

        if os.name != "nt":
            dict_list = p_map(make_data_dictionary, files, num_cpus=self.num_cpus)
        else:
            print("Multiprocessing is not supported on Windows, loading files sequentially.")
            dict_list = tqdm([make_data_dictionary(f) for f in files])

        self.visibility = {}
        self.min_frames = {}
        self.max_frames = {}
        self.original_coordinates = []
        self.metadata = []
        X = []
        for (
            names,
            lengths,
            coords,
            bp_dictionary,
            min_frames,
            max_frames,
            metadata,
        ) in dict_list:
            X += names
            self.original_coordinates += coords
            self.visibility.update(bp_dictionary)
            self.min_frames.update(min_frames)
            self.max_frames.update(max_frames)
            if metadata is not None:
                self.metadata += metadata
        del dict_list
        if len(self.metadata) != len(self.original_coordinates):
            self.metadata = None
        else:
            self.metadata = np.array(self.metadata)

        self.min_frames = dict(self.min_frames)
        self.max_frames = dict(self.max_frames)
        self.original_coordinates = np.array(self.original_coordinates)
        return np.array(X)

    @abstractmethod
    def _open_data(
        self, filename: str, default_clip_name: str
    ) -> Tuple[Dict, Optional[Dict]]:
        """
        Load the keypoints from filename and organize them in a dictionary

        In `data_dictionary`, the keys are clip ids and the values are `pandas` dataframes with two-level indices.
        The first level is the frame numbers and the second is the body part names. The dataframes should have from
        two to four columns labeled `"x"`, `"y"` and (optionally) `"z"` and `"likelihood"`. Each frame should have
        information on all the body parts. You don't have to filter the data in any way or fill the nans, it will
        be done automatically.

        Parameters
        ----------
        filename : str
            path to the pose file
        default_clip_name : str
            the name to assign to a clip if it does not have a name in the raw data

        Returns
        -------
        data dictionary : dict
            a dictionary where the keys are clip ids and the values are keypoint dataframes (see above for details)
        metadata_dictionary : dict
            a dictionary where the keys are clip ids and the values are metadata objects (can be any additional information,
            like the annotator tag; for no metadata pass `None`)
        """


class SequenceInputStore(GeneralInputStore):
    """
    An implementation of `dlc2action.data.InputStore` for datasets where input data files correspond to multiple videos
    """

    def _count_bodyparts(
        self, data: Dict, stripped_name: str, max_frames: Dict
    ) -> Dict:
        """
        Create a visibility score dictionary (with a score from 0 to 1 assigned to each frame of each clip)
        """

        result = {stripped_name: {}}
        for ind in data.keys():
            num_bp = len(data[ind].index.unique(level=1))
            coords = (
                data[ind].values.reshape(-1, num_bp, len(data[ind].columns))[
                    : max_frames[ind], :, 0
                ]
                != 0
            )
            res = np.sum(coords, axis=1)
            result[stripped_name][ind] = res / coords.shape[1]
        return result

    def _generate_features(self, data: Dict, name: str) -> Dict:
        """
        Generate features for an individual
        """

        features = self.extractor.extract_features(data, name, prefix=None)
        if self.feature_suffix is not None:
            loaded_features = self._load_saved_features(name)
            for clip_id, feature_tensor in loaded_features.items():
                if not isinstance(feature_tensor, torch.Tensor):
                    feature_tensor = torch.tensor(feature_tensor)
                if self.convert_int_indices and (
                    isinstance(clip_id, int) or isinstance(clip_id, np.integer)
                ):
                    clip_id = f"ind{clip_id}"
                key1 = f"{os.path.basename(name)}---{clip_id}"
                if key1 in features:
                    try:
                        key2 = list(features[key1].keys())[0]
                        n_frames = features[key1][key2].shape[0]
                        if feature_tensor.shape[0] != n_frames:
                            n = feature_tensor.shape[0] - n_frames
                            if (
                                abs(n) > 2
                                and abs(feature_tensor.shape[1] - n_frames) <= 2
                            ):
                                feature_tensor = feature_tensor.T
                            # If off by <=2 frames, just clip the end
                            elif n > 0 and n <= 2:
                                feature_tensor = feature_tensor[:n_frames, :]
                            elif n < 0 and n >= -2:
                                filler = feature_tensor[-2:-1, :]
                                for i in range(n_frames - feature_tensor.shape[0]):
                                    feature_tensor = torch.cat(
                                        [feature_tensor, filler], 0
                                    )
                            else:
                                raise RuntimeError(
                                    print(
                                        f"Number of frames in precomputed features with shape"
                                        f" {feature_tensor.shape} is inconsistent with generated features!"
                                    )
                                )
                        features[key1]["loaded"] = feature_tensor
                    except ValueError:
                        raise RuntimeError(
                            print(
                                "Individuals in precomputed features are inconsistent "
                                "with generated features"
                            )
                        )
        return features

    def _load_data(self) -> np.array:
        """
        Load input data and generate data prompts
        """

        if self.video_order is None:
            return None

        files = []
        for f in self.file_paths:
            if os.path.basename(f) in self.video_order:
                files.append(f)

        def make_data_dictionary(seq_tuple):
            seq_id, sequence = seq_tuple
            data, tag = self._get_data(seq_id, sequence, self.default_agent_name)
            if "loaded" in data.keys():
                loaded_features = data.pop("loaded")
            data, min_frames, max_frames = self._filter(data)
            sample_df = list(data.values())[0]
            self.bodyparts = sorted(list(sample_df.index.unique(1)))
            data_dict = self._generate_features(data, seq_id)
            for key in data_dict.keys():
                data_dict[key]["loaded"] = loaded_features
            bp_dict = self._count_bodyparts(
                data=data, stripped_name=seq_id, max_frames=max_frames
            )
            min_frames = {seq_id: min_frames}  # name is e.g. 20190707T1126-1226
            max_frames = {seq_id: max_frames}
            names, lengths, coords = self._make_trimmed_data(data_dict)
            return names, lengths, coords, bp_dict, min_frames, max_frames, tag

        seq_tuples = []
        for file in files:
            opened = self._open_file(file)
            seq_tuples += opened
        if os.name != "nt":
            dict_list = p_map(
                make_data_dictionary, sorted(seq_tuples), num_cpus=self.num_cpus
            )
        else:
            print("Multiprocessing is not supported on Windows, loading files sequentially.")
            dict_list = tqdm([make_data_dictionary(f) for f in files])

        self.visibility = {}
        self.min_frames = {}
        self.max_frames = {}
        self.original_coordinates = []
        self.metadata = []
        X = []
        for (
            names,
            lengths,
            coords,
            bp_dictionary,
            min_frames,
            max_frames,
            metadata,
        ) in dict_list:
            X += names
            self.original_coordinates += coords
            self.visibility.update(bp_dictionary)
            self.min_frames.update(min_frames)
            self.max_frames.update(max_frames)
            if metadata is not None:
                self.metadata += metadata
        del dict_list

        if len(self.metadata) != len(self.original_coordinates):
            self.metadata = None
        else:
            self.metadata = np.array(self.metadata)
        self.min_frames = dict(self.min_frames)
        self.max_frames = dict(self.max_frames)
        self.original_coordinates = np.array(self.original_coordinates)
        return np.array(X)

    @classmethod
    def get_file_ids(
        cls,
        filenames: Set = None,
        data_path: Union[str, Set] = None,
        file_paths: Set = None,
        *args,
        **kwargs,
    ) -> List:
        """
        Process data parameters and return a list of ids  of the videos that should
        be processed by the __init__ function

        Parameters
        ----------
        filenames : set, optional
            a set of string filenames to search for (only basenames, not the whole paths)
        data_path : str | set, optional
            the path to the folder where the pose and feature files are stored or a set of such paths
            (not passed if creating from key objects or from `file_paths`)
        file_paths : set, optional
            a set of string paths to the pose and feature files
            (not passed if creating from key objects or from `data_path`)

        Returns
        -------
        video_ids : list
            a list of video file ids
        """

        if file_paths is None:
            file_paths = []
        if data_path is not None:
            if isinstance(data_path, str):
                data_path = [data_path]
            file_paths = []
            for folder in data_path:
                file_paths += [os.path.join(folder, x) for x in os.listdir(folder)]
        ids = set()
        for f in file_paths:
            if os.path.basename(f) in filenames:
                ids.add(os.path.basename(f))
        ids = sorted(ids)
        return ids

    @abstractmethod
    def _open_file(self, filename: str) -> List:
        """
        Open a file and make a list of sequences

        The sequence objects should contain information about all clips in one video. The sequences and
        video ids will be processed in the `_get_data` function.

        Parameters
        ----------
        filename : str
            the name of the file

        Returns
        -------
        video_tuples : list
            a list of video tuples: `(video_id, sequence)`
        """

    @abstractmethod
    def _get_data(
        self, video_id: str, sequence, default_agent_name: str
    ) -> Tuple[Dict, Optional[Dict]]:
        """
        Get the keypoint dataframes from a sequence

        The sequences and video ids are generated in the `_open_file` function.
        In `data_dictionary`, the keys are clip ids and the values are `pandas` dataframes with two-level indices.
        The first level is the frame numbers and the second is the body part names. The dataframes should have from
        two to four columns labeled `"x"`, `"y"` and (optionally) `"z"` and `"likelihood"`. Each frame should have
        information on all the body parts. You don't have to filter the data in any way or fill the nans, it will
        be done automatically.

        Parameters
        ----------
        video_id : str
            the video id
        sequence
            an object containing information about all clips in one video
        default_agent_name

        Returns
        -------
        data dictionary : dict
            a dictionary where the keys are clip ids and the values are keypoint dataframes (see above for details)
        metadata_dictionary : dict
            a dictionary where the keys are clip ids and the values are metadata objects (can be any additional information,
            like the annotator tag; for no metadata pass `None`)
        """


class DLCTrackStore(FileInputStore):
    """
    DLC track data

    Assumes the following file structure:
    ```
    data_path
    ├── video1DLC1000.pickle
    ├── video2DLC400.pickle
    ├── video1_features.pt
    └── video2_features.pt
    ```
    Here `data_suffix` is `{'DLC1000.pickle', 'DLC400.pickle'}` and `feature_suffix` (optional) is `'_features.pt'`.

    The feature files should to be dictionaries where keys are clip IDs (e.g. animal names) and values are
    feature values (arrays of shape `(#frames, #features)`). If the arrays are shaped as `(#features, #frames)`,
    set `transpose_features` to `True`.

    The files can be saved with `numpy.save()` (with `.npy` extension), `torch.save()` (with `.pt` extension) or
    with `pickle.dump()` (with `.pickle` or `.pkl` extension).
    """

    def _open_data(
        self, filename: str, default_agent_name: str
    ) -> Tuple[Dict, Optional[Dict]]:
        """
        Load the keypoints from filename and organize them in a dictionary

        In `data_dictionary`, the keys are clip ids and the values are `pandas` dataframes with two-level indices.
        The first level is the frame numbers and the second is the body part names. The dataframes should have from
        two to four columns labeled `"x"`, `"y"` and (optionally) `"z"` and `"likelihood"`. Each frame should have
        information on all the body parts. You don't have to filter the data in any way or fill the nans, it will
        be done automatically.

        Parameters
        ----------
        filename : str
            path to the pose file
        default_clip_name : str
            the name to assign to a clip if it does not have a name in the raw data

        Returns
        -------
        data dictionary : dict
            a dictionary where the keys are clip ids and the values are keypoint dataframes (see above for details)
        metadata_dictionary : dict
            a dictionary where the keys are clip ids and the values are metadata objects (can be any additional information,
            like the annotator tag; for no metadata pass `None`)
        """

        if filename.endswith("h5"):
            temp = pd.read_hdf(filename)
            temp = temp.droplevel("scorer", axis=1)
        else:
            temp = pd.read_csv(filename, header=[1, 2])
            temp.columns.names = ["bodyparts", "coords"]
        if "individuals" not in temp.columns.names:
            old_idx = temp.columns.to_frame()
            old_idx.insert(0, "individuals", self.default_agent_name)
            temp.columns = pd.MultiIndex.from_frame(old_idx)
        df = temp.stack(["individuals", "bodyparts"])
        idx = pd.MultiIndex.from_product(
            [df.index.levels[0], df.index.levels[1], df.index.levels[2]],
            names=df.index.names,
        )
        df = df.reindex(idx).fillna(value=0)
        animals = sorted(list(df.index.levels[1]))
        dic = {}
        for ind in animals:
            coord = df.iloc[df.index.get_level_values(1) == ind].droplevel(1)
            coord = coord[["x", "y", "likelihood"]]
            dic[ind] = coord

        return dic, None


class DLCTrackletStore(FileInputStore):
    """
    DLC tracklet data

    Assumes the following file structure:
    ```
    data_path
    ├── video1DLC1000.pickle
    ├── video2DLC400.pickle
    ├── video1_features.pt
    └── video2_features.pt
    ```
    Here `data_suffix` is `{'DLC1000.pickle', 'DLC400.pickle'}` and `feature_suffix` (optional) is `'_features.pt'`.

    The feature files should to be dictionaries where keys are clip IDs (e.g. animal names) and values are
    feature values (arrays of shape `(#frames, #features)`). If the arrays are shaped as `(#features, #frames)`,
    set `transpose_features` to `True`.

    The files can be saved with `numpy.save()` (with `.npy` extension), `torch.save()` (with `.pt` extension) or
    with `pickle.dump()` (with `.pickle` or `.pkl` extension).
    """

    def _open_data(
        self, filename: str, default_agent_name: str
    ) -> Tuple[Dict, Optional[Dict]]:
        """
        Load the keypoints from filename and organize them in a dictionary

        In `data_dictionary`, the keys are clip ids and the values are `pandas` dataframes with two-level indices.
        The first level is the frame numbers and the second is the body part names. The dataframes should have from
        two to four columns labeled `"x"`, `"y"` and (optionally) `"z"` and `"likelihood"`. Each frame should have
        information on all the body parts. You don't have to filter the data in any way or fill the nans, it will
        be done automatically.

        Parameters
        ----------
        filename : str
            path to the pose file
        default_clip_name : str
            the name to assign to a clip if it does not have a name in the raw data

        Returns
        -------
        data dictionary : dict
            a dictionary where the keys are clip ids and the values are keypoint dataframes (see above for details)
        metadata_dictionary : dict
            a dictionary where the keys are clip ids and the values are metadata objects (can be any additional information,
            like the annotator tag; for no metadata pass `None`)
        """

        output = {}
        with open(filename, "rb") as f:
            data_p = pickle.load(f)
        header = data_p["header"]
        bodyparts = header.unique("bodyparts")

        keys = sorted([key for key in data_p.keys() if key != "header"])
        min_frames = defaultdict(lambda: 10**5)
        max_frames = defaultdict(lambda: 0)
        for tr_id in keys:
            coords = {}
            fr_i = int(list(data_p[tr_id].keys())[0][5:]) - 1
            for frame in data_p[tr_id]:
                count = 0
                while int(frame[5:]) > fr_i + 1:
                    count += 1
                    fr_i = fr_i + 1
                    if count <= 3:
                        for bp, name in enumerate(bodyparts):
                            coords[(fr_i, name)] = coords[(fr_i - 1, name)]
                    else:
                        for bp, name in enumerate(bodyparts):
                            coords[(fr_i, name)] = np.zeros(
                                coords[(fr_i - 1, name)].shape
                            )
                fr_i = int(frame[5:])
                if fr_i > max_frames[f"ind{tr_id}"]:
                    max_frames[f"ind{tr_id}"] = fr_i
                if fr_i < min_frames[f"ind{tr_id}"]:
                    min_frames[f"ind{tr_id}"] = fr_i
                for bp, name in enumerate(bodyparts):
                    coords[(fr_i, name)] = data_p[tr_id][frame][bp][:3]

            output[f"ind{tr_id}"] = pd.DataFrame(
                data=coords, index=["x", "y", "likelihood"]
            ).T
        return output, None


class PKUMMDInputStore(FileInputStore):
    """
    PKU-MMD data

    Assumes the following file structure:
    ```
    data_path
    ├── 0073-R.txt
    ...
    └── 0274-L.txt
    ```
    """

    data_suffix = ".txt"

    def __init__(
        self,
        video_order: str = None,
        data_path: Union[str, Set] = None,
        file_paths: Set = None,
        feature_save_path: str = None,
        feature_extraction: str = "kinematic",
        len_segment: int = 128,
        overlap: int = 0,
        key_objects: Tuple = None,
        num_cpus: int = None,
        interactive: bool = False,
        feature_extraction_pars: Dict = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        video_order : list, optional
            a list of video ids that should be processed in the same order (not passed if creating from key objects)
        data_path : str | set, optional
            the path to the folder where the pose and feature files are stored or a set of such paths
            (not passed if creating from key objects or from `file_paths`)
        file_paths : set, optional
            a set of string paths to the pose and feature files
            (not passed if creating from key objects or from `data_path`)
        feature_save_path : str, optional
            the path to the folder where pre-processed files are stored (not passed if creating from key objects)
        feature_extraction : str, default 'kinematic'
            the feature extraction method (run options.feature_extractors to see available options)
        len_segment : int, default 128
            the length of the segments in which the data should be cut (in frames)
        overlap : int, default 0
            the length of the overlap between neighboring segments (in frames)
        interactive : bool, default False
            if True, distances between two agents are included; if False, only the first agent features are computed
        key_objects : tuple, optional
            a tuple of key objects
        num_cpus : int, optional
            the number of cpus to use in data processing
        feature_extraction_pars : dict, optional
            parameters of the feature extractor
        """

        if feature_extraction_pars is None:
            feature_extraction_pars = {}
        feature_extraction_pars["interactive"] = interactive
        self.interactive = interactive
        super().__init__(
            video_order,
            data_path,
            file_paths,
            data_suffix=".txt",
            data_prefix=None,
            feature_suffix=None,
            convert_int_indices=False,
            feature_save_path=feature_save_path,
            canvas_shape=[1, 1, 1],
            len_segment=len_segment,
            overlap=overlap,
            feature_extraction=feature_extraction,
            ignored_clips=None,
            ignored_bodyparts=None,
            default_agent_name="ind0",
            key_objects=key_objects,
            likelihood_threshold=0,
            num_cpus=num_cpus,
            frame_limit=1,
            interactive=interactive,
            feature_extraction_pars=feature_extraction_pars,
        )

    def _open_data(
        self, filename: str, default_clip_name: str
    ) -> Tuple[Dict, Optional[Dict]]:
        """
        Load the keypoints from filename and organize them in a dictionary

        In `data_dictionary`, the keys are clip ids and the values are `pandas` dataframes with two-level indices.
        The first level is the frame numbers and the second is the body part names. The dataframes should have from
        two to four columns labeled `"x"`, `"y"` and (optionally) `"z"` and `"likelihood"`. Each frame should have
        information on all the body parts. You don't have to filter the data in any way or fill the nans, it will
        be done automatically.

        Parameters
        ----------
        filename : str
            path to the pose file
        default_clip_name : str
            the name to assign to a clip if it does not have a name in the raw data

        Returns
        -------
        data dictionary : dict
            a dictionary where the keys are clip ids and the values are keypoint dataframes (see above for details)
        metadata_dictionary : dict
            a dictionary where the keys are clip ids and the values are metadata objects (can be any additional information,
            like the annotator tag; for no metadata pass `None`)
        """

        keypoint_dict = {"0": [], "1": []}
        with open(filename) as f:
            for line in f.readlines():
                line_data = list(map(float, line.split()))
                line_data = np.array(line_data)
                line_data = line_data.reshape((2, 25, 3))[:, :, [0, 2, 1]]
                for ind in keypoint_dict:
                    keypoint_dict[ind].append(line_data[int(ind)])
        for ind in keypoint_dict:
            data = np.stack(keypoint_dict[ind])
            mi = pd.MultiIndex.from_product(
                [list(range(data.shape[0])), list(range(data.shape[1]))]
            )
            data = data.reshape((-1, 3))
            keypoint_dict[ind] = pd.DataFrame(
                data=data, index=mi, columns=["x", "y", "z"]
            )
        if not self.interactive:
            keypoint_dict.pop("1")
        return keypoint_dict, None


class CalMS21InputStore(SequenceInputStore):
    """
    CalMS21 data

    Use the `'random:test_from_name:{name}'` and `'val-from-name:{val_name}:test-from-name:{test_name}'`
    partitioning methods with `'train'`, `'test'` and `'unlabeled'` names to separate into train, test and validation
    subsets according to the original files. For example, with `'val-from-name:test:test-from-name:unlabeled'`
    the data from the test file will go into validation and the unlabeled files will be the test.

    Assumes the following file structure:
    ```
    data_path
    ├── calms21_task1_train.npy
    ├── calms21_task1_test.npy
    ├── calms21_task1_test_features.npy
    ├── calms21_task1_test_features.npy
    ├── calms21_unlabeled_videos_part1.npy
    ├── calms21_unlabeled_videos_part1.npy
    ├── calms21_unlabeled_videos_part2.npy
    └── calms21_unlabeled_videos_part3.npy
    ```
    """

    def __init__(
        self,
        video_order: List = None,
        data_path: Union[Set, str] = None,
        file_paths: Set = None,
        task_n: int = 1,
        include_task1: bool = True,
        feature_save_path: str = None,
        len_segment: int = 128,
        overlap: int = 0,
        feature_extraction: str = "kinematic",
        key_objects: Dict = None,
        treba_files: bool = False,
        num_cpus: int = None,
        feature_extraction_pars: Dict = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        video_order : list, optional
            a list of video ids that should be processed in the same order (not passed if creating from key objects)
        data_path : str | set, optional
            the path to the folder where the pose and feature files are stored or a set of such paths
            (not passed if creating from key objects or from `file_paths`)
        file_paths : set, optional
            a set of string paths to the pose and feature files
            (not passed if creating from key objects or from `data_path`)
        task_n : [1, 2]
            the number of the task
        include_task1 : bool, default True
            include task 1 data to training set
        feature_save_path : str, optional
            the path to the folder where pre-processed files are stored (not passed if creating from key objects)
        len_segment : int, default 128
            the length of the segments in which the data should be cut (in frames)
        overlap : int, default 0
            the length of the overlap between neighboring segments (in frames)
        feature_extraction : str, default 'kinematic'
            the feature extraction method (see options.feature_extractors for available options)
        ignored_bodyparts : list, optional
            list of strings of bodypart names to ignore
        key_objects : tuple, optional
            a tuple of key objects
        treba_files : bool, default False
            if `True`, TREBA feature files will be loaded
        num_cpus : int, optional
            the number of cpus to use in data processing
        feature_extraction_pars : dict, optional
            parameters of the feature extractor
        """

        self.task_n = int(task_n)
        self.include_task1 = include_task1
        self.treba_files = treba_files
        if feature_extraction_pars is not None:
            feature_extraction_pars["interactive"] = True

        super().__init__(
            video_order,
            data_path,
            file_paths,
            data_prefix=None,
            feature_suffix=None,
            convert_int_indices=False,
            feature_save_path=feature_save_path,
            canvas_shape=[1024, 570],
            len_segment=len_segment,
            overlap=overlap,
            feature_extraction=feature_extraction,
            ignored_clips=None,
            ignored_bodyparts=None,
            default_agent_name="ind0",
            key_objects=key_objects,
            likelihood_threshold=0,
            num_cpus=num_cpus,
            frame_limit=1,
            feature_extraction_pars=feature_extraction_pars,
        )

    @classmethod
    def get_file_ids(
        cls,
        task_n: int = 1,
        include_task1: bool = False,
        treba_files: bool = False,
        data_path: Union[str, Set] = None,
        file_paths=None,
        *args,
        **kwargs,
    ) -> Iterable:
        """
        Process data parameters and return a list of ids  of the videos that should
        be processed by the __init__ function

        Parameters
        ----------
        task_n : {1, 2, 3}
            the index of the CalMS21 challenge task
        include_task1 : bool, default False
            if `True`, the training file of the task 1 will be loaded
        treba_files : bool, default False
            if `True`, the TREBA feature files will be loaded
        filenames : set, optional
            a set of string filenames to search for (only basenames, not the whole paths)
        data_path : str | set, optional
            the path to the folder where the pose and feature files are stored or a set of such paths
            (not passed if creating from key objects or from `file_paths`)
        file_paths : set, optional
            a set of string paths to the pose and feature files
            (not passed if creating from key objects or from `data_path`)

        Returns
        -------
        video_ids : list
            a list of video file ids
        """

        task_n = int(task_n)
        if task_n == 1:
            include_task1 = False
        files = []
        if treba_files:
            postfix = "_features"
        else:
            postfix = ""
        files.append(f"calms21_task{task_n}_train{postfix}.npy")
        files.append(f"calms21_task{task_n}_test{postfix}.npy")
        if include_task1:
            files.append(f"calms21_task1_train{postfix}.npy")
        for i in range(1, 5):
            files.append(f"calms21_unlabeled_videos_part{i}{postfix}.npy")
        filenames = set(files)
        return SequenceInputStore.get_file_ids(filenames, data_path, file_paths)

    def _open_file(self, filename: str) -> List:
        """
        Open a file and make a list of sequences

        The sequence objects should contain information about all clips in one video. The sequences and
        video ids will be processed in the `_get_data` function.

        Parameters
        ----------
        filename : str
            the name of the file

        Returns
        -------
        video_tuples : list
            a list of video tuples: `(video_id, sequence)`
        """

        if os.path.basename(filename).startswith("calms21_unlabeled_videos"):
            mode = "unlabeled"
        elif os.path.basename(filename).startswith(f"calms21_task{self.task_n}_test"):
            mode = "test"
        else:
            mode = "train"
        data_dict = np.load(filename, allow_pickle=True).item()
        data = {}
        keys = list(data_dict.keys())
        for key in keys:
            data.update(data_dict[key])
            data_dict.pop(key)
        dict_list = [(f'{mode}--{k.split("/")[-1]}', v) for k, v in data.items()]
        return dict_list

    def _get_data(
        self, video_id: str, sequence, default_agent_name: str
    ) -> Tuple[Dict, Optional[Dict]]:
        """
        Get the keypoint dataframes from a sequence

        The sequences and video ids are generated in the `_open_file` function.
        In `data_dictionary`, the keys are clip ids and the values are `pandas` dataframes with two-level indices.
        The first level is the frame numbers and the second is the body part names. The dataframes should have from
        two to four columns labeled `"x"`, `"y"` and (optionally) `"z"` and `"likelihood"`. Each frame should have
        information on all the body parts. You don't have to filter the data in any way or fill the nans, it will
        be done automatically.

        Parameters
        ----------
        video_id : str
            the video id
        sequence
            an object containing information about all clips in one video
        default_agent_name

        Returns
        -------
        data dictionary : dict
            a dictionary where the keys are clip ids and the values are keypoint dataframes (see above for details)
        metadata_dictionary : dict
            a dictionary where the keys are clip ids and the values are metadata objects (can be any additional information,
            like the annotator tag; for no metadata pass `None`)
        """

        if "metadata" in sequence:
            annotator = sequence["metadata"]["annotator-id"]
        else:
            annotator = 0
        bodyparts = [
            "nose",
            "left ear",
            "right ear",
            "neck",
            "left hip",
            "right hip",
            "tail",
        ]
        columns = ["x", "y"]
        if "keypoints" in sequence:
            sequence = sequence["keypoints"]
            index = pd.MultiIndex.from_product([range(sequence.shape[0]), bodyparts])
            data = {
                "mouse1": pd.DataFrame(
                    data=(sequence[:, 0, :, :]).transpose((0, 2, 1)).reshape(-1, 2),
                    columns=columns,
                    index=index,
                ),
                "mouse2": pd.DataFrame(
                    data=(sequence[:, 1, :, :]).transpose((0, 2, 1)).reshape(-1, 2),
                    columns=columns,
                    index=index,
                ),
            }
        else:
            sequence = sequence["features"]
            mice = sequence[:, :-32].reshape((-1, 2, 2, 7))
            index = pd.MultiIndex.from_product([range(mice.shape[0]), bodyparts])
            data = {
                "mouse1": pd.DataFrame(
                    data=(mice[:, 0, :, :]).transpose((0, 2, 1)).reshape(-1, 2),
                    columns=columns,
                    index=index,
                ),
                "mouse2": pd.DataFrame(
                    data=(mice[:, 1, :, :]).transpose((0, 2, 1)).reshape(-1, 2),
                    columns=columns,
                    index=index,
                ),
                "loaded": sequence[:, -32:],
            }
        metadata = {k: annotator for k in data.keys()}
        return data, metadata


class Numpy3DInputStore(FileInputStore):
    """
    3D data

    Assumes the data files to be `numpy` arrays saved in `.npy` format with shape `(#frames, #keypoints, 3)`.

    Assumes the following file structure:
    ```
    data_path
    ├── video1_suffix1.npy
    ├── video2_suffix2.npy
    ├── video1_features.pt
    └── video2_features.pt
    ```
    Here `data_suffix` is `{'_suffix1.npy', '_suffix1.npy'}` and `feature_suffix` (optional) is `'_features.pt'`.

    The feature files should to be dictionaries where keys are clip IDs (e.g. animal names) and values are
    feature values (arrays of shape `(#frames, #features)`). If the arrays are shaped as `(#features, #frames)`,
    set `transpose_features` to `True`.

    The files can be saved with `numpy.save()` (with `.npy` extension), `torch.save()` (with `.pt` extension) or
    with `pickle.dump()` (with `.pickle` or `.pkl` extension).
    """

    def __init__(
        self,
        video_order: List = None,
        data_path: Union[Set, str] = None,
        file_paths: Set = None,
        data_suffix: Union[Set, str] = None,
        data_prefix: Union[Set, str] = None,
        feature_suffix: Union[Set, str] = None,
        convert_int_indices: bool = True,
        feature_save_path: str = None,
        canvas_shape: List = None,
        len_segment: int = 128,
        overlap: int = 0,
        feature_extraction: str = "kinematic",
        ignored_clips: List = None,
        ignored_bodyparts: List = None,
        default_agent_name: str = "ind0",
        key_objects: Dict = None,
        likelihood_threshold: float = 0,
        num_cpus: int = None,
        frame_limit: int = 1,
        feature_extraction_pars: Dict = None,
        centered: bool = False,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        video_order : list, optional
            a list of video ids that should be processed in the same order (not passed if creating from key objects
        data_path : str | set, optional
            the path to the folder where the pose and feature files are stored or a set of such paths
            (not passed if creating from key objects or from `file_paths`)
        file_paths : set, optional
            a set of string paths to the pose and feature files
            (not passed if creating from key objects or from `data_path`)
        data_suffix : str | set, optional
            the suffix or the set of suffices such that the pose files are named {video_id}{data_suffix}
            (not passed if creating from key objects or if irrelevant for the dataset)
        data_prefix : str | set, optional
            the prefix or the set of prefixes such that the pose files for different video views of the same
            clip are named {prefix}{sep}{video_id}{data_suffix} (not passed if creating from key objects
            or if irrelevant for the dataset)
        feature_suffix : str | set, optional
            the suffix or the set of suffices such that the additional feature files are named
            {video_id}{feature_suffix} (and placed at the data_path folder)
        convert_int_indices : bool, default True
            if `True`, convert any integer key `i` in feature files to `'ind{i}'`
        feature_save_path : str, optional
            the path to the folder where pre-processed files are stored (not passed if creating from key objects)
        canvas_shape : List, default [1, 1]
            the canvas size where the pose is defined
        len_segment : int, default 128
            the length of the segments in which the data should be cut (in frames)
        overlap : int, default 0
            the length of the overlap between neighboring segments (in frames)
        feature_extraction : str, default 'kinematic'
            the feature extraction method (see options.feature_extractors for available options)
        ignored_clips : list, optional
            list of strings of clip ids to ignore
        ignored_bodyparts : list, optional
            list of strings of bodypart names to ignore
        default_agent_name : str, default 'ind0'
            the agent name used as default in the pose files for a single agent
        key_objects : tuple, optional
            a tuple of key objects
        likelihood_threshold : float, default 0
            coordinate values with likelihoods less than this value will be set to 'unknown'
        num_cpus : int, optional
            the number of cpus to use in data processing
        frame_limit : int, default 1
            clips shorter than this number of frames will be ignored
        feature_extraction_pars : dict, optional
            parameters of the feature extractor
        """

        super().__init__(
            video_order,
            data_path,
            file_paths,
            data_suffix=data_suffix,
            data_prefix=data_prefix,
            feature_suffix=feature_suffix,
            convert_int_indices=convert_int_indices,
            feature_save_path=feature_save_path,
            canvas_shape=canvas_shape,
            len_segment=len_segment,
            overlap=overlap,
            feature_extraction=feature_extraction,
            ignored_clips=ignored_clips,
            ignored_bodyparts=ignored_bodyparts,
            default_agent_name=default_agent_name,
            key_objects=key_objects,
            likelihood_threshold=likelihood_threshold,
            num_cpus=num_cpus,
            frame_limit=frame_limit,
            feature_extraction_pars=feature_extraction_pars,
            centered=centered,
        )

    def _open_data(
        self, filename: str, default_clip_name: str
    ) -> Tuple[Dict, Optional[Dict]]:
        """
        Load the keypoints from filename and organize them in a dictionary

        In `data_dictionary`, the keys are clip ids and the values are `pandas` dataframes with two-level indices.
        The first level is the frame numbers and the second is the body part names. The dataframes should have from
        two to four columns labeled `"x"`, `"y"` and (optionally) `"z"` and `"likelihood"`. Each frame should have
        information on all the body parts. You don't have to filter the data in any way or fill the nans, it will
        be done automatically.

        Parameters
        ----------
        filename : str
            path to the pose file
        default_clip_name : str
            the name to assign to a clip if it does not have a name in the raw data

        Returns
        -------
        data dictionary : dict
            a dictionary where the keys are clip ids and the values are keypoint dataframes (see above for details)
        metadata_dictionary : dict
            a dictionary where the keys are clip ids and the values are metadata objects (can be any additional information,
            like the annotator tag; for no metadata pass `None`)
        """

        data = np.load(filename)
        bodyparts = [str(i) for i in range(data.shape[1])]
        clip_id = self.default_agent_name
        columns = ["x", "y", "z"]
        index = pd.MultiIndex.from_product([range(data.shape[0]), bodyparts])
        data_dict = {
            clip_id: pd.DataFrame(
                data=data.reshape(-1, 3), columns=columns, index=index
            )
        }
        return data_dict, None


class LoadedFeaturesInputStore(GeneralInputStore):
    """
    Non-pose feature files

    The feature files should to be dictionaries where keys are clip IDs (e.g. animal names) and values are
    feature values (arrays of shape `(#frames, #features)`). If the arrays are shaped as `(#features, #frames)`,
    set `transpose_features` to `True`.

    The files can be saved with `numpy.save()` (with `.npy` extension), `torch.save()` (with `.pt` extension) or
    with `pickle.dump()` (with `.pickle` or `.pkl` extension).

    Assumes the following file structure:
    ```
    data_path
    ├── video1_features.pt
    └── video2_features.pt
    ```
    Here `feature_suffix` (optional) is `'_features.pt'`.
    """

    def __init__(
        self,
        video_order: List = None,
        data_path: Union[Set, str] = None,
        file_paths: Set = None,
        feature_suffix: Union[Set, str] = None,
        convert_int_indices: bool = True,
        feature_save_path: str = None,
        len_segment: int = 128,
        overlap: int = 0,
        ignored_clips: List = None,
        key_objects: Dict = None,
        num_cpus: int = None,
        frame_limit: int = 1,
        transpose_features: bool = False,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        video_order : list, optional
            a list of video ids that should be processed in the same order (not passed if creating from key objects
        data_path : str | set, optional
            the path to the folder where the pose and feature files are stored or a set of such paths
            (not passed if creating from key objects or from `file_paths`)
        file_paths : set, optional
            a set of string paths to the pose and feature files
            (not passed if creating from key objects or from `data_path`)
        feature_suffix : str | set, optional
            the suffix or the set of suffices such that the additional feature files are named
            {video_id}{feature_suffix} (and placed at the data_path folder)
        feature_save_path : str, optional
            the path to the folder where pre-processed files are stored (not passed if creating from key objects)
        len_segment : int, default 128
            the length of the segments in which the data should be cut (in frames)
        overlap : int, default 0
            the length of the overlap between neighboring segments (in frames)
        ignored_clips : list, optional
            list of strings of clip ids to ignore
        default_agent_name : str, default 'ind0'
            the agent name used as default in the pose files for a single agent
        key_objects : tuple, optional
            a tuple of key objects
        num_cpus : int, optional
            the number of cpus to use in data processing
        frame_limit : int, default 1
            clips shorter than this number of frames will be ignored
        feature_extraction_pars : dict, optional
            parameters of the feature extractor
        """

        super().__init__(
            video_order,
            data_path,
            file_paths,
            feature_suffix=feature_suffix,
            convert_int_indices=convert_int_indices,
            feature_save_path=feature_save_path,
            len_segment=len_segment,
            overlap=overlap,
            ignored_clips=ignored_clips,
            key_objects=key_objects,
            num_cpus=num_cpus,
            frame_limit=frame_limit,
            transpose_features=transpose_features,
        )

    def get_visibility(
        self, video_id: str, clip_id: str, start: int, end: int, score: int
    ) -> float:
        """
        Get the fraction of the frames in that have a visibility score better than a hard_threshold

        For example, in the case of keypoint data the visibility score can be the number of identified keypoints.

        Parameters
        ----------
        video_id : str
            the video id of the frames
        clip_id : str
            the clip id of the frames
        start : int
            the start frame
        end : int
            the end frame
        score : float
            the visibility score hard_threshold

        Returns
        -------
        frac_visible: float
            the fraction of frames with visibility above the hard_threshold
        """

        return 1

    def _generate_features(
        self, video_id: str
    ) -> Tuple[Dict, Dict, Dict, Union[str, int]]:
        """
        Generate features from the raw coordinates
        """

        features = defaultdict(lambda: {})
        loaded_features = self._load_saved_features(video_id)
        min_frames = None
        max_frames = None
        video_tag = None
        for clip_id, feature_tensor in loaded_features.items():
            if clip_id == "max_frames":
                max_frames = feature_tensor
            elif clip_id == "min_frames":
                min_frames = feature_tensor
            elif clip_id == "video_tag":
                video_tag = feature_tensor
            else:
                if not isinstance(feature_tensor, torch.Tensor):
                    feature_tensor = torch.tensor(feature_tensor)
                if self.convert_int_indices and (
                    isinstance(clip_id, int) or isinstance(clip_id, np.integer)
                ):
                    clip_id = f"ind{clip_id}"
                key = f"{os.path.basename(video_id)}---{clip_id}"
                features[key]["loaded"] = feature_tensor
        if min_frames is None:
            min_frames = {}
            for key, value in features.items():
                video_id, clip_id = key.split("---")
                min_frames[clip_id] = 0
        if max_frames is None:
            max_frames = {}
            for key, value in features.items():
                video_id, clip_id = key.split("---")
                max_frames[clip_id] = value["loaded"].shape[0] - 1 + min_frames[clip_id]
        return features, min_frames, max_frames, video_tag

    def _load_data(self) -> np.array:
        """
        Load input data and generate data prompts
        """

        if self.video_order is None:
            return None

        files = []
        for video_id in self.video_order:
            for f in self.file_paths:
                if f.endswith(tuple(self.feature_suffix)):
                    bn = os.path.basename(f)
                    if video_id == strip_suffix(bn, self.feature_suffix):
                        files.append(f)

        def make_data_dictionary(filename):
            name = strip_suffix(filename, self.feature_suffix)
            name = os.path.basename(name)
            data_dict, min_frames, max_frames, video_tag = self._generate_features(name)
            bp_dict = defaultdict(lambda: {})
            for key, value in data_dict.items():
                video_id, clip_id = key.split("---")
                bp_dict[video_id][clip_id] = 1
            min_frames = {name: min_frames}  # name is e.g. 20190707T1126-1226
            max_frames = {name: max_frames}
            names, lengths, coords = self._make_trimmed_data(data_dict)
            return names, lengths, coords, bp_dict, min_frames, max_frames, video_tag

        if os.name != "nt":
            dict_list = p_map(make_data_dictionary, files, num_cpus=self.num_cpus)
        else:
            print("Multiprocessing is not supported on Windows, loading files sequentially.")
            dict_list = tqdm([make_data_dictionary(f) for f in files])

        self.visibility = {}
        self.min_frames = {}
        self.max_frames = {}
        self.original_coordinates = []
        self.metadata = []
        X = []
        for (
            names,
            lengths,
            coords,
            bp_dictionary,
            min_frames,
            max_frames,
            metadata,
        ) in dict_list:
            X += names
            self.original_coordinates += coords
            self.visibility.update(bp_dictionary)
            self.min_frames.update(min_frames)
            self.max_frames.update(max_frames)
            if metadata is not None:
                self.metadata += metadata
        del dict_list
        if len(self.metadata) != len(self.original_coordinates):
            self.metadata = None
        else:
            self.metadata = np.array(self.metadata)

        self.min_frames = dict(self.min_frames)
        self.max_frames = dict(self.max_frames)
        self.original_coordinates = np.array(self.original_coordinates)
        return np.array(X)

    @classmethod
    def get_file_ids(
        cls,
        data_path: Union[Set, str] = None,
        file_paths: Set = None,
        feature_suffix: Set = None,
        *args,
        **kwargs,
    ) -> List:
        """
        Process data parameters and return a list of ids  of the videos that should
        be processed by the __init__ function

        Parameters
        ----------
        data_suffix : set | str, optional
            the suffix (or a set of suffixes) of the input data files
        data_path : set | str, optional
            the path to the folder where the pose and feature files are stored or a set of such paths
            (not passed if creating from key objects or from `file_paths`)
        data_prefix : set | str, optional
            the prefix or the set of prefixes such that the pose files for different video views of the same
            clip are named {prefix}{sep}{video_id}{data_suffix} (not passed if creating from key objects
            or if irrelevant for the dataset)
        file_paths : set, optional
            a set of string paths to the pose and feature files
        feature_suffix : str | set, optional
            the suffix or the set of suffices such that the additional feature files are named
            {video_id}{feature_suffix} (and placed at the `data_path` folder or at `file_paths`)

        Returns
        -------
        video_ids : list
            a list of video file ids
        """

        if feature_suffix is None:
            feature_suffix = []
        if isinstance(feature_suffix, str):
            feature_suffix = [feature_suffix]
        feature_suffix = tuple(feature_suffix)
        if file_paths is None:
            file_paths = []
        if data_path is not None:
            if isinstance(data_path, str):
                data_path = [data_path]
            file_paths = []
            for folder in data_path:
                file_paths += [os.path.join(folder, x) for x in os.listdir(folder)]
        ids = set()
        for f in file_paths:
            if f.endswith(feature_suffix):
                bn = os.path.basename(f)
                video_id = strip_suffix(bn, feature_suffix)
                ids.add(video_id)
        ids = sorted(ids)
        return ids


class SIMBAInputStore(FileInputStore):
    """
    SIMBA paper format data

     Assumes the following file structure

     ```
     data_path
     ├── Video1.csv
     ...
     └── Video9.csv
     ```
     Here `data_suffix` is `.csv`.
    """

    def __init__(
        self,
        video_order: List = None,
        data_path: Union[Set, str] = None,
        file_paths: Set = None,
        data_prefix: Union[Set, str] = None,
        feature_suffix: str = None,
        feature_save_path: str = None,
        canvas_shape: List = None,
        len_segment: int = 128,
        overlap: int = 0,
        feature_extraction: str = "kinematic",
        ignored_clips: List = None,
        ignored_bodyparts: List = None,
        key_objects: Tuple = None,
        likelihood_threshold: float = 0,
        num_cpus: int = None,
        normalize: bool = False,
        feature_extraction_pars: Dict = None,
        centered: bool = False,
        data_suffix: str = None,
        use_features: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        video_order : list, optional
            a list of video ids that should be processed in the same order (not passed if creating from key objects
        data_path : str | set, optional
            the path to the folder where the pose and feature files are stored or a set of such paths
            (not passed if creating from key objects or from `file_paths`)
        file_paths : set, optional
            a set of string paths to the pose and feature files
            (not passed if creating from key objects or from `data_path`)
        data_suffix : str | set, optional
            the suffix or the set of suffices such that the pose files are named {video_id}{data_suffix}
            (not passed if creating from key objects or if irrelevant for the dataset)
        data_prefix : str | set, optional
            the prefix or the set of prefixes such that the pose files for different video views of the same
            clip are named {prefix}{sep}{video_id}{data_suffix} (not passed if creating from key objects
            or if irrelevant for the dataset)
        feature_suffix : str | set, optional
            the suffix or the set of suffices such that the additional feature files are named
            {video_id}{feature_suffix} (and placed at the data_path folder)
        feature_save_path : str, optional
            the path to the folder where pre-processed files are stored (not passed if creating from key objects)
        canvas_shape : List, default [1, 1]
            the canvas size where the pose is defined
        len_segment : int, default 128
            the length of the segments in which the data should be cut (in frames)
        overlap : int, default 0
            the length of the overlap between neighboring segments (in frames)
        feature_extraction : str, default 'kinematic'
            the feature extraction method (see options.feature_extractors for available options)
        ignored_clips : list, optional
            list of strings of clip ids to ignore
        ignored_bodyparts : list, optional
            list of strings of bodypart names to ignore
        key_objects : tuple, optional
            a tuple of key objects
        likelihood_threshold : float, default 0
            coordinate values with likelihoods less than this value will be set to 'unknown'
        num_cpus : int, optional
            the number of cpus to use in data processing
        feature_extraction_pars : dict, optional
            parameters of the feature extractor
        """

        self.use_features = use_features
        if feature_extraction_pars is not None:
            feature_extraction_pars["interactive"] = True
        super().__init__(
            video_order=video_order,
            data_path=data_path,
            file_paths=file_paths,
            data_suffix=data_suffix,
            data_prefix=data_prefix,
            feature_suffix=feature_suffix,
            convert_int_indices=False,
            feature_save_path=feature_save_path,
            canvas_shape=canvas_shape,
            len_segment=len_segment,
            overlap=overlap,
            feature_extraction=feature_extraction,
            ignored_clips=ignored_clips,
            ignored_bodyparts=ignored_bodyparts,
            default_agent_name="",
            key_objects=key_objects,
            likelihood_threshold=likelihood_threshold,
            num_cpus=num_cpus,
            min_frames=0,
            normalize=normalize,
            feature_extraction_pars=feature_extraction_pars,
            centered=centered,
        )

    def _open_data(
        self, filename: str, default_clip_name: str
    ) -> Tuple[Dict, Optional[Dict]]:
        data = pd.read_csv(filename)
        output = {}
        column_dict = {"x": "x", "y": "y", "z": "z", "p": "likelihood"}
        columns = [x for x in data.columns if x.split("_")[-1] in column_dict]
        animals = sorted(set([x.split("_")[-2] for x in columns]))
        coords = sorted(set([x.split("_")[-1] for x in columns]))
        names = sorted(set(["_".join(x.split("_")[:-2]) for x in columns]))
        for animal in animals:
            data_dict = {}
            for i, row in data.iterrows():
                for col_name in names:
                    data_dict[(i, col_name)] = [
                        row[f"{col_name}_{animal}_{coord}"] for coord in coords
                    ]
            output[animal] = pd.DataFrame(data_dict).T
            output[animal].columns = [column_dict[x] for x in coords]
        if self.use_features:
            columns_to_avoid = [
                x
                for x in data.columns
                if x.split("_")[-1] in column_dict
                or x.split("_")[-1].startswith("prediction")
            ]
            columns_to_avoid += ["scorer", "frames", "video_no"]
            output["loaded"] = (
                data[[x for x in data.columns if x not in columns_to_avoid]]
                .interpolate()
                .values
            )
        return output, None
