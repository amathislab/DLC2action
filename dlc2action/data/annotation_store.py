#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
"""
Specific implementations of `dlc2action.data.base_store.AnnotationStore` are defined here
"""

from dlc2action.data.base_store import AnnotationStore
from dlc2action.utils import strip_suffix
from typing import Dict, List, Tuple, Set, Union
import torch
import numpy as np
import pickle
from copy import copy
from collections import Counter
from collections.abc import Iterable
from abc import abstractmethod
import os
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from itertools import combinations


class EmptyAnnotationStore(AnnotationStore):
    def __init__(
        self, video_order: List = None, key_objects: Tuple = None, *args, **kwargs
    ):
        """
        Parameters
        ----------
        video_order : list, optional
            a list of video ids that should be processed in the same order (not passed if creating from key objects)
        key_objects : tuple, optional
            a tuple of key objects
        """

        pass

    def __len__(self) -> int:
        """
        Get the number of available samples

        Returns
        -------
        length : int
            the number of available samples
        """

        return 0

    def remove(self, indices: List) -> None:
        """
        Remove the samples corresponding to indices

        Parameters
        ----------
        indices : int
            a list of integer indices to remove
        """

        pass

    def key_objects(self) -> Tuple:
        """
        Return a tuple of the key objects necessary to re-create the Store

        Returns
        -------
        key_objects : tuple
            a tuple of key objects
        """

        return ()

    def load_from_key_objects(self, key_objects: Tuple) -> None:
        """
        Load the information from a tuple of key objects

        Parameters
        ----------
        key_objects : tuple
            a tuple of key objects
        """

        pass

    def to_ram(self) -> None:
        """
        Transfer the data samples to RAM if they were previously stored as file paths
        """

        pass

    def get_original_coordinates(self) -> np.ndarray:
        """
        Return the original coordinates array

        Returns
        -------
        np.ndarray
            an array that contains the coordinates of the data samples in original input data (video id, clip id,
            start frame)
        """

        return None

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

        return self.new()

    @classmethod
    def get_file_ids(cls, *args, **kwargs) -> List:
        """
        Process data parameters and return a list of ids  of the videos that should
        be processed by the __init__ function

        Returns
        -------
        video_ids : list
            a list of video file ids
        """

        return None

    def __getitem__(self, ind: int) -> torch.Tensor:
        """
        Return the annotation of the sample corresponding to an index

        Parameters
        ----------
        ind : int
            index of the sample

        Returns
        -------
        sample : torch.Tensor
            the corresponding annotation tensor
        """

        return torch.tensor(float("nan"))

    def get_len(self, return_unlabeled: bool) -> int:
        """
        Get the length of the subsample of labeled/unlabeled data

        If return_unlabeled is True, the index is in the subsample of unlabeled data, if False in labeled
        and if return_unlabeled is None the index is already correct

        Parameters
        ----------
        return_unlabeled : bool
            the identifier for the subsample

        Returns
        -------
        length : int
            the length of the subsample
        """

        return None

    def get_idx(self, index: int, return_unlabeled: bool) -> int:
        """
        Convert from an index in the subsample of labeled/unlabeled data to an index in the full array

        If return_unlabeled is True, the index is in the subsample of unlabeled data, if False in labeled
        and if return_unlabeled is None the index is already correct

        Parameters
        ----------
        index : int
            the index in the subsample
        return_unlabeled : bool
            the identifier for the subsample

        Returns
        -------
        corrected_index : int
            the index in the full dataset
        """

        return index

    def count_classes(
        self, frac: bool = False, zeros: bool = False, bouts: bool = False
    ) -> Dict:
        """
        Get a dictionary with class-wise frame counts

        Parameters
        ----------
        frac : bool, default False
            if True, a fraction of the total frame count is returned

        Returns
        -------
        count_dictionary : dict
            a dictionary with class indices as keys and frame counts as values
        """

        return {}

    def behaviors_dict(self) -> Dict:
        """
        Get a dictionary of class names

        Returns
        -------
        behavior_dictionary: dict
            a dictionary with class indices as keys and class names as values
        """

        return {}

    def annotation_class(self) -> str:
        """
        Get the type of annotation ('exclusive_classification', 'nonexclusive_classification', more coming soon)

        Returns
        -------
        annotation_class : str
            the type of annotation
        """

        return "none"

    def size(self) -> int:
        """
        Get the total number of frames in the data

        Returns
        -------
        size : int
            the total number of frames
        """

        return None

    def filtered_indices(self) -> List:
        """
        Return the indices of the samples that should be removed

        Choosing the indices can be based on any kind of filering defined in the __init__ function by the data
        parameters

        Returns
        -------
        indices_to_remove : list
            a list of integer indices that should be removed
        """

        return []

    def set_pseudo_labels(self, labels: torch.Tensor) -> None:
        """
        Set pseudo labels to the unlabeled data

        Parameters
        ----------
        labels : torch.Tensor
            a tensor of pseudo-labels for the unlabeled data
        """

        pass


class ActionSegmentationStore(AnnotationStore):  # +
    """
    A general realization of an annotation store for action segmentation tasks

    Assumes the following file structure:
    ```
    annotation_path
    ├── video1_annotation.pickle
    └── video2_labels.pickle
    ```
    Here `annotation_suffix` is `{'_annotation.pickle', '_labels.pickle'}`.
    """

    def __init__(
        self,
        video_order: List = None,
        min_frames: Dict = None,
        max_frames: Dict = None,
        visibility: Dict = None,
        exclusive: bool = True,
        len_segment: int = 128,
        overlap: int = 0,
        behaviors: Set = None,
        ignored_classes: Set = None,
        ignored_clips: Set = None,
        annotation_suffix: Union[Set, str] = None,
        annotation_path: Union[Set, str] = None,
        behavior_file: str = None,
        correction: Dict = None,
        frame_limit: int = 0,
        filter_annotated: bool = False,
        filter_background: bool = False,
        error_class: str = None,
        min_frames_action: int = None,
        key_objects: Tuple = None,
        visibility_min_score: float = 0.2,
        visibility_min_frac: float = 0.7,
        mask: Dict = None,
        use_hard_negatives: bool = False,
        interactive: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        video_order : list, optional
            a list of video ids that should be processed in the same order (not passed if creating from key objects)
        min_frames : dict, optional
            a nested dictionary where first-level keys are video ids, second-level keys are clip ids and values are
            clip start frames (not passed if creating from key objects)
        max_frames : dict, optional
            a nested dictionary where first-level keys are video ids, second-level keys are clip ids and values are
            clip end frames (not passed if creating from key objects)
        visibility : dict, optional
            a nested dictionary where first-level keys are video ids, second-level keys are clip ids and values are
            visibility score arrays (not passed if creating from key objects or if irrelevant for the dataset)
        exclusive : bool, default True
            if True, the annotation is single-label; if False, multi-label
        len_segment : int, default 128
            the length of the segments in which the data should be cut (in frames)
        overlap : int, default 0
            the length of the overlap between neighboring segments (in frames)
        behaviors : set, optional
            the list of behaviors to put in the annotation (not passed if creating a blank instance or if behaviors are
            loaded from a file)
        ignored_classes : set, optional
            the list of behaviors from the behaviors list or file to not annotate
        ignored_clips : set, optional
            clip ids to ignore
        annotation_suffix : str | set, optional
            the suffix or the set of suffices such that the annotation files are named {video_id}{annotation_suffix}
            (not passed if creating from key objects or if irrelevant for the dataset)
        annotation_path : str | set, optional
            the path or the set of paths to the folder where the annotation files are stored (not passed if creating
            from key objects)
        behavior_file : str, optional
            the path to an .xlsx behavior file (not passed if creating from key objects or if irrelevant for the dataset)
        correction : dict, optional
            a dictionary of corrections for the labels (e.g. {'sleping': 'sleeping', 'calm locomotion': 'locomotion'},
            can be used to correct for variations in naming or to merge several labels in one
        frame_limit : int, default 0
            the smallest possible length of a clip (shorter clips are discarded)
        filter_annotated : bool, default False
            if True, the samples that do not have any labels will be filtered
        filter_background : bool, default False
            if True, only the unlabeled frames that are close to annotated frames will be labeled as background
        error_class : str, optional
            the name of the error class (the annotations that intersect with this label will be discarded)
        min_frames_action : int, default 0
            the minimum length of an action (shorter actions are not annotated)
        key_objects : tuple, optional
            the key objects to load the AnnotationStore from
        visibility_min_score : float, default 5
            the minimum visibility score for visibility filtering
        visibility_min_frac : float, default 0.7
            the minimum fraction of visible frames for visibility filtering
        mask : dict, optional
            a masked value dictionary (for active learning simulation experiments)
        use_hard_negatives : bool, default False
            mark hard negatives as 2 instead of 0 or 1, for loss functions that have options for hard negative processing
        interactive : bool, default False
            if `True`, annotation is assigned to pairs of individuals
        """

        super().__init__()

        if ignored_clips is None:
            ignored_clips = []
        self.len_segment = int(len_segment)
        self.exclusive = exclusive
        if overlap < 1:
            overlap = overlap * len_segment
        self.overlap = int(overlap)
        self.video_order = video_order
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.visibility = visibility
        self.vis_min_score = visibility_min_score
        self.vis_min_frac = visibility_min_frac
        self.mask = mask
        self.use_negatives = use_hard_negatives
        self.interactive = interactive
        self.ignored_clips = ignored_clips
        self.file_paths = self._get_file_paths(annotation_path)
        self.ignored_classes = ignored_classes
        self.update_behaviors = False

        self.ram = True
        self.original_coordinates = []
        self.filtered = []

        self.step = self.len_segment - self.overlap

        self.ann_suffix = annotation_suffix
        self.annotation_folder = annotation_path
        self.filter_annotated = filter_annotated
        self.filter_background = filter_background
        self.frame_limit = frame_limit
        self.min_frames_action = min_frames_action
        self.error_class = error_class

        if correction is None:
            correction = {}
        self.correction = correction

        if self.max_frames is None:
            self.max_frames = defaultdict(lambda: {})
        if self.min_frames is None:
            self.min_frames = defaultdict(lambda: {})

        lists = [self.annotation_folder, self.ann_suffix]
        for i in range(len(lists)):
            iterable = isinstance(lists[i], Iterable) * (not isinstance(lists[i], str))
            if lists[i] is not None:
                if not iterable:
                    lists[i] = [lists[i]]
                lists[i] = [x for x in lists[i]]
        self.annotation_folder, self.ann_suffix = lists

        if ignored_classes is None:
            ignored_classes = []
        self.ignored_classes = ignored_classes
        self._set_behaviors(behaviors, ignored_classes, behavior_file)

        if key_objects is None and self.video_order is not None:
            self.data = self._load_data()
        elif key_objects is not None:
            self.load_from_key_objects(key_objects)
        else:
            self.data = None
        self.labeled_indices, self.unlabeled_indices = self._compute_labeled()

    def __getitem__(self, ind):
        if self.data is None:
            raise RuntimeError("The annotation store data has not been initialized!")
        return self.data[ind]

    def __len__(self) -> int:
        if self.data is None:
            raise RuntimeError("The annotation store data has not been initialized!")
        return len(self.data)

    def remove(self, indices: List) -> None:
        """
        Remove the samples corresponding to indices

        Parameters
        ----------
        indices : list
            a list of integer indices to remove
        """

        if len(indices) > 0:
            mask = np.ones(len(self.data))
            mask[indices] = 0
            mask = mask.astype(bool)
            self.data = self.data[mask]
            self.original_coordinates = self.original_coordinates[mask]

    def key_objects(self) -> Tuple:
        """
        Return a tuple of the key objects necessary to re-create the Store

        Returns
        -------
        key_objects : tuple
            a tuple of key objects
        """

        return (
            self.original_coordinates,
            self.data,
            self.behaviors,
            self.exclusive,
            self.len_segment,
            self.step,
            self.overlap,
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
            self.data,
            self.behaviors,
            self.exclusive,
            self.len_segment,
            self.step,
            self.overlap,
        ) = key_objects
        self.labeled_indices, self.unlabeled_indices = self._compute_labeled()

    def to_ram(self) -> None:
        """
        Transfer the data samples to RAM if they were previously stored as file paths
        """

        pass

    def get_original_coordinates(self) -> np.ndarray:
        """
        Return the video_indices array

        Returns
        -------
        original_coordinates : numpy.ndarray
            an array that contains the coordinates of the data samples in original input data
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
        data = copy(self.data)
        data[ssl_indices, ...] = -100
        new = self.new()
        new.original_coordinates = self.original_coordinates[indices + ssl_indices]
        new.data = self.data[indices + ssl_indices]
        new.labeled_indices, new.unlabeled_indices = new._compute_labeled()
        new.behaviors = self.behaviors
        new.exclusive = self.exclusive
        new.len_segment = self.len_segment
        new.step = self.step
        new.overlap = self.overlap
        new.max_frames = self.max_frames
        new.min_frames = self.min_frames
        return new

    def get_len(self, return_unlabeled: bool) -> int:
        """
        Get the length of the subsample of labeled/unlabeled data

        If return_unlabeled is True, the index is in the subsample of unlabeled data, if False in labeled
        and if return_unlabeled is None the index is already correct

        Parameters
        ----------
        return_unlabeled : bool
            the identifier for the subsample

        Returns
        -------
        length : int
            the length of the subsample
        """

        if self.data is None:
            raise RuntimeError("The annotation store data has not been initialized!")
        elif return_unlabeled is None:
            return len(self.data)
        elif return_unlabeled:
            return len(self.unlabeled_indices)
        else:
            return len(self.labeled_indices)

    def get_indices(self, return_unlabeled: bool) -> List:
        """
        Get a list of indices of samples in the labeled/unlabeled subset

        Parameters
        ----------
        return_unlabeled : bool
            the identifier for the subsample (`True` for unlabeled, `False` for labeled, `None` for the
            whole dataset)

        Returns
        -------
        indices : list
            a list of indices that meet the criteria
        """

        return list(range(len(self.data)))

    def count_classes(
        self, perc: bool = False, zeros: bool = False, bouts: bool = False
    ) -> Dict:
        """
        Get a dictionary with class-wise frame counts

        Parameters
        ----------
        perc : bool, default False
            if `True`, a fraction of the total frame count is returned
        zeros : bool, default False
            if `True` and annotation is not exclusive, zero counts are returned
        bouts : bool, default False
            if `True`, instead of frame counts segment counts are returned

        Returns
        -------
        count_dictionary : dict
            a dictionary with class indices as keys and frame counts as values
        """

        if bouts:
            if self.overlap != 0:
                data = {}
                for video, value in self.max_frames.items():
                    for clip, end in value.items():
                        length = end - self._get_min_frame(video, clip)
                        if self.exclusive:
                            data[f"{video}---{clip}"] = -100 * torch.ones(length)
                        else:
                            data[f"{video}---{clip}"] = -100 * torch.ones(
                                (len(self.behaviors_dict()), length)
                            )
                for x, coords in zip(self.data, self.original_coordinates):
                    split = coords[0].split("---")
                    l = self._get_max_frame(split[0], split[1]) - self._get_min_frame(
                        split[0], split[1]
                    )
                    i = coords[1]
                    start = int(i) * self.step
                    end = min(start + self.len_segment, l)
                    data[coords[0]][..., start:end] = x[..., : end - start]
                values = []
                for key, value in data.items():
                    values.append(value)
                    values.append(-100 * torch.ones((*value.shape[:-1], 1)))
                data = torch.cat(values, -1).T
            else:
                data = copy(self.data)
                if self.exclusive:
                    data = data.flatten()
                else:
                    data = data.transpose(1, 2).reshape(-1, len(self.behaviors))
            count_dictionary = {}
            for c in self.behaviors_dict():
                if self.exclusive:
                    arr = data == c
                else:
                    if zeros:
                        arr = data[:, c] == 0
                    else:
                        arr = data[:, c] == 1
                output, indices = torch.unique_consecutive(arr, return_inverse=True)
                true_indices = torch.where(output)[0]
                count_dictionary[c] = len(true_indices)
        else:
            ind = 1
            if zeros:
                ind = 0
            if self.exclusive:
                count_dictionary = dict(Counter(self.data.flatten().cpu().numpy()))
            else:
                d = {}
                for i in range(self.data.shape[1]):
                    cnt = Counter(self.data[:, i, :].flatten().cpu().numpy())
                    d[i] = cnt[ind]
                count_dictionary = d
            if perc:
                total = sum([v for k, v in count_dictionary.items()])
                count_dictionary = {k: v / total for k, v in count_dictionary.items()}
        for i in self.behaviors_dict():
            if i not in count_dictionary:
                count_dictionary[i] = 0
        return count_dictionary

    def behaviors_dict(self) -> Dict:
        """
        Get a dictionary of class names

        Returns
        -------
        behavior_dictionary: dict
            a dictionary with class indices as keys and class names as values
        """

        # if self.behaviors is None
        if self.exclusive and "other" not in self.behaviors:
            d = {i + 1: b for i, b in enumerate(self.behaviors)}
            d[0] = "other"
        else:
            d = {i: b for i, b in enumerate(self.behaviors)}
        return d

    def annotation_class(self) -> str:
        """
        Get the type of annotation ('exclusive_classification', 'nonexclusive_classification')

        Returns
        -------
        annotation_class : str
            the type of annotation
        """

        if self.exclusive:
            return "exclusive_classification"
        else:
            return "nonexclusive_classification"

    def size(self) -> int:
        """
        Get the total number of frames in the data

        Returns
        -------
        size : int
            the total number of frames
        """

        return self.data.shape[0] * self.data.shape[-1]

    def filtered_indices(self) -> List:
        """
        Return the indices of the samples that should be removed

        Choosing the indices can be based on any kind of filering defined in the __init__ function by the data
        parameters

        Returns
        -------
        indices_to_remove : list
            a list of integer indices that should be removed
        """

        return self.filtered

    def set_pseudo_labels(self, labels: torch.Tensor) -> None:
        """
        Set pseudo labels to the unlabeled data

        Parameters
        ----------
        labels : torch.Tensor
            a tensor of pseudo-labels for the unlabeled data
        """

        self.data[self.unlabeled_indices] = labels

    @classmethod
    def get_file_ids(
        cls,
        annotation_path: Union[str, Set],
        annotation_suffix: Union[str, Set],
        *args,
        **kwargs,
    ) -> List:
        """
        Process data parameters and return a list of ids  of the videos that should
        be processed by the __init__ function

        Parameters
        ----------
        annotation_path : str | set
            the path or the set of paths to the folder where the annotation files are stored
        annotation_suffix : str | set, optional
            the suffix or the set of suffices such that the annotation files are named {video_id}{annotation_suffix}

        Returns
        -------
        video_ids : list
            a list of video file ids
        """

        lists = [annotation_path, annotation_suffix]
        for i in range(len(lists)):
            iterable = isinstance(lists[i], Iterable) * (not isinstance(lists[i], str))
            if lists[i] is not None:
                if not iterable:
                    lists[i] = [lists[i]]
                lists[i] = [x for x in lists[i]]
        annotation_path, annotation_suffix = lists
        files = []
        for folder in annotation_path:
            files += [
                strip_suffix(os.path.basename(file), annotation_suffix)
                for file in os.listdir(folder)
                if file.endswith(tuple([x for x in annotation_suffix]))
            ]
        files = sorted(files, key=lambda x: os.path.basename(x))
        return files

    def _set_behaviors(
        self, behaviors: List, ignored_classes: List, behavior_file: str
    ):
        """
        Get a list of behaviors that should be annotated from behavior parameters
        """

        if behaviors is not None:
            for b in ignored_classes:
                if b in behaviors:
                    behaviors.remove(b)
        self.behaviors = behaviors

    def _compute_labeled(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the indices of labeled (annotated) and unlabeled samples
        """

        if self.data is not None and len(self.data) > 0:
            unlabeled = torch.sum(self.data != -100, dim=1) == 0
            labeled_indices = torch.where(~unlabeled)[0]
            unlabeled_indices = torch.where(unlabeled)[0]
        else:
            labeled_indices, unlabeled_indices = torch.tensor([]), torch.tensor([])
        return labeled_indices, unlabeled_indices

    def _generate_annotation(self, times: Dict, name: str) -> Dict:
        """
        Process a loaded annotation file to generate a training labels dictionary
        """

        annotation = {}
        if self.behaviors is None and times is not None:
            behaviors = set()
            for d in times.values():
                behaviors.update([k for k, v in d.items()])
            self.behaviors = [
                x
                for x in sorted(behaviors)
                if x not in self.ignored_classes
                and not x.startswith("negative")
                and not x.startswith("unknown")
            ]
        elif self.behaviors is None and times is None:
            raise ValueError("Cannot generate annotqtion without behavior information!")
        beh_inv = {v: k for k, v in self.behaviors_dict().items()}
        # if there is no annotation file, generate empty annotation
        if self.interactive:
            clips = [
                "+".join(sorted(x))
                for x in combinations(self.max_frames[name].keys(), 2)
            ]
        else:
            clips = list(self.max_frames[name].keys())
        if times is None:
            clips = [x for x in clips if x not in self.ignored_clips]
        # otherwise, apply filters and generate label arrays
        else:
            clips = [
                x
                for x in clips
                if x not in self.ignored_clips and x not in times.keys()
            ]
            for ind in times.keys():
                try:
                    min_frame = self._get_min_frame(name, ind)
                    max_frame = self._get_max_frame(name, ind)
                except KeyError:
                    continue
                go_on = max_frame - min_frame + 1 >= self.frame_limit
                if go_on:
                    v_len = max_frame - min_frame + 1
                    if self.exclusive:
                        if not self.filter_background:
                            value = beh_inv.get("other", 0)
                            labels = np.ones(v_len, dtype=np.compat.long) * value
                        else:
                            labels = -100 * np.ones(v_len, dtype=np.compat.long)
                    else:
                        labels = np.zeros((len(self.behaviors), v_len), dtype=np.float)
                    cat_new = []
                    for cat in times[ind].keys():
                        if cat.startswith("unknown"):
                            cat_new.append(cat)
                    for cat in times[ind].keys():
                        if cat.startswith("negative"):
                            cat_new.append(cat)
                    for cat in times[ind].keys():
                        if not cat.startswith("negative") and not cat.startswith(
                            "unknown"
                        ):
                            cat_new.append(cat)
                    for cat in cat_new:
                        neg = False
                        unknown = False
                        cat_times = times[ind][cat]
                        if self.use_negatives and cat.startswith("negative"):
                            cat = " ".join(cat.split()[1:])
                            neg = True
                        elif cat.startswith("unknown"):
                            cat = " ".join(cat.split()[1:])
                            unknown = True
                        if cat in self.correction:
                            cat = self.correction[cat]
                        for start, end, amb in cat_times:
                            if end > self._get_max_frame(name, ind) + 1:
                                end = self._get_max_frame(name, ind) + 1
                            if amb != 0:
                                continue
                            start -= min_frame
                            end -= min_frame
                            if (
                                self.min_frames_action is not None
                                and end - start < self.min_frames_action
                            ):
                                continue
                            if (
                                self.vis_min_frac > 0
                                and self.vis_min_score > 0
                                and self.visibility is not None
                            ):
                                s = 0
                                for ind_k in ind.split("+"):
                                    s += np.sum(
                                        self.visibility[name][ind_k][start:end]
                                        > self.vis_min_score
                                    )
                                if s < self.vis_min_frac * (end - start) * len(
                                    ind.split("+")
                                ):
                                    continue
                            if cat in beh_inv:
                                cat_i_global = beh_inv[cat]
                                if self.exclusive:
                                    labels[start:end] = cat_i_global
                                else:
                                    if unknown:
                                        labels[cat_i_global, start:end] = -100
                                    elif neg:
                                        labels[cat_i_global, start:end] = 2
                                    else:
                                        labels[cat_i_global, start:end] = 1
                            else:
                                self.not_found.add(cat)
                                if self.filter_background:
                                    if not self.exclusive:
                                        labels[:, start:end][
                                            labels[:, start:end] == 0
                                        ] = 3
                                    else:
                                        labels[start:end][labels[start:end] == -100] = 0

                    if self.error_class is not None and self.error_class in times[ind]:
                        for start, end, amb in times[ind][self.error_class]:
                            if self.exclusive:
                                labels[start:end] = -100
                            else:
                                labels[:, start:end] = -100
                    annotation[os.path.basename(name) + "---" + str(ind)] = labels
        for ind in clips:
            try:
                min_frame = self._get_min_frame(name, ind)
                max_frame = self._get_max_frame(name, ind)
            except KeyError:
                continue
            go_on = max_frame - min_frame + 1 >= self.frame_limit
            if go_on:
                v_len = max_frame - min_frame + 1
                if self.exclusive:
                    annotation[
                        os.path.basename(name) + "---" + str(ind)
                    ] = -100 * np.ones(v_len, dtype=np.compat.long)
                else:
                    annotation[
                        os.path.basename(name) + "---" + str(ind)
                    ] = -100 * np.ones((len(self.behaviors), v_len), dtype=np.float)
        return annotation

    def _make_trimmed_annotations(self, annotations_dict: Dict) -> torch.Tensor:
        """
        Cut a label dictionary into overlapping pieces of equal length
        """

        labels = []
        self.original_coordinates = []
        masked_all = []
        for v_id in sorted(annotations_dict.keys()):
            if v_id in annotations_dict:
                annotations = annotations_dict[v_id]
            else:
                raise ValueError(
                    f'The id list in {v_id.split("---")[0]} is not consistent across files'
                )
            split = v_id.split("---")
            if len(split) > 1:
                video_id, ind = split
            else:
                video_id = split[0]
                ind = ""
            min_frame = self._get_min_frame(video_id, ind)
            max_frame = self._get_max_frame(video_id, ind)
            v_len = max_frame - min_frame + 1
            sp = np.arange(0, v_len, self.step)
            pad = sp[-1] + self.len_segment - v_len
            if self.exclusive:
                annotations = np.pad(annotations, ((0, pad)), constant_values=-100)
            else:
                annotations = np.pad(
                    annotations, ((0, 0), (0, pad)), constant_values=-100
                )
            masked = np.zeros(annotations.shape)
            if (
                self.mask is not None
                and video_id in self.mask["masked"]
                and ind in self.mask["masked"][video_id]
            ):
                for start, end in self.mask["masked"][video_id][ind]:
                    masked[..., int(start) : int(end)] = 1
            for i, start in enumerate(sp):
                self.original_coordinates.append((v_id, i))
                if self.exclusive:
                    ann = annotations[start : start + self.len_segment]
                    m = masked[start : start + self.len_segment]
                else:
                    ann = annotations[:, start : start + self.len_segment]
                    m = masked[:, start : start + self.len_segment]
                labels.append(ann)
                masked_all.append(m)
        self.original_coordinates = np.array(self.original_coordinates)
        labels = torch.tensor(np.array(labels))
        masked_all = torch.tensor(np.array(masked_all)).int().bool()
        if self.filter_background and not self.exclusive:
            for i, label in enumerate(labels):
                label[:, torch.sum((label == 1) | (label == 3), 0) == 0] = -100
                label[label == 3] = 0
        labels[(labels != -100) & masked_all] = -200
        return labels

    @classmethod
    def _get_file_paths(cls, annotation_path: Union[str, Set]) -> List:
        """
        Get a list of relevant files
        """

        file_paths = []
        if annotation_path is not None:
            if isinstance(annotation_path, str):
                annotation_path = [annotation_path]
            for folder in annotation_path:
                file_paths += [os.path.join(folder, x) for x in os.listdir(folder)]
        return file_paths

    def _get_max_frame(self, video_id: str, clip_id: str):
        """
        Get the end frame of a clip in a video
        """

        if clip_id in self.max_frames[video_id]:
            return self.max_frames[video_id][clip_id]
        else:
            return min(
                [self.max_frames[video_id][ind_k] for ind_k in clip_id.split("+")]
            )

    def _get_min_frame(self, video_id, clip_id):
        """
        Get the start frame of a clip in a video
        """

        if clip_id in self.min_frames[video_id]:
            return self.min_frames[video_id][clip_id]
        else:
            return max(
                [self.min_frames[video_id][ind_k] for ind_k in clip_id.split("+")]
            )

    @abstractmethod
    def _load_data(self) -> torch.Tensor:
        """
        Load behavior annotation and generate annotation prompts
        """


class FileAnnotationStore(ActionSegmentationStore):  # +
    """
    A generalized implementation of `ActionSegmentationStore` for datasets where one file corresponds to one video
    """

    def _generate_max_min_frames(self, times: Dict, video_id: str) -> None:
        """
        Generate `max_frames` and `min_frames` objects in case they were not passed from an `InputStore`
        """

        if video_id in self.max_frames:
            return
        for ind, cat_dict in times.items():
            maxes = []
            mins = []
            for cat, cat_list in cat_dict.items():
                if len(cat_list) > 0:
                    maxes.append(max([x[1] for x in cat_list]))
                    mins.append(min([x[0] for x in cat_list]))
            self.max_frames[video_id][ind] = max(maxes)
            self.min_frames[video_id][ind] = min(mins)

    def _load_data(self) -> torch.Tensor:
        """
        Load behavior annotation and generate annotation prompts
        """

        if self.video_order is None:
            return None

        files = []
        for x in self.video_order:
            ok = False
            for folder in self.annotation_folder:
                for s in self.ann_suffix:
                    file = os.path.join(folder, x + s)
                    if os.path.exists(file):
                        files.append(file)
                        ok = True
                        break
            if not ok:
                files.append(None)
        self.not_found = set()
        annotations_dict = {}
        print("Computing annotation arrays...")
        for name, filename in tqdm(list(zip(self.video_order, files))):
            if filename is not None:
                times = self._open_annotations(filename)
            else:
                times = None
            if times is not None:
                self._generate_max_min_frames(times, name)
            annotations_dict.update(self._generate_annotation(times, name))
            del times
        annotation = self._make_trimmed_annotations(annotations_dict)
        del annotations_dict
        if self.filter_annotated:
            if self.exclusive:
                s = torch.sum((annotation != -100), dim=1)
            else:
                s = torch.sum(
                    torch.sum((annotation != -100), dim=1) == annotation.shape[1], dim=1
                )
            self.filtered += torch.where(s == 0)[0].tolist()
        annotation[annotation == -200] = -100
        return annotation

    @abstractmethod
    def _open_annotations(self, filename: str) -> Dict:
        """
        Load the annotation from filename

        Parameters
        ----------
        filename : str
            path to an annotation file

        Returns
        -------
        times : dict
            a nested dictionary where first-level keys are clip ids, second-level keys are categories and values are
            lists of (start, end, ambiguity status) lists
        """


class SequenceAnnotationStore(ActionSegmentationStore):  # +
    """
    A generalized implementation of `ActionSegmentationStore` for datasets where one file corresponds to multiple videos
    """

    def _generate_max_min_frames(self, times: Dict) -> None:
        """
        Generate `max_frames` and `min_frames` objects in case they were not passed from an `InputStore`
        """

        for video_id in times:
            if video_id in self.max_frames:
                continue
            self.max_frames[video_id] = {}
            for ind, cat_dict in times[video_id].items():
                maxes = []
                mins = []
                for cat, cat_list in cat_dict.items():
                    maxes.append(max([x[1] for x in cat_list]))
                    mins.append(min([x[0] for x in cat_list]))
                self.max_frames[video_id][ind] = max(maxes)
                self.min_frames[video_id][ind] = min(mins)

    @classmethod
    def get_file_ids(
        cls,
        filenames: List = None,
        annotation_path: str = None,
        *args,
        **kwargs,
    ) -> List:
        """
        Process data parameters and return a list of ids  of the videos that should
        be processed by the __init__ function

        Parameters
        ----------
        filenames : list, optional
            a list of annotation file paths
        annotation_path : str, optional
            path to the annotation folder

        Returns
        -------
        video_ids : list
            a list of video file ids
        """

        file_paths = []
        if annotation_path is not None:
            if isinstance(annotation_path, str):
                annotation_path = [annotation_path]
            file_paths = []
            for folder in annotation_path:
                file_paths += [os.path.join(folder, x) for x in os.listdir(folder)]
        ids = set()
        for f in file_paths:
            if os.path.basename(f) in filenames:
                ids.add(os.path.basename(f))
        ids = sorted(ids)
        return ids

    def _load_data(self) -> torch.Tensor:
        """
        Load behavior annotation and generate annotation prompts
        """

        if self.video_order is None:
            return None

        files = []
        for f in self.file_paths:
            if os.path.basename(f) in self.video_order:
                files.append(f)
        self.not_found = set()
        annotations_dict = {}
        for name, filename in tqdm(zip(self.video_order, files)):
            if filename is not None:
                times = self._open_sequences(filename)
            else:
                times = None
            if times is not None:
                self._generate_max_min_frames(times)
                none_ids = []
                for video_id, sequence_dict in times.items():
                    if sequence_dict is None:
                        none_ids.append(sequence_dict)
                        continue
                    annotations_dict.update(
                        self._generate_annotation(sequence_dict, video_id)
                    )
                for video_id in none_ids:
                    annotations_dict.update(self._generate_annotation(None, video_id))
                del times
        annotation = self._make_trimmed_annotations(annotations_dict)
        del annotations_dict
        if self.filter_annotated:
            if self.exclusive:
                s = torch.sum((annotation != -100), dim=1)
            else:
                s = torch.sum(
                    torch.sum((annotation != -100), dim=1) == annotation.shape[1], dim=1
                )
            self.filtered += torch.where(s == 0)[0].tolist()
        annotation[annotation == -200] = -100
        return annotation

    @abstractmethod
    def _open_sequences(self, filename: str) -> Dict:
        """
        Load the annotation from filename

        Parameters
        ----------
        filename : str
            path to an annotation file

        Returns
        -------
        times : dict
            a nested dictionary where first-level keys are video ids, second-level keys are clip ids,
            third-level keys are categories and values are
            lists of (start, end, ambiguity status) lists
        """


class DLCAnnotationStore(FileAnnotationStore):  # +
    """
    DLC type annotation data

    The files are either the DLC2Action GUI output or a pickled dictionary of the following structure:
        - nested dictionary,
        - first-level keys are individual IDs,
        - second-level keys are labels,
        - values are lists of intervals,
        - the lists of intervals is formatted as `[start_frame, end_frame, ambiguity]`,
        - ambiguity is 1 if the action is ambiguous (!!at the moment DLC2Action will IGNORE those intervals!!) or 0 if it isn't.

    A minimum working example of such a dictionary is:
    ```
    {
        "ind0": {},
        "ind1": {
            "running": [60, 70, 0]],
            "eating": []
        }
    }
    ```

    Here there are two animals: `"ind0"` and `"ind1"`, and two actions: running and eating.
    The only annotated action is eating for `"ind1"` between frames 60 and 70.

    If you generate those files manually, run this code for a sanity check:
    ```
    import pickle

    with open("/path/to/annotation.pickle", "rb") as f:
    data = pickle.load(f)

    for ind, ind_dict in data.items():
        print(f'individual {ind}:')
        for label, intervals in ind_dict.items():
            for start, end, ambiguity in intervals:
                if ambiguity == 0:
                    print(f'  from {start} to {end} frame: {label}')
    ```

    Assumes the following file structure:
    ```
    annotation_path
    ├── video1_annotation.pickle
    └── video2_labels.pickle
    ```
    Here `annotation_suffix` is `{'_annotation.pickle', '_labels.pickle'}`.
    """

    def _open_annotations(self, filename: str) -> Dict:
        """
        Load the annotation from `filename`
        """

        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                annotation = data
                for ind in annotation:
                    for cat, cat_list in annotation[ind].items():
                        annotation[ind][cat] = [
                            [start, end, 0] for start, end in cat_list
                        ]
            else:
                _, loaded_labels, animals, loaded_times = data
                annotation = {}
                for ind, ind_list in zip(animals, loaded_times):
                    annotation[ind] = {}
                    for cat, cat_list in zip(loaded_labels, ind_list):
                        annotation[ind][cat] = cat_list
            return annotation
        except:
            print(f"{filename} is invalid or does not exist")
            return None


class BorisAnnotationStore(FileAnnotationStore):  # +
    """
    BORIS type annotation data

    Assumes the following file structure:
    ```
    annotation_path
    ├── video1_annotation.pickle
    └── video2_labels.pickle
    ```
    Here `annotation_suffix` is `{'_annotation.pickle', '_labels.pickle'}`.
    """

    def __init__(
        self,
        video_order: List = None,
        min_frames: Dict = None,
        max_frames: Dict = None,
        visibility: Dict = None,
        exclusive: bool = True,
        len_segment: int = 128,
        overlap: int = 0,
        behaviors: Set = None,
        ignored_classes: Set = None,
        annotation_suffix: Union[Set, str] = None,
        annotation_path: Union[Set, str] = None,
        behavior_file: str = None,
        correction: Dict = None,
        frame_limit: int = 0,
        filter_annotated: bool = False,
        filter_background: bool = False,
        error_class: str = None,
        min_frames_action: int = None,
        key_objects: Tuple = None,
        visibility_min_score: float = 0.2,
        visibility_min_frac: float = 0.7,
        mask: Dict = None,
        use_hard_negatives: bool = False,
        default_agent_name: str = "ind0",
        interactive: bool = False,
        ignored_clips: Set = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        video_order : list, optional
            a list of video ids that should be processed in the same order (not passed if creating from key objects)
        min_frames : dict, optional
            a nested dictionary where first-level keys are video ids, second-level keys are clip ids and values are
            clip start frames (not passed if creating from key objects)
        max_frames : dict, optional
            a nested dictionary where first-level keys are video ids, second-level keys are clip ids and values are
            clip end frames (not passed if creating from key objects)
        visibility : dict, optional
            a nested dictionary where first-level keys are video ids, second-level keys are clip ids and values are
            visibility score arrays (not passed if creating from key objects or if irrelevant for the dataset)
        exclusive : bool, default True
            if True, the annotation is single-label; if False, multi-label
        len_segment : int, default 128
            the length of the segments in which the data should be cut (in frames)
        overlap : int, default 0
            the length of the overlap between neighboring segments (in frames)
        behaviors : set, optional
            the list of behaviors to put in the annotation (not passed if creating a blank instance or if behaviors are
            loaded from a file)
        ignored_classes : set, optional
            the list of behaviors from the behaviors list or file to not annotate
        annotation_suffix : str | set, optional
            the suffix or the set of suffices such that the annotation files are named {video_id}{annotation_suffix}
            (not passed if creating from key objects or if irrelevant for the dataset)
        annotation_path : str | set, optional
            the path or the set of paths to the folder where the annotation files are stored (not passed if creating
            from key objects)
        behavior_file : str, optional
            the path to an .xlsx behavior file (not passed if creating from key objects or if irrelevant for the dataset)
        correction : dict, optional
            a dictionary of corrections for the labels (e.g. {'sleping': 'sleeping', 'calm locomotion': 'locomotion'},
            can be used to correct for variations in naming or to merge several labels in one
        frame_limit : int, default 0
            the smallest possible length of a clip (shorter clips are discarded)
        filter_annotated : bool, default False
            if True, the samples that do not have any labels will be filtered
        filter_background : bool, default False
            if True, only the unlabeled frames that are close to annotated frames will be labeled as background
        error_class : str, optional
            the name of the error class (the annotations that intersect with this label will be discarded)
        min_frames_action : int, default 0
            the minimum length of an action (shorter actions are not annotated)
        key_objects : tuple, optional
            the key objects to load the AnnotationStore from
        visibility_min_score : float, default 5
            the minimum visibility score for visibility filtering
        visibility_min_frac : float, default 0.7
            the minimum fraction of visible frames for visibility filtering
        mask : dict, optional
            a masked value dictionary (for active learning simulation experiments)
        use_hard_negatives : bool, default False
            mark hard negatives as 2 instead of 0 or 1, for loss functions that have options for hard negative processing
        interactive : bool, default False
            if `True`, annotation is assigned to pairs of individuals
        ignored_clips : set, optional
            a set of clip ids to ignore
        """

        self.default_agent_name = default_agent_name
        super().__init__(
            video_order=video_order,
            min_frames=min_frames,
            max_frames=max_frames,
            visibility=visibility,
            exclusive=exclusive,
            len_segment=len_segment,
            overlap=overlap,
            behaviors=behaviors,
            ignored_classes=ignored_classes,
            annotation_suffix=annotation_suffix,
            annotation_path=annotation_path,
            behavior_file=behavior_file,
            correction=correction,
            frame_limit=frame_limit,
            filter_annotated=filter_annotated,
            filter_background=filter_background,
            error_class=error_class,
            min_frames_action=min_frames_action,
            key_objects=key_objects,
            visibility_min_score=visibility_min_score,
            visibility_min_frac=visibility_min_frac,
            mask=mask,
            use_hard_negatives=use_hard_negatives,
            interactive=interactive,
            ignored_clips=ignored_clips,
        )

    def _open_annotations(self, filename: str) -> Dict:
        """
        Load the annotation from filename
        """

        try:
            df = pd.read_csv(filename, header=15)
            fps = df.iloc[0]["FPS"]
            df["Subject"] = df["Subject"].fillna(self.default_agent_name)
            loaded_labels = list(df["Behavior"].unique())
            animals = list(df["Subject"].unique())
            loaded_times = {}
            for ind in animals:
                loaded_times[ind] = {}
                agent_df = df[df["Subject"] == ind]
                for cat in loaded_labels:
                    filtered_df = agent_df[agent_df["Behavior"] == cat]
                    starts = (
                        filtered_df["Time"][filtered_df["Status"] == "START"] * fps
                    ).astype(int)
                    ends = (
                        filtered_df["Time"][filtered_df["Status"] == "STOP"] * fps
                    ).astype(int)
                    loaded_times[ind][cat] = [
                        [start, end, 0] for start, end in zip(starts, ends)
                    ]
            return loaded_times
        except:
            print(f"{filename} is invalid or does not exist")
            return None


class PKUMMDAnnotationStore(FileAnnotationStore):  # +
    """
    PKU-MMD annotation data

    Assumes the following file structure:
    ```
    annotation_path
    ├── 0364-L.txt
    ...
    └── 0144-M.txt
    ```
    """

    def __init__(
        self,
        video_order: List = None,
        min_frames: Dict = None,
        max_frames: Dict = None,
        visibility: Dict = None,
        exclusive: bool = True,
        len_segment: int = 128,
        overlap: int = 0,
        ignored_classes: Set = None,
        annotation_path: Union[Set, str] = None,
        behavior_file: str = None,
        correction: Dict = None,
        frame_limit: int = 0,
        filter_annotated: bool = False,
        filter_background: bool = False,
        error_class: str = None,
        min_frames_action: int = None,
        key_objects: Tuple = None,
        visibility_min_score: float = 0,
        visibility_min_frac: float = 0,
        mask: Dict = None,
        use_hard_negatives: bool = False,
        interactive: bool = False,
        ignored_clips: Set = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        video_order : list, optional
            a list of video ids that should be processed in the same order (not passed if creating from key objects)
        min_frames : dict, optional
            a nested dictionary where first-level keys are video ids, second-level keys are clip ids and values are
            clip start frames (not passed if creating from key objects)
        max_frames : dict, optional
            a nested dictionary where first-level keys are video ids, second-level keys are clip ids and values are
            clip end frames (not passed if creating from key objects)
        visibility : dict, optional
            a nested dictionary where first-level keys are video ids, second-level keys are clip ids and values are
            visibility score arrays (not passed if creating from key objects or if irrelevant for the dataset)
        exclusive : bool, default True
            if True, the annotation is single-label; if False, multi-label
        len_segment : int, default 128
            the length of the segments in which the data should be cut (in frames)
        overlap : int, default 0
            the length of the overlap between neighboring segments (in frames)
        ignored_classes : set, optional
            the list of behaviors from the behaviors list or file to not annotate
        annotation_path : str | set, optional
            the path or the set of paths to the folder where the annotation files are stored (not passed if creating
            from key objects)
        behavior_file : str, optional
            the path to an .xlsx behavior file (not passed if creating from key objects or if irrelevant for the dataset)
        correction : dict, optional
            a dictionary of corrections for the labels (e.g. {'sleping': 'sleeping', 'calm locomotion': 'locomotion'},
            can be used to correct for variations in naming or to merge several labels in one
        frame_limit : int, default 0
            the smallest possible length of a clip (shorter clips are discarded)
        filter_annotated : bool, default False
            if True, the samples that do not have any labels will be filtered
        filter_background : bool, default False
            if True, only the unlabeled frames that are close to annotated frames will be labeled as background
        error_class : str, optional
            the name of the error class (the annotations that intersect with this label will be discarded)
        min_frames_action : int, default 0
            the minimum length of an action (shorter actions are not annotated)
        key_objects : tuple, optional
            the key objects to load the AnnotationStore from
        visibility_min_score : float, default 5
            the minimum visibility score for visibility filtering
        visibility_min_frac : float, default 0.7
            the minimum fraction of visible frames for visibility filtering
        mask : dict, optional
            a masked value dictionary (for active learning simulation experiments)
        use_hard_negatives : bool, default False
            mark hard negatives as 2 instead of 0 or 1, for loss functions that have options for hard negative processing
        interactive : bool, default False
            if `True`, annotation is assigned to pairs of individuals
        ignored_clips : set, optional
            a set of clip ids to ignore
        """

        super().__init__(
            video_order=video_order,
            min_frames=min_frames,
            max_frames=max_frames,
            visibility=visibility,
            exclusive=exclusive,
            len_segment=len_segment,
            overlap=overlap,
            behaviors=None,
            ignored_classes=ignored_classes,
            annotation_suffix={".txt"},
            annotation_path=annotation_path,
            behavior_file=behavior_file,
            correction=correction,
            frame_limit=frame_limit,
            filter_annotated=filter_annotated,
            filter_background=filter_background,
            error_class=error_class,
            min_frames_action=min_frames_action,
            key_objects=key_objects,
            visibility_min_score=visibility_min_score,
            visibility_min_frac=visibility_min_frac,
            mask=mask,
            use_hard_negatives=use_hard_negatives,
            interactive=interactive,
            ignored_clips=ignored_clips,
            *args,
            **kwargs,
        )

    @classmethod
    def get_file_ids(cls, annotation_path: Union[str, Set], *args, **kwargs) -> List:
        """
        Process data parameters and return a list of ids  of the videos that should
        be processed by the __init__ function

        Parameters
        ----------
        annotation_path : str | set
            the path or the set of paths to the folder where the annotation files are stored

        Returns
        -------
        video_ids : list
            a list of video file ids
        """

        if isinstance(annotation_path, str):
            annotation_path = [annotation_path]
        files = []
        for folder in annotation_path:
            files += [
                os.path.basename(x)[:-4] for x in os.listdir(folder) if x[-4:] == ".txt"
            ]
        files = sorted(files, key=lambda x: os.path.basename(x))
        return files

    def _open_annotations(self, filename: str) -> Dict:
        """
        Load the annotation from filename
        """

        if self.interactive:
            agent_name = "0+1"
        else:
            agent_name = "0"
        times = {agent_name: defaultdict(lambda: [])}
        with open(filename) as f:
            for line in f.readlines():
                label, start, end, *_ = map(int, line.split(","))
                times[agent_name][self.all_behaviors[int(label) - 1]].append(
                    [start, end, 0]
                )
        return times

    def _set_behaviors(
        self, behaviors: List, ignored_classes: List, behavior_file: str
    ):
        """
        Get a list of behaviors that should be annotated from behavior parameters
        """

        if behavior_file is not None:
            behaviors = list(pd.read_excel(behavior_file)["Action"])
        self.all_behaviors = copy(behaviors)
        for b in ignored_classes:
            if b in behaviors:
                behaviors.remove(b)
        self.behaviors = behaviors


class CalMS21AnnotationStore(SequenceAnnotationStore):  # +
    """
    CalMS21 annotation data

    Use the `'random:test_from_name:{name}'` and `'val-from-name:{val_name}:test-from-name:{test_name}'`
    partitioning methods with `'train'`, `'test'` and `'unlabeled'` names to separate into train, test and validation
    subsets according to the original files. For example, with `'val-from-name:test:test-from-name:unlabeled'`
    the data from the test file will go into validation and the unlabeled files will be the test.

    Assumes the following file structure:
    ```
    annotation_path
    ├── calms21_task_train.npy
    ├── calms21_task_test.npy
    ├── calms21_unlabeled_videos_part1.npy
    ├── calms21_unlabeled_videos_part2.npy
    └── calms21_unlabeled_videos_part3.npy
    ```
    """

    def __init__(
        self,
        task_n: int = 1,
        include_task1: bool = True,
        video_order: List = None,
        min_frames: Dict = None,
        max_frames: Dict = None,
        len_segment: int = 128,
        overlap: int = 0,
        ignored_classes: Set = None,
        annotation_path: Union[Set, str] = None,
        key_objects: Tuple = None,
        treba_files: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """

        Parameters
        ----------
        task_n : [1, 2]
            the number of the task
        include_task1 : bool, default True
            include task 1 data to training set
        video_order : list, optional
            a list of video ids that should be processed in the same order (not passed if creating from key objects)
        min_frames : dict, optional
            a nested dictionary where first-level keys are video ids, second-level keys are clip ids and values are
            clip start frames (not passed if creating from key objects)
        max_frames : dict, optional
            a nested dictionary where first-level keys are video ids, second-level keys are clip ids and values are
            clip end frames (not passed if creating from key objects)
        len_segment : int, default 128
            the length of the segments in which the data should be cut (in frames)
        overlap : int, default 0
            the length of the overlap between neighboring segments (in frames)
        ignored_classes : set, optional
            the list of behaviors from the behaviors list or file to not annotate
        annotation_path : str | set, optional
            the path or the set of paths to the folder where the annotation files are stored (not passed if creating
            from key objects)
        key_objects : tuple, optional
            the key objects to load the AnnotationStore from
        treba_files : bool, default False
            if `True`, TREBA feature files will be loaded
        """

        self.task_n = int(task_n)
        self.include_task1 = include_task1
        if self.task_n == 1:
            self.include_task1 = False
        self.treba_files = treba_files
        if "exclusive" in kwargs:
            exclusive = kwargs["exclusive"]
        else:
            exclusive = True
        if "behaviors" in kwargs and kwargs["behaviors"] is not None:
            behaviors = kwargs["behaviors"]
        else:
            behaviors = ["attack", "investigation", "mount", "other"]
            if task_n == 3:
                exclusive = False
                behaviors += [
                    "approach",
                    "disengaged",
                    "groom",
                    "intromission",
                    "mount_attempt",
                    "sniff_face",
                    "whiterearing",
                ]
        super().__init__(
            video_order=video_order,
            min_frames=min_frames,
            max_frames=max_frames,
            exclusive=exclusive,
            len_segment=len_segment,
            overlap=overlap,
            behaviors=behaviors,
            ignored_classes=ignored_classes,
            annotation_path=annotation_path,
            key_objects=key_objects,
            filter_annotated=False,
            interactive=True,
        )

    @classmethod
    def get_file_ids(
        cls,
        task_n: int = 1,
        include_task1: bool = False,
        treba_files: bool = False,
        annotation_path: Union[str, Set] = None,
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
        annotation_path : str | set, optional
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
        filenames = set(files)
        return SequenceAnnotationStore.get_file_ids(
            filenames, annotation_path=annotation_path
        )

    def _open_sequences(self, filename: str) -> Dict:
        """
        Load the annotation from filename

        Parameters
        ----------
        filename : str
            path to an annotation file

        Returns
        -------
        times : dict
            a nested dictionary where first-level keys are video ids, second-level keys are clip ids,
            third-level keys are categories and values are
            lists of (start, end, ambiguity status) lists
        """

        data_dict = np.load(filename, allow_pickle=True).item()
        data = {}
        result = {}
        keys = list(data_dict.keys())
        if "test" in os.path.basename(filename):
            mode = "test"
        elif "unlabeled" in os.path.basename(filename):
            mode = "unlabeled"
        else:
            mode = "train"
        if "approach" in keys:
            for behavior in keys:
                for key in data_dict[behavior].keys():
                    ann = data_dict[behavior][key]["annotations"]
                    result[f'{mode}--{key.split("/")[-1]}'] = {
                        "mouse1+mouse2": defaultdict(lambda: [])
                    }
                    starts = np.where(
                        np.diff(np.concatenate([np.array([0]), ann, np.array([0])]))
                        == 1
                    )[0]
                    ends = np.where(
                        np.diff(np.concatenate([np.array([0]), ann, np.array([0])]))
                        == -1
                    )[0]
                    for start, end in zip(starts, ends):
                        result[f'{mode}--{key.split("/")[-1]}']["mouse1+mouse2"][
                            behavior
                        ].append([start, end, 0])
                    for b in self.behaviors:
                        result[f'{mode}--{key.split("/")[-1]}---mouse1+mouse2'][
                            "mouse1+mouse2"
                        ][f"unknown {b}"].append([0, len(ann), 0])
        for key in keys:
            data.update(data_dict[key])
            data_dict.pop(key)
        if "approach" not in keys and self.task_n == 3:
            for key in data.keys():
                result[f'{mode}--{key.split("/")[-1]}'] = {"mouse1+mouse2": {}}
                ann = data[key]["annotations"]
                for i in range(4):
                    starts = np.where(
                        np.diff(
                            np.concatenate(
                                [np.array([0]), (ann == i).astype(int), np.array([0])]
                            )
                        )
                        == 1
                    )[0]
                    ends = np.where(
                        np.diff(
                            np.concatenate(
                                [np.array([0]), (ann == i).astype(int), np.array([0])]
                            )
                        )
                        == -1
                    )[0]
                    result[f'{mode}--{key.split("/")[-1]}']["mouse1+mouse2"][
                        self.behaviors_dict()[i]
                    ] = [[start, end, 0] for start, end in zip(starts, ends)]
        if self.task_n != 3:
            for seq_name, seq_dict in data.items():
                if "annotations" not in seq_dict:
                    return None
                behaviors = np.unique(seq_dict["annotations"])
                ann = seq_dict["annotations"]
                key = f'{mode}--{seq_name.split("/")[-1]}'
                result[key] = {"mouse1+mouse2": {}}
                for i in behaviors:
                    starts = np.where(
                        np.diff(
                            np.concatenate(
                                [np.array([0]), (ann == i).astype(int), np.array([0])]
                            )
                        )
                        == 1
                    )[0]
                    ends = np.where(
                        np.diff(
                            np.concatenate(
                                [np.array([0]), (ann == i).astype(int), np.array([0])]
                            )
                        )
                        == -1
                    )[0]
                    result[key]["mouse1+mouse2"][self.behaviors_dict()[i]] = [
                        [start, end, 0] for start, end in zip(starts, ends)
                    ]
        return result


class CSVAnnotationStore(FileAnnotationStore):  # +
    """
    CSV type annotation data

    Assumes that files are saved as .csv tables with at least the following columns:
    - from / start : start of action,
    - to / end : end of action,
    - class / behavior / behaviour / label / type : action label.

    If the times are set in seconds instead of frames, don't forget to set the `fps` parameter to your frame rate.

    Assumes the following file structure:
    ```
    annotation_path
    ├── video1_annotation.csv
    └── video2_labels.csv
    ```
    Here `annotation_suffix` is `{'_annotation.csv', '_labels.csv'}`.
    """

    def __init__(
        self,
        video_order: List = None,
        min_frames: Dict = None,
        max_frames: Dict = None,
        visibility: Dict = None,
        exclusive: bool = True,
        len_segment: int = 128,
        overlap: int = 0,
        behaviors: Set = None,
        ignored_classes: Set = None,
        annotation_suffix: Union[Set, str] = None,
        annotation_path: Union[Set, str] = None,
        behavior_file: str = None,
        correction: Dict = None,
        frame_limit: int = 0,
        filter_annotated: bool = False,
        filter_background: bool = False,
        error_class: str = None,
        min_frames_action: int = None,
        key_objects: Tuple = None,
        visibility_min_score: float = 0.2,
        visibility_min_frac: float = 0.7,
        mask: Dict = None,
        default_agent_name: str = "ind0",
        separator: str = ",",
        fps: int = 30,
        *args,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        video_order : list, optional
            a list of video ids that should be processed in the same order (not passed if creating from key objects)
        min_frames : dict, optional
            a nested dictionary where first-level keys are video ids, second-level keys are clip ids and values are
            clip start frames (not passed if creating from key objects)
        max_frames : dict, optional
            a nested dictionary where first-level keys are video ids, second-level keys are clip ids and values are
            clip end frames (not passed if creating from key objects)
        visibility : dict, optional
            a nested dictionary where first-level keys are video ids, second-level keys are clip ids and values are
            visibility score arrays (not passed if creating from key objects or if irrelevant for the dataset)
        exclusive : bool, default True
            if True, the annotation is single-label; if False, multi-label
        len_segment : int, default 128
            the length of the segments in which the data should be cut (in frames)
        overlap : int, default 0
            the length of the overlap between neighboring segments (in frames)
        behaviors : set, optional
            the list of behaviors to put in the annotation (not passed if creating a blank instance or if behaviors are
            loaded from a file)
        ignored_classes : set, optional
            the list of behaviors from the behaviors list or file to not annotate
        annotation_suffix : str | set, optional
            the suffix or the set of suffices such that the annotation files are named {video_id}{annotation_suffix}
            (not passed if creating from key objects or if irrelevant for the dataset)
        annotation_path : str | set, optional
            the path or the set of paths to the folder where the annotation files are stored (not passed if creating
            from key objects)
        behavior_file : str, optional
            the path to an .xlsx behavior file (not passed if creating from key objects or if irrelevant for the dataset)
        correction : dict, optional
            a dictionary of corrections for the labels (e.g. {'sleping': 'sleeping', 'calm locomotion': 'locomotion'},
            can be used to correct for variations in naming or to merge several labels in one
        frame_limit : int, default 0
            the smallest possible length of a clip (shorter clips are discarded)
        filter_annotated : bool, default False
            if True, the samples that do not have any labels will be filtered
        filter_background : bool, default False
            if True, only the unlabeled frames that are close to annotated frames will be labeled as background
        error_class : str, optional
            the name of the error class (the annotations that intersect with this label will be discarded)
        min_frames_action : int, default 0
            the minimum length of an action (shorter actions are not annotated)
        key_objects : tuple, optional
            the key objects to load the AnnotationStore from
        visibility_min_score : float, default 5
            the minimum visibility score for visibility filtering
        visibility_min_frac : float, default 0.7
            the minimum fraction of visible frames for visibility filtering
        mask : dict, optional
            a masked value dictionary (for active learning simulation experiments)
        default_agent_name : str, default "ind0"
            the clip id to use when there is no given
        separator : str, default ","
            the separator in the csv files
        fps : int, default 30
            frames per second in the videos
        """

        self.default_agent_name = default_agent_name
        self.separator = separator
        self.fps = fps
        super().__init__(
            video_order=video_order,
            min_frames=min_frames,
            max_frames=max_frames,
            visibility=visibility,
            exclusive=exclusive,
            len_segment=len_segment,
            overlap=overlap,
            behaviors=behaviors,
            ignored_classes=ignored_classes,
            ignored_clips=None,
            annotation_suffix=annotation_suffix,
            annotation_path=annotation_path,
            behavior_file=behavior_file,
            correction=correction,
            frame_limit=frame_limit,
            filter_annotated=filter_annotated,
            filter_background=filter_background,
            error_class=error_class,
            min_frames_action=min_frames_action,
            key_objects=key_objects,
            visibility_min_score=visibility_min_score,
            visibility_min_frac=visibility_min_frac,
            mask=mask,
        )

    def _open_annotations(self, filename: str) -> Dict:
        """
        Load the annotation from `filename`
        """

        # try:
        data = pd.read_csv(filename, sep=self.separator)
        data.columns = list(map(lambda x: x.lower(), data.columns))
        starts, ends, actions = None, None, None
        start_names = ["from", "start"]
        for x in start_names:
            if x in data.columns:
                starts = data[x]
        end_names = ["to", "end"]
        for x in end_names:
            if x in data.columns:
                ends = data[x]
        class_names = ["class", "behavior", "behaviour", "type", "label"]
        for x in class_names:
            if x in data.columns:
                actions = data[x]
        if starts is None:
            raise ValueError("The file must have a column titled 'from' or 'start'!")
        if ends is None:
            raise ValueError("The file must have a column titled 'to' or 'end'!")
        if actions is None:
            raise ValueError(
                "The file must have a column titled 'class', 'behavior', 'behaviour', 'type' or 'label'!"
            )
        times = defaultdict(lambda: defaultdict(lambda: []))
        for start, end, action in zip(starts, ends, actions):
            if any([np.isnan(x) for x in [start, end]]):
                continue
            times[self.default_agent_name][action].append(
                [int(start * self.fps), int(end * self.fps), 0]
            )
        return times


class SIMBAAnnotationStore(FileAnnotationStore):  # +
    """
    SIMBA paper format data

    Assumes the following file structure:
    ```
    annotation_path
    ├── Video1.csv
    ...
    └── Video9.csv
    """

    def __init__(
        self,
        video_order: List = None,
        min_frames: Dict = None,
        max_frames: Dict = None,
        visibility: Dict = None,
        exclusive: bool = True,
        len_segment: int = 128,
        overlap: int = 0,
        behaviors: Set = None,
        ignored_classes: Set = None,
        ignored_clips: Set = None,
        annotation_path: Union[Set, str] = None,
        correction: Dict = None,
        filter_annotated: bool = False,
        filter_background: bool = False,
        error_class: str = None,
        min_frames_action: int = None,
        key_objects: Tuple = None,
        visibility_min_score: float = 0.2,
        visibility_min_frac: float = 0.7,
        mask: Dict = None,
        use_hard_negatives: bool = False,
        annotation_suffix: str = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        video_order : list, optional
            a list of video ids that should be processed in the same order (not passed if creating from key objects)
        min_frames : dict, optional
            a nested dictionary where first-level keys are video ids, second-level keys are clip ids and values are
            clip start frames (not passed if creating from key objects)
        max_frames : dict, optional
            a nested dictionary where first-level keys are video ids, second-level keys are clip ids and values are
            clip end frames (not passed if creating from key objects)
        visibility : dict, optional
            a nested dictionary where first-level keys are video ids, second-level keys are clip ids and values are
            visibility score arrays (not passed if creating from key objects or if irrelevant for the dataset)
        exclusive : bool, default True
            if True, the annotation is single-label; if False, multi-label
        len_segment : int, default 128
            the length of the segments in which the data should be cut (in frames)
        overlap : int, default 0
            the length of the overlap between neighboring segments (in frames)
        behaviors : set, optional
            the list of behaviors to put in the annotation (not passed if creating a blank instance or if behaviors are
            loaded from a file)
        ignored_classes : set, optional
            the list of behaviors from the behaviors list or file to not annotate
        ignored_clips : set, optional
            clip ids to ignore
        annotation_path : str | set, optional
            the path or the set of paths to the folder where the annotation files are stored (not passed if creating
            from key objects)
        behavior_file : str, optional
            the path to an .xlsx behavior file (not passed if creating from key objects or if irrelevant for the dataset)
        correction : dict, optional
            a dictionary of corrections for the labels (e.g. {'sleping': 'sleeping', 'calm locomotion': 'locomotion'},
            can be used to correct for variations in naming or to merge several labels in one
        filter_annotated : bool, default False
            if True, the samples that do not have any labels will be filtered
        filter_background : bool, default False
            if True, only the unlabeled frames that are close to annotated frames will be labeled as background
        error_class : str, optional
            the name of the error class (the annotations that intersect with this label will be discarded)
        min_frames_action : int, default 0
            the minimum length of an action (shorter actions are not annotated)
        key_objects : tuple, optional
            the key objects to load the AnnotationStore from
        visibility_min_score : float, default 5
            the minimum visibility score for visibility filtering
        visibility_min_frac : float, default 0.7
            the minimum fraction of visible frames for visibility filtering
        mask : dict, optional
            a masked value dictionary (for active learning simulation experiments)
        use_hard_negatives : bool, default False
            mark hard negatives as 2 instead of 0 or 1, for loss functions that have options for hard negative processing
        annotation_suffix : str | set, optional
            the suffix or the set of suffices such that the annotation files are named {video_id}{annotation_suffix}
            (not passed if creating from key objects or if irrelevant for the dataset)
        """

        super().__init__(
            video_order=video_order,
            min_frames=min_frames,
            max_frames=max_frames,
            visibility=visibility,
            exclusive=exclusive,
            len_segment=len_segment,
            overlap=overlap,
            behaviors=behaviors,
            ignored_classes=ignored_classes,
            ignored_clips=ignored_clips,
            annotation_suffix=annotation_suffix,
            annotation_path=annotation_path,
            behavior_file=None,
            correction=correction,
            frame_limit=0,
            filter_annotated=filter_annotated,
            filter_background=filter_background,
            error_class=error_class,
            min_frames_action=min_frames_action,
            key_objects=key_objects,
            visibility_min_score=visibility_min_score,
            visibility_min_frac=visibility_min_frac,
            mask=mask,
            use_hard_negatives=use_hard_negatives,
            interactive=True,
        )

    def _open_annotations(self, filename: str) -> Dict:
        """
        Load the annotation from filename

        Parameters
        ----------
        filename : str
            path to an annotation file

        Returns
        -------
        times : dict
            a nested dictionary where first-level keys are clip ids, second-level keys are categories and values are
            lists of (start, end, ambiguity status) lists
        """

        data = pd.read_csv(filename)
        columns = [x for x in data.columns if x.split("_")[-1] == "x"]
        animals = sorted(set([x.split("_")[-2] for x in columns]))
        if len(animals) > 2:
            raise ValueError(
                "SIMBAAnnotationStore is only implemented for files with 1 or 2 animals!"
            )
        if len(animals) == 1:
            ind = animals[0]
        else:
            ind = "+".join(animals)
        behaviors = [
            "_".join(x.split("_")[:-1])
            for x in data.columns
            if x.split("_")[-1] == "prediction"
        ]
        result = {}
        for behavior in behaviors:
            ann = data[f"{behavior}_prediction"].values
            diff = np.diff(
                np.concatenate([np.array([0]), (ann == 1).astype(int), np.array([0])])
            )
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            result[behavior] = [[start, end, 0] for start, end in zip(starts, ends)]
            diff = np.diff(
                np.concatenate(
                    [np.array([0]), (np.isnan(ann)).astype(int), np.array([0])]
                )
            )
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            result[f"unknown {behavior}"] = [
                [start, end, 0] for start, end in zip(starts, ends)
            ]
        if self.behaviors is not None:
            for behavior in self.behaviors:
                if behavior not in behaviors:
                    result[f"unknown {behavior}"] = [[0, len(data), 0]]
        return {ind: result}
