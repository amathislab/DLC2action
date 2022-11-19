#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
"""
Behavior dataset (class that manages high-level data interactions)
"""

import warnings
from typing import Dict, Union, Tuple, List, Optional, Any

from numpy import ndarray
from torch.utils.data import Dataset
import torch
from abc import ABC
import numpy as np
from copy import copy
from tqdm import tqdm
from collections import Counter
import inspect
from collections import defaultdict
from dlc2action.utils import (
    apply_threshold_hysteresis,
    apply_threshold,
    apply_threshold_max,
)
from dlc2action.data.base_store import InputStore, AnnotationStore
from copy import deepcopy, copy
import os
import pickle
from dlc2action import options


class BehaviorDataset(Dataset, ABC):
    """
    A generalized dataset class

    Data and annotation are stored in separate InputStore and AnnotationStore objects; the dataset class
    manages their interactions.
    """

    def __init__(
        self,
        data_type: str,
        annotation_type: str = "none",
        ssl_transformations: List = None,
        saved_data_path: str = None,
        input_store: InputStore = None,
        annotation_store: AnnotationStore = None,
        only_load_annotated: bool = False,
        recompute_annotation: bool = False,
        # mask: str = None,
        ids: List = None,
        **data_parameters,
    ) -> None:
        """
        Parameters
        ----------
        data_type : str
            the data type (see available types by running BehaviorDataset.data_types())
        annotation_type : str
            the annotation type (see available types by running BehaviorDataset.annotation_types())
        ssl_transformations : list
            a list of functions that take a sample dictionary as input and return an (ssl input, ssl target) tuple
        saved_data_path : str
            the path to a pre-computed pickled dataset
        input_store : InputStore
            a pre-computed input store
        annotation_store : AnnotationStore
            a precomputed annotation store
        only_load_annotated : bool
            if True, the input files that don't have a matching annotation file will be disregarded
        *data_parameters : dict
            parameters to initialize the input and annotation stores
        """

        mask = None
        if len(data_parameters) == 0:
            recompute_annotation = False
        feature_extraction = data_parameters.get("feature_extraction")
        if feature_extraction is not None and not issubclass(
            options.input_stores[data_type],
            options.feature_extractors[feature_extraction].input_store_class,
        ):
            raise ValueError(
                f"The {feature_extraction} feature extractor does not work with "
                f"the {data_type} data type, please choose a suclass of "
                f"{options.feature_extractors[feature_extraction].input_store_class}"
            )
        if ssl_transformations is None:
            ssl_transformations = []
        self.ssl_transformations = ssl_transformations
        self.input_type = data_type
        self.annotation_type = annotation_type
        self.stats = None
        if mask is not None:
            with open(mask, "rb") as f:
                self.mask = pickle.load(f)
        else:
            self.mask = None
        self.ids = ids
        self.tag = None
        self.return_unlabeled = None
        # load saved key objects for annotation and input if they exist
        input_key_objects, annotation_key_objects = None, None
        if saved_data_path is not None:
            if os.path.exists(saved_data_path):
                with open(saved_data_path, "rb") as f:
                    input_key_objects, annotation_key_objects = pickle.load(f)
        # if the input or the annotation store need to be created, generate the common video order
        if len(data_parameters) > 0:
            input_files = options.input_stores[data_type].get_file_ids(
                **data_parameters
            )
            annotation_files = options.annotation_stores[annotation_type].get_file_ids(
                **data_parameters
            )
            if only_load_annotated:
                data_parameters["video_order"] = [
                    x for x in input_files if x in annotation_files
                ]
            else:
                data_parameters["video_order"] = input_files
            if len(data_parameters["video_order"]) == 0:
                raise RuntimeError(
                    "The length of file list is 0! Please check your data parameters!"
                )
        data_parameters["mask"] = self.mask
        # load or create the input store
        ok = False
        if input_store is not None:
            self.input_store = input_store
            ok = True
        elif input_key_objects is not None:
            try:
                self.input_store = self._load_input_store(data_type, input_key_objects)
                ok = True
            except:
                warnings.warn("Loading input store from key objects failed")
        if not ok:
            self.input_store = self._get_input_store(
                data_type, deepcopy(data_parameters)
            )
        # get the objects needed to create the annotation store (like a clip length dictionary)
        annotation_objects = self.input_store.get_annotation_objects()
        data_parameters.update(annotation_objects)
        # load or create the annotation store
        ok = False
        if annotation_store is not None:
            self.annotation_store = annotation_store
            ok = True
        elif (
            annotation_key_objects is not None
            and mask is None
            and not recompute_annotation
        ):
            try:
                self.annotation_store = self._load_annotation_store(
                    annotation_type, annotation_key_objects
                )
                ok = True
            except:
                warnings.warn("Loading annotation store from key objects failed")
        if not ok:
            self.annotation_store = self._get_annotation_store(
                annotation_type, deepcopy(data_parameters)
            )
        if (
            mask is None
            and annotation_type != "none"
            and not recompute_annotation
            and (
                self.annotation_store.get_original_coordinates()
                != self.input_store.get_original_coordinates()
            ).any()
        ):
            raise RuntimeError(
                "The clip orders in the annotation store and input store are different!"
            )
        # filter the data based on data parameters
        # print(f"1 {self.annotation_store.get_original_coordinates().shape=}")
        # print(f"1 {self.input_store.get_original_coordinates().shape=}")
        to_remove = self.annotation_store.filtered_indices()
        if len(to_remove) > 0:
            print(
                f"Filtering {100 * len(to_remove) / len(self.annotation_store):.2f}% of samples"
            )
        if len(self.input_store) == len(self.annotation_store):
            self.input_store.remove(to_remove)
        self.annotation_store.remove(to_remove)
        self.input_indices = list(range(len(self.input_store)))
        self.annotation_indices = list(range(len(self.input_store)))
        self.indices = list(range(len(self.input_store)))
        # print(f'{data_parameters["video_order"]=}')
        # print(f"{self.annotation_store.get_original_coordinates().shape=}")
        # print(f"{self.input_store.get_original_coordinates().shape=}")
        # count = 0
        # for i, (x, y) in enumerate(zip(
        #     self.annotation_store.get_original_coordinates(),
        #     self.input_store.get_original_coordinates(),
        # )):
        #     if (x != y).any():
        #         count += 1
        #         print({i})
        #         print(f"ann: {x}")
        #         print(f"inp: {y}")
        #         print("\n")
        #     if count > 50:
        #         break
        if annotation_type != "none" and (
            self.annotation_store.get_original_coordinates().shape
            != self.input_store.get_original_coordinates().shape
            or (
                self.annotation_store.get_original_coordinates()
                != self.input_store.get_original_coordinates()
            ).any()
        ):
            raise RuntimeError(
                "The clip orders in the annotation store and input store are different!"
            )

    def __getitem__(self, item: int) -> Dict:
        idx = self._get_idx(item)
        input = deepcopy(self.input_store[idx])
        target = self.annotation_store[idx]
        tag = self.input_store.get_tag(idx)
        ssl_inputs, ssl_targets = self._get_SSL_targets(input)
        batch = {"input": input}
        for name, x in zip(
            ["target", "ssl_inputs", "ssl_targets", "tag"],
            [target, ssl_inputs, ssl_targets, tag],
        ):
            if x is not None:
                batch[name] = x
        batch["index"] = idx
        if self.stats is not None:
            for key in batch["input"].keys():
                key_name = key.split("---")[0]
                if key_name in self.stats:
                    batch["input"][key][:, batch["input"][key].sum(0) != 0] = (
                        (batch["input"][key] - self.stats[key_name]["mean"])
                        / (self.stats[key_name]["std"] + 1e-7)
                    )[:, batch["input"][key].sum(0) != 0]
        return batch

    def __len__(self) -> int:
        return len(self.indices)
        # if self.annotation_type != "none":
        #     return self.annotation_store.get_len(return_unlabeled=self.return_unlabeled)
        # else:
        #     return len(self.input_store)

    def get_tags(self) -> List:
        """
        Get a list of all meta tags

        Returns
        -------
        tags: List
            a list of unique meta tag values
        """

        return self.input_store.get_tags()

    def save(self, save_path: str) -> None:
        """
        Save the dictionary

        Parameters
        ----------
        save_path : str
            the path where the pickled file will be stored
        """

        input_obj = self.input_store.key_objects()
        annotation_obj = self.annotation_store.key_objects()
        with open(save_path, "wb") as f:
            pickle.dump((input_obj, annotation_obj), f)

    def to_ram(self) -> None:
        """
        Transfer the dataset to RAM
        """

        self.input_store.to_ram()
        self.annotation_store.to_ram()

    def generate_full_length_gt(self) -> Dict:
        if self.annotation_class() == "exclusive_classification":
            gt = torch.zeros((len(self), self.len_segment()))
        else:
            gt = torch.zeros(
                (len(self), len(self.behaviors_dict()), self.len_segment())
            )
        for i in range(len(self)):
            gt[i] = self.annotation_store[i]
        return self.generate_full_length_prediction(gt)

    def generate_full_length_prediction(self, predicted: torch.Tensor) -> Dict:
        """
        Map predictions for the equal-length pieces to predictions for the original data

        Probabilities are averaged over predictions on overlapping intervals.

        Parameters
        ----------
        predicted: torch.Tensor
            a tensor of predicted probabilities of shape `(N, #classes, #frames)`

        Returns
        -------
        full_length_prediction : dict
            a nested dictionary where first-level keys are video ids, second-level keys are clip ids and values are
            averaged probability tensors
        """

        result = defaultdict(lambda: {})
        counter = defaultdict(lambda: {})
        coordinates = self.input_store.get_original_coordinates()
        for coords, prediction in zip(coordinates, predicted):
            l = self.input_store.get_clip_length_from_coords(coords)
            video_name = self.input_store.get_video_id(coords)
            clip_id = self.input_store.get_clip_id(coords)
            start, end = self.input_store.get_clip_start_end(coords)
            if clip_id not in result[video_name].keys():
                result[video_name][clip_id] = torch.zeros(*prediction.shape[:-1], l)
                counter[video_name][clip_id] = torch.zeros(*prediction.shape[:-1], l)
            result[video_name][clip_id][..., start:end] += (
                prediction.squeeze()[..., : end - start].detach().cpu()
            )
            counter[video_name][clip_id][..., start:end] += 1
        for video_name in result:
            for clip_id in result[video_name]:
                result[video_name][clip_id] /= counter[video_name][clip_id]
                result[video_name][clip_id][counter[video_name][clip_id] == 0] = -100
        result = dict(result)
        return result

    def find_valleys(
        self,
        predicted: Union[torch.Tensor, Dict],
        threshold: float = 0.5,
        min_frames: int = 0,
        visibility_min_score: float = 0,
        visibility_min_frac: float = 0,
        main_class: int = 1,
        low: bool = True,
        predicted_error: torch.Tensor = None,
        error_threshold: float = 0.5,
        hysteresis: bool = False,
        threshold_diff: float = None,
        min_frames_error: int = None,
        smooth_interval: int = 1,
        cut_annotated: bool = False,
    ) -> Dict:
        """
        Find the intervals where the probability of a certain class is below or above a certain hard_threshold

        Parameters
        ----------
        predicted : torch.Tensor | dict
            either a tensor of predictions for the data prompts or the output of
            `BehaviorDataset.generate_full_length_prediction`
        threshold : float, default 0.5
            the main hard_threshold
        min_frames : int, default 0
            the minimum length of the intervals
        visibility_min_score : float, default 0
            the minimum visibility score in the intervals
        visibility_min_frac : float, default 0
            fraction of the interval that has to have the visibility score larger than visibility_score_thr
        main_class : int, default 1
            the index of the class the function is inspecting
        low : bool, default True
            if True, the probability in the intervals has to be below the hard_threshold, and if False, it has to be above
        predicted_error : torch.Tensor, optional
            a tensor of error predictions for the data prompts
        error_threshold : float, default 0.5
            maximum possible probability of error at the intervals
        hysteresis: bool, default False
            if True, the function will apply a hysteresis hard_threshold with the soft hard_threshold defined by threshold_diff
        threshold_diff: float, optional
            the difference between the soft and hard hard_threshold if hysteresis is used; if hysteresis is True, low is False and threshold_diff is None, the soft hard_threshold condition is set to the main_class having a larger probability than other classes
        min_frames_error: int, optional
            if not None, the intervals will only be considered where the error probability is below error_threshold at at least min_frames_error consecutive frames

        Returns
        -------
        valleys : dict
            a dictionary where keys are video ids and values are lists of (start, end, individual name) tuples that denote the chosen intervals
        """

        result = defaultdict(lambda: [])
        if type(predicted) is not dict:
            predicted = self.generate_full_length_prediction(predicted)
        if predicted_error is not None:
            predicted_error = self.generate_full_length_prediction(predicted_error)
        elif min_frames_error is not None and min_frames_error != 0:
            # warnings.warn(
            #     f"The min_frames_error parameter is set to {min_frames_error} but no error prediction "
            #     f"is given! Setting min_frames_error to 0."
            # )
            min_frames_error = 0
        if low and hysteresis and threshold_diff is None:
            raise ValueError(
                "Cannot set low=True, hysteresis=True and threshold_diff=None! Please set threshold_diff."
            )
        if cut_annotated:
            masked_intervals_dict = self.get_annotated_intervals()
        else:
            masked_intervals_dict = None
        print("Valleys found:")
        for v_id in predicted:
            for clip_id in predicted[v_id].keys():
                if predicted_error is not None:
                    error_mask = predicted_error[v_id][clip_id][1, :] < error_threshold
                    if min_frames_error is not None:
                        output, indices, counts = torch.unique_consecutive(
                            error_mask, return_inverse=True, return_counts=True
                        )
                        wrong_indices = torch.where(
                            output * (counts < min_frames_error)
                        )[0]
                        if len(wrong_indices) > 0:
                            for i in wrong_indices:
                                error_mask[indices == i] = False
                else:
                    error_mask = None
                if masked_intervals_dict is not None:
                    masked_intervals = masked_intervals_dict[v_id][clip_id]
                else:
                    masked_intervals = None
                if not hysteresis:
                    res_indices_start, res_indices_end = apply_threshold(
                        predicted[v_id][clip_id][main_class, :],
                        threshold,
                        low,
                        error_mask,
                        min_frames,
                        smooth_interval,
                        masked_intervals,
                    )
                elif threshold_diff is not None:
                    if low:
                        soft_threshold = threshold + threshold_diff
                    else:
                        soft_threshold = threshold - threshold_diff
                    res_indices_start, res_indices_end = apply_threshold_hysteresis(
                        predicted[v_id][clip_id][main_class, :],
                        soft_threshold,
                        threshold,
                        low,
                        error_mask,
                        min_frames,
                        smooth_interval,
                        masked_intervals,
                    )
                else:
                    res_indices_start, res_indices_end = apply_threshold_max(
                        predicted[v_id][clip_id],
                        threshold,
                        main_class,
                        error_mask,
                        min_frames,
                        smooth_interval,
                        masked_intervals,
                    )
                start = self.input_store.get_clip_start(v_id, clip_id)
                result[v_id] += [
                    [i + start, j + start, clip_id]
                    for i, j in zip(res_indices_start, res_indices_end)
                    if self.input_store.get_visibility(
                        v_id, clip_id, i, j, visibility_min_score
                    )
                    > visibility_min_frac
                ]
            result[v_id] = sorted(result[v_id])
            print(f"    {v_id}: {len(result[v_id])}")
        return dict(result)

    def valleys_union(self, valleys_list) -> Dict:
        """
        Find the intersection of two valleys dictionaries

        Parameters
        ----------
        valleys_list : list
            a list of valleys dictionaries

        Returns
        -------
        intersection : dict
            a new valleys dictionary with the intersection of the input intervals
        """

        valleys_list = [x for x in valleys_list if x is not None]
        if len(valleys_list) == 1:
            return valleys_list[0]
        elif len(valleys_list) == 0:
            return {}
        union = {}
        keys_list = [set(valleys.keys()) for valleys in valleys_list]
        keys = set.union(*keys_list)
        for v_id in keys:
            res = []
            clips_list = [
                set([x[-1] for x in valleys[v_id]]) for valleys in valleys_list
            ]
            clips = set.union(*clips_list)
            for clip_id in clips:
                clip_intervals = [
                    x
                    for valleys in valleys_list
                    for x in valleys[v_id]
                    if x[-1] == clip_id
                ]
                v_len = self.input_store.get_clip_length(v_id, clip_id)
                arr = torch.zeros(v_len)
                for start, end, _ in clip_intervals:
                    arr[start:end] += 1
                output, indices, counts = torch.unique_consecutive(
                    arr > 0, return_inverse=True, return_counts=True
                )
                long_indices = torch.where(output)[0]
                res += [
                    (
                        (indices == i).nonzero(as_tuple=True)[0][0].item(),
                        (indices == i).nonzero(as_tuple=True)[0][-1].item(),
                        clip_id,
                    )
                    for i in long_indices
                ]
            union[v_id] = res
        return union

    def valleys_intersection(self, valleys_list) -> Dict:
        """
        Find the intersection of two valleys dictionaries

        Parameters
        ----------
        valleys_list : list
            a list of valleys dictionaries

        Returns
        -------
        intersection : dict
            a new valleys dictionary with the intersection of the input intervals
        """

        valleys_list = [x for x in valleys_list if x is not None]
        if len(valleys_list) == 1:
            return valleys_list[0]
        elif len(valleys_list) == 0:
            return {}
        intersection = {}
        keys_list = [set(valleys.keys()) for valleys in valleys_list]
        keys = set.intersection(*keys_list)
        for v_id in keys:
            res = []
            clips_list = [
                set([x[-1] for x in valleys[v_id]]) for valleys in valleys_list
            ]
            clips = set.intersection(*clips_list)
            for clip_id in clips:
                clip_intervals = [
                    x
                    for valleys in valleys_list
                    for x in valleys[v_id]
                    if x[-1] == clip_id
                ]
                v_len = self.input_store.get_clip_length(v_id, clip_id)
                arr = torch.zeros(v_len)
                for start, end, _ in clip_intervals:
                    arr[start:end] += 1
                output, indices, counts = torch.unique_consecutive(
                    arr, return_inverse=True, return_counts=True
                )
                long_indices = torch.where(output == 2)[0]
                res += [
                    (
                        (indices == i).nonzero(as_tuple=True)[0][0].item(),
                        (indices == i).nonzero(as_tuple=True)[0][-1].item(),
                        clip_id,
                    )
                    for i in long_indices
                ]
            intersection[v_id] = res
        return intersection

    def partition_train_test_val(
        self,
        use_test: float = 0,
        split_path: str = None,
        method: str = "random",
        val_frac: float = 0,
        test_frac: float = 0,
        save_split: bool = False,
        normalize: bool = False,
        skip_normalization_keys: List = None,
        stats: Dict = None,
    ) -> Tuple:
        """
        Partition the dataset into three new datasets

        Parameters
        ----------
        use_test : float, default 0
            The fraction of the test dataset to be used in training without labels
        split_path : str, optional
            The path to load the split information from (if `'file'` method is used) and to save it to
            (if `'save_split'` is `True`)
        method : {'random', 'random:test-from-name', 'random:test-from-name:{name}',
            'val-from-name:{val_name}:test-from-name:{test_name}',
            'random:equalize:segments', 'random:equalize:videos',
            'folders', 'time', 'time:strict', 'file'}
            The partitioning method:
            - `'random'`: sort videos into subsets randomly,
            - `'random:test-from-name'` (or `'random:test-from-name:{name}'`): sort videos into training and validation
                subsets randomly and create
                the test subset from the video ids that start with a speific substring (`'test'` by default, or `name`
                if provided),
            - `'random:equalize:segments'` and `'random:equalize:videos'`: sort videos into subsets randomly but
                making sure that for the rarest classes at least `0.8 * val_frac` of the videos/segments that contain
                occurences of the class get into the validation subset and `0.8 * test_frac` get into the test subset;
                this in ensured for all classes in order of increasing number of occurences until the validation and
                test subsets are full
            - `'val-from-name:{val_name}:test-from-name:{test_name}'`: create the validation and test
                subsets from the video ids that start with specific substrings (`val_name` for validation
                and `test_name` for test) and sort all other videos into the training subset
            - `'folders'`: read videos from folders named *test*, *train* and *val* into corresponding subsets,
            - `'time'`: split each video into training, validation and test subsequences,
            - `'time:strict'`: split each video into validation, test and training subsequences
                and throw out the last segments in validation and test (to get rid of overlaps),
            - `'file'`: split according to a split file.
        val_frac : float, default 0
            The fraction of the dataset to be used in validation
        test_frac : float, default 0
            The fraction of the dataset to be used in test
        save_split : bool, default False
            Save a split file if True

        Returns
        -------
        train_dataset : BehaviorDataset
            train dataset

        val_dataset : BehaviorDataset
            validation dataset

        test_dataset : BehaviorDataset
            test dataset
        """

        train_indices, test_indices, val_indices = self._partition_indices(
            split_path=split_path,
            method=method,
            val_frac=val_frac,
            test_frac=test_frac,
            save_split=save_split,
        )
        val_dataset = self._create_new_dataset(val_indices)
        test_dataset = self._create_new_dataset(test_indices)
        train_dataset = self._create_new_dataset(
            train_indices, ssl_indices=test_indices[: int(len(test_indices) * use_test)]
        )
        print("Number of samples:")
        print(f"    validation:")
        print(f"      {val_dataset.count_classes()}")
        print(f"    training:")
        print(f"      {train_dataset.count_classes()}")
        print(f"    test:")
        print(f"      {test_dataset.count_classes()}")
        if normalize:
            if stats is None:
                print("Computing normalization statistics...")
                stats = train_dataset.get_normalization_stats(skip_normalization_keys)
            else:
                print("Setting loaded normalization statistics...")
            train_dataset.set_normalization_stats(stats)
            val_dataset.set_normalization_stats(stats)
            test_dataset.set_normalization_stats(stats)
        return train_dataset, test_dataset, val_dataset

    def class_weights(self, proportional=False) -> List:
        """
        Calculate class weights in inverse proportion to number of samples
        Returns
        -------
        weights: list
            a list of class weights
        """

        items = sorted(
            [
                (k, v)
                for k, v in self.annotation_store.count_classes().items()
                if k != -100
            ]
        )
        if self.annotation_store.annotation_class() == "exclusive_classification":
            if not proportional:
                numerator = len(self.annotation_store)
            else:
                numerator = max([x[1] for x in items])
            weights = [numerator / (v + 1e-7) for _, v in items]
        else:
            items_zero = sorted(
                [
                    (k, v)
                    for k, v in self.annotation_store.count_classes(zeros=True).items()
                    if k != -100
                ]
            )
            if not proportional:
                numerators = defaultdict(lambda: len(self.annotation_store))
            else:
                numerators = {
                    item_one[0]: max(item_one[1], item_zero[1])
                    for item_one, item_zero in zip(items, items_zero)
                }
            weights = {}
            weights[0] = [numerators[k] / (v + 1e-7) for k, v in items_zero]
            weights[1] = [numerators[k] / (v + 1e-7) for k, v in items]
        return weights

    def boundary_class_weight(self):
        if self.annotation_type != "none":
            f = self.annotation_store.data.flatten()
            _, inv = torch.unique_consecutive(f, return_inverse=True)
            boundary = torch.cat([torch.tensor([0]), torch.diff(inv)]).reshape(
                self.annotation_store.data.shape
            )
            boundary[..., 0] = 0
            cnt = Counter(boundary.flatten().numpy())
            return cnt[1] / cnt[0]
        else:
            return 0

    def count_classes(self, bouts: bool = False) -> Dict:
        """
        Get a class counter dictionary

        Parameters
        ----------
        bouts : bool, default False
            if `True`, instead of frame counts segment counts are returned

        Returns
        -------
        count_dictionary : dict
            a dictionary with class indices as keys and frame or bout counts as values
        """

        return self.annotation_store.count_classes(bouts=bouts)

    def behaviors_dict(self) -> Dict:
        """
        Get a behavior dictionary

        Returns
        -------
        dict
            behavior dictionary
        """

        return self.annotation_store.behaviors_dict()

    def bodyparts_order(self) -> List:
        try:
            return self.input_store.get_bodyparts()
        except:
            raise RuntimeError(
                f"The {self.input_type} input store does not have bodyparts implemented!"
            )

    def features_shape(self) -> Dict:
        """
        Get the shapes of the input features

        Returns
        -------
        shapes : Dict
            a dictionary with the shapes of the features
        """

        sample = self.input_store[0]
        shapes = {k: v.shape for k, v in sample.items()}
        # for key, value in shapes.items():
        #     print(f'{key}: {value}')
        return shapes

    def num_classes(self) -> int:
        """
        Get the number of classes in the data

        Returns
        -------
        num_classes : int
            the number of classes
        """

        return len(self.annotation_store.behaviors_dict())

    def len_segment(self) -> int:
        """
        Get the segment length in the data

        Returns
        -------
        len_segment : int
            the segment length
        """

        sample = self.input_store[0]
        key = list(sample.keys())[0]
        return sample[key].shape[-1]

    def set_ssl_transformations(self, ssl_transformations: List) -> None:
        """
        Set new SSL transformations

        Parameters
        ----------
        ssl_transformations : list
            a list of functions that take a sample feature dictionary as input and output ssl_inputs and ssl_targets
            lists
        """

        self.ssl_transformations = ssl_transformations

    @classmethod
    def new(cls, *args, **kwargs):
        """
        Create a new object of the same class

        Returns
        -------
        new_instance: BehaviorDataset
            a new instance of the same class
        """

        return cls(*args, **kwargs)

    @classmethod
    def get_parameters(cls, data_type: str, annotation_type: str) -> List:
        """
        Get parameters necessary for initialization

        Parameters
        ----------
        data_type : str
            the data type
        annotation_type : str
            the annotation type
        """

        input_features = options.input_stores[data_type].get_parameters()
        annotation_features = options.annotation_stores[
            annotation_type
        ].get_parameters()
        self_features = inspect.getfullargspec(cls.__init__).args
        return self_features + input_features + annotation_features

    @staticmethod
    def data_types() -> List:
        """
        List available data types

        Returns
        -------
        data_types : list
            available data types
        """

        return list(options.input_stores.keys())

    @staticmethod
    def annotation_types() -> List:
        """
        List available annotation types

        Returns
        -------
        annotation_types : list
            available annotation types
        """

        return list(options.annotation_stores.keys())

    def _get_SSL_targets(self, input: Dict) -> Tuple[List, List]:
        """
        Get the SSL inputs and targets from a sample dictionary
        """

        ssl_inputs = []
        ssl_targets = []
        for transform in self.ssl_transformations:
            ssl_input, ssl_target = transform(copy(input))
            ssl_inputs.append(ssl_input)
            ssl_targets.append(ssl_target)
        return ssl_inputs, ssl_targets

    def _create_new_dataset(self, indices: List, ssl_indices: List = None):
        """
        Create a subsample of the dataset, with samples at ssl_indices losing the annotation
        """

        if ssl_indices is None:
            ssl_indices = []
        input_store = self.input_store.create_subsample(indices, ssl_indices)
        annotation_store = self.annotation_store.create_subsample(indices, ssl_indices)
        new = self.new(
            data_type=self.input_type,
            annotation_type=self.annotation_type,
            ssl_transformations=self.ssl_transformations,
            annotation_store=annotation_store,
            input_store=input_store,
            ids=list(indices) + list(ssl_indices),
            recompute_annotation=False,
        )
        return new

    def _load_input_store(self, data_type: str, key_objects: Tuple) -> InputStore:
        """
        Load input store from key objects
        """

        input_store = options.input_stores[data_type](key_objects=key_objects)
        return input_store

    def _load_annotation_store(
        self, annotation_type: str, key_objects: Tuple
    ) -> AnnotationStore:
        """
        Load annotation store from key objects
        """

        annotation_store = options.annotation_stores[annotation_type](
            key_objects=key_objects
        )
        return annotation_store

    def _get_input_store(self, data_type: str, data_parameters: Dict) -> InputStore:
        """
        Create input store from parameters
        """

        data_parameters["key_objects"] = None
        input_store = options.input_stores[data_type](**data_parameters)
        return input_store

    def _get_annotation_store(
        self, annotation_type: str, data_parameters: Dict
    ) -> AnnotationStore:
        """
        Create annotation store from parameters
        """

        annotation_store = options.annotation_stores[annotation_type](**data_parameters)
        return annotation_store

    def set_indexing_parameters(self, unlabeled: bool, tag: int) -> None:
        """
        Set the parameters that change the subset that is returned at `__getitem__`

        Parameters
        ----------
        unlabeled : bool
            a pseudolabeling parameter; return only unlabeled samples if `True`, only labeled if `False` and
            all if `None`
        tag : int
            if not `None`, only samples with this meta tag will be returned
        """

        if unlabeled != self.return_unlabeled:
            self.annotation_indices = self.annotation_store.get_indices(unlabeled)
            self.return_unlabeled = unlabeled
        if tag != self.tag:
            self.input_indices = self.input_store.get_indices(tag)
            self.tag = tag
        self.indices = [x for x in self.annotation_indices if x in self.input_indices]

    def _get_idx(self, index: int) -> int:
        """
        Get index in full dataset
        """

        return self.indices[index]

        # return self.annotation_store.get_idx(
        #     index, return_unlabeled=self.return_unlabeled
        # )

    def _partition_indices(
        self,
        split_path: str = None,
        method: str = "random",
        val_frac: float = 0,
        test_frac: float = 0,
        save_split: bool = False,
    ) -> Tuple[List, List, List]:
        """
        Partition indices into train, validation, test subsets
        """

        if self.mask is not None:
            val_indices = self.mask["val_ids"]
            train_indices = [x for x in range(len(self)) if x not in val_indices]
            test_indices = []
        elif method == "random":
            videos = np.array(self.input_store.get_video_id_order())
            all_videos = list(set(videos))
            if len(all_videos) == 1:
                warnings.warn(
                    "There is only one video in the dataset, so train/val/test split is done on segments; "
                    'that might lead to overlaps, please consider using "time" or "time:strict" as the '
                    "partitioning method instead"
                )
                # Quick fix for single video: the problem with this is that the segments can overlap
                # length = int(self.input_store.get_original_coordinates()[-1][1])    # number of segments
                length = len(self.input_store.get_original_coordinates())
                val_len = int(val_frac * length)
                test_len = int(test_frac * length)
                all_indices = np.random.choice(np.arange(length), length, replace=False)
                val_indices = all_indices[:val_len]
                test_indices = all_indices[val_len : val_len + test_len]
                train_indices = all_indices[val_len + test_len :]
                coords = self.input_store.get_original_coordinates()
                if save_split:
                    self._save_partition(
                        coords[train_indices],
                        coords[val_indices],
                        coords[test_indices],
                        split_path,
                        coords=True,
                    )
            else:
                length = len(all_videos)
                val_len = int(val_frac * length)
                test_len = int(test_frac * length)
                validation = all_videos[:val_len]
                test = all_videos[val_len : val_len + test_len]
                training = all_videos[val_len + test_len :]
                train_indices = np.where(np.isin(videos, training))[0]
                val_indices = np.where(np.isin(videos, validation))[0]
                test_indices = np.where(np.isin(videos, test))[0]
                if save_split:
                    self._save_partition(training, validation, test, split_path)
        elif method.startswith("random:equalize"):
            counter = self.count_classes()
            counter = sorted(list([(v, k) for k, v in counter.items()]))
            classes = [x[1] for x in counter]
            indicator = {c: [] for c in classes}
            if method.endswith("videos"):
                videos = np.array(self.input_store.get_video_id_order())
                all_videos = list(set(videos))
                total_len = len(all_videos)
                for video_id in all_videos:
                    video_coords = np.where(videos == video_id)[0]
                    ann = torch.cat(
                        [self.annotation_store[i] for i in video_coords], dim=-1
                    )
                    for c in classes:
                        if self.annotation_class() == "nonexclusive_classification":
                            indicator[c].append(torch.sum(ann[c] == 1) > 0)
                        elif self.annotation_class() == "exclusive_classification":
                            indicator[c].append(torch.sum(ann == c) > 0)
                        else:
                            raise ValueError(
                                f"The random:equalize partition method is not implemented"
                                f"for the {self.annotation_class()} annotation class"
                            )
            elif method.endswith("segments"):
                total_len = len(self)
                for ann in self.annotation_store:
                    for c in classes:
                        if self.annotation_class() == "nonexclusive_classification":
                            indicator[c].append(torch.sum(ann[c] == 1) > 0)
                        elif self.annotation_class() == "exclusive_classification":
                            indicator[c].append(torch.sum(ann == c) > 0)
                        else:
                            raise ValueError(
                                f"The random:equalize partition method is not implemented"
                                f"for the {self.annotation_class()} annotation class"
                            )
            else:
                values = []
                for v in options.partition_methods.values():
                    values += v
                raise ValueError(
                    f"The {method} partition method is not recognized; please choose from {values}"
                )
            val_indices = []
            test_indices = []
            for c in classes:
                indicator[c] = np.array(indicator[c])
                ind = np.where(indicator[c])[0]
                np.random.shuffle(ind)
                c_sum = len(ind)
                in_val = np.sum(indicator[c][val_indices])
                in_test = np.sum(indicator[c][test_indices])
                while (
                    len(val_indices) < val_frac * total_len
                    and in_val < val_frac * c_sum * 0.8
                ):
                    first, ind = ind[0], ind[1:]
                    val_indices = list(set(val_indices).union({first}))
                    in_val = np.sum(indicator[c][val_indices])
                while (
                    len(test_indices) < test_frac * total_len
                    and in_test < test_frac * c_sum * 0.8
                ):
                    first, ind = ind[0], ind[1:]
                    test_indices = list(set(test_indices).union({first}))
                    in_test = np.sum(indicator[c][test_indices])
            if len(val_indices) < int(val_frac * total_len):
                left_val = int(val_frac * total_len) - len(val_indices)
            else:
                left_val = 0
            if len(test_indices) < int(test_frac * total_len):
                left_test = int(test_frac * total_len) - len(test_indices)
            else:
                left_test = 0
            indicator = np.ones(total_len)
            indicator[val_indices] = 0
            indicator[test_indices] = 0
            ind = np.where(indicator)[0]
            np.random.shuffle(ind)
            val_indices += list(ind[:left_val])
            test_indices += list(ind[left_val : left_val + left_test])
            train_indices = list(ind[left_val + left_test :])
            if save_split:
                if method.endswith("segments"):
                    coords = self.input_store.get_original_coordinates()
                    self._save_partition(
                        coords[train_indices],
                        coords[val_indices],
                        coords[test_indices],
                        coords[split_path],
                        coords=True,
                    )
                else:
                    all_videos = np.array(all_videos)
                    validation = all_videos[val_indices]
                    test = all_videos[test_indices]
                    training = all_videos[train_indices]
                    self._save_partition(training, validation, test, split_path)
        elif method.startswith("random:test-from-name"):
            split = method.split(":")
            if len(split) > 2:
                test_name = split[-1]
            else:
                test_name = "test"
            videos = np.array(self.input_store.get_video_id_order())
            all_videos = list(set(videos))
            test = []
            train_videos = []
            for x in all_videos:
                if x.startswith(test_name):
                    test.append(x)
                else:
                    train_videos.append(x)
            length = len(train_videos)
            val_len = int(val_frac * length)
            validation = train_videos[:val_len]
            training = train_videos[val_len:]
            train_indices = np.where(np.isin(videos, training))[0]
            val_indices = np.where(np.isin(videos, validation))[0]
            test_indices = np.where(np.isin(videos, test))[0]
            if save_split:
                self._save_partition(training, validation, test, split_path)
        elif method.startswith("val-from-name"):
            split = method.split(":")
            if split[2] != "test-from-name":
                raise ValueError(
                    f"The {method} partition method is not recognized, please choose from {options.partition_methods}"
                )
            val_name = split[1]
            test_name = split[-1]
            videos = np.array(self.input_store.get_video_id_order())
            all_videos = list(set(videos))
            test = []
            validation = []
            training = []
            for x in all_videos:
                if x.startswith(test_name):
                    test.append(x)
                elif x.startswith(val_name):
                    validation.append(x)
                else:
                    training.append(x)
            train_indices = np.where(np.isin(videos, training))[0]
            val_indices = np.where(np.isin(videos, validation))[0]
            test_indices = np.where(np.isin(videos, test))[0]
        elif method == "folders":
            folders = np.array(self.input_store.get_folder_order())
            videos = np.array(self.input_store.get_video_id_order())
            train_indices = np.where(np.isin(folders, ["training", "train"]))[0]
            if np.sum(np.isin(folders, ["validation", "val"])) > 0:
                val_indices = np.where(np.isin(folders, ["validation", "val"]))[0]
            else:
                train_videos = list(set(videos[train_indices]))
                val_len = int(val_frac * len(train_videos))
                validation = train_videos[:val_len]
                training = train_videos[val_len:]
                train_indices = np.where(np.isin(videos, training))[0]
                val_indices = np.where(np.isin(videos, validation))[0]
            test_indices = np.where(folders == "test")[0]
            if save_split:
                self._save_partition(
                    list(set(videos[train_indices])),
                    list(set(videos[val_indices])),
                    list(set(videos[test_indices])),
                    split_path,
                )
        elif method.startswith("leave-one-out"):
            n = int(method.split(":")[-1])
            videos = np.array(self.input_store.get_video_id_order())
            all_videos = sorted(list(set(videos)))
            validation = [all_videos.pop(n)]
            training = all_videos
            train_indices = np.where(np.isin(videos, training))[0]
            val_indices = np.where(np.isin(videos, validation))[0]
            test_indices = np.array([])
        elif method.startswith("time"):
            if method.endswith("strict"):
                len_segment = self.len_segment()
                # TODO: change this
                step = self.input_store.step
                num_removed = len_segment // step
            else:
                num_removed = 0
            videos = np.array(self.input_store.get_video_id_order())
            all_videos = set(videos)
            train_indices = []
            val_indices = []
            test_indices = []
            start = 0
            if len(method.split(":")) > 1 and method.split(":")[1] == "start-from":
                start = float(method.split(":")[2])
            for video_id in all_videos:
                video_indices = np.where(videos == video_id)[0]
                val_len = int(val_frac * len(video_indices))
                test_len = int(test_frac * len(video_indices))
                start_pos = int(start * len(video_indices))
                all_ind = np.ones(len(video_indices))
                val_indices += list(video_indices[start_pos : start_pos + val_len])
                all_ind[start_pos : start_pos + val_len] = 0
                if start_pos + val_len > len(video_indices):
                    p = start_pos + val_len - len(video_indices)
                    val_indices += list(video_indices[:p])
                    all_ind[:p] = 0
                else:
                    p = start_pos + val_len
                test_indices += list(video_indices[p : p + test_len])
                all_ind[p : p + test_len] = 0
                if p + test_len > len(video_indices):
                    p = test_len + p - len(video_indices)
                    test_indices += list(video_indices[:p])
                    all_ind[:p] = 0
                train_indices += list(video_indices[all_ind > 0])
                for _ in range(num_removed):
                    if len(val_indices) > 0:
                        val_indices.pop(-1)
                    if len(test_indices) > 0:
                        test_indices.pop(-1)
                    if start > 0 and len(train_indices) > 0:
                        train_indices.pop(-1)
        elif method == "file":
            if split_path is None:
                raise ValueError(
                    'You need to either set split_path or change partition method ("file" requires a file)'
                )
            with open(split_path) as f:
                train_line = f.readline()
                line = f.readline()
                while not line.startswith("Validation") and not line.startswith(
                    "Validataion"
                ):
                    line = f.readline()
                if line.startswith("Validation"):
                    validation = []
                    test = []
                    while True:
                        line = f.readline()
                        if line.startswith("Test") or len(line) == 0:
                            break
                        validation.append(line.rstrip())
                    while True:
                        line = f.readline()
                        if len(line) == 0:
                            break
                        test.append(line.rstrip())
                    type = train_line[9:-2]
                else:
                    line = f.readline()
                    validation = line.rstrip(",\n ").split(", ")
                    test = []
                    type = "videos"
            if type == "videos":
                videos = np.array(self.input_store.get_video_id_order())
                val_indices = np.where(np.isin(videos, validation))[0]
                test_indices = np.where(np.isin(videos, test))[0]
            elif type == "coords":
                coords = self.input_store.get_original_coordinates()
                video_ids = self.input_store.get_video_id_order()
                clip_ids = [self.input_store.get_clip_id(coord) for coord in coords]
                starts, ends = zip(
                    *[self.input_store.get_clip_start_end(coord) for coord in coords]
                )
                coords = np.array(
                    [
                        f"{video_id}---{clip_id}---{start}-{end}"
                        for video_id, clip_id, start, end in zip(
                            video_ids, clip_ids, starts, ends
                        )
                    ]
                )
                val_indices = np.where(np.isin(coords, validation))[0]
                test_indices = np.where(np.isin(coords, test))[0]
            else:
                raise ValueError("The split path has unrecognized format!")
            all = np.ones(len(self))
            all[val_indices] = 0
            all[test_indices] = 0
            train_indices = np.where(all)[0]
        else:
            raise ValueError(
                f"The {method} partition is not recognized, please choose from {options.partition_methods}"
            )
        return sorted(train_indices), sorted(test_indices), sorted(val_indices)

    def _save_partition(
        self,
        training: List,
        validation: List,
        test: List,
        split_path: str,
        coords: bool = False,
    ) -> None:
        """
        Save a split file
        """

        if coords:
            name = "coords"
            training_coords = []
            val_coords = []
            test_coords = []
            for coord in training:
                video_id = self.input_store.get_video_id(coord)
                clip_id = self.input_store.get_clip_id(coord)
                start, end = self.input_store.get_clip_start_end(coord)
                training_coords.append(f"{video_id}---{clip_id}---{start}-{end}")
            for coord in validation:
                video_id = self.input_store.get_video_id(coord)
                clip_id = self.input_store.get_clip_id(coord)
                start, end = self.input_store.get_clip_start_end(coord)
                val_coords.append(f"{video_id}---{clip_id}---{start}-{end}")
            for coord in test:
                video_id = self.input_store.get_video_id(coord)
                clip_id = self.input_store.get_clip_id(coord)
                start, end = self.input_store.get_clip_start_end(coord)
                test_coords.append(f"{video_id}---{clip_id}---{start}-{end}")
            training, validation, test = training_coords, val_coords, test_coords
        else:
            name = "videos"
        if split_path is not None:
            with open(split_path, "w") as f:
                f.write(f"Training {name}:\n")
                for x in training:
                    f.write(x + "\n")
                f.write(f"Validation {name}:\n")
                for x in validation:
                    f.write(x + "\n")
                f.write(f"Test {name}:\n")
                for x in test:
                    f.write(x + "\n")

    def _get_intervals_from_ind(self, frame_indices: np.ndarray):
        """
        Get a list of intervals from a list of frame indices

        Example: `[0, 1, 2, 5, 6, 8] -> [[0, 3], [5, 7], [8, 9]]`.

        Parameters
        ----------
        frame_indices : np.ndarray
            a list of frame indices

        Returns
        -------
        intervals : list
            a list of interval boundaries
        """

        masked_intervals = []
        breaks = np.where(np.diff(frame_indices) != 1)[0]
        if len(frame_indices) > 0:
            start = frame_indices[0]
            for k in breaks:
                masked_intervals.append([start, frame_indices[k] + 1])
                start = frame_indices[k + 1]
            masked_intervals.append([start, frame_indices[-1] + 1])
        return masked_intervals

    def get_intervals(self) -> Tuple[dict, Optional[list]]:
        """
        Get a list of intervals covered by the dataset in the original coordinates

        Returns
        -------
        intervals : dict
            a nested dictionary where first-level keys are video ids, second-level keys are clip ids and
            values are lists of the intervals in `[start, end]` format
        """

        counter = defaultdict(lambda: {})
        coordinates = self.input_store.get_original_coordinates()
        for coords in coordinates:
            l = self.input_store.get_clip_length_from_coords(coords)
            video_name = self.input_store.get_video_id(coords)
            clip_id = self.input_store.get_clip_id(coords)
            start, end = self.input_store.get_clip_start_end(coords)
            if clip_id not in counter[video_name]:
                counter[video_name][clip_id] = np.zeros(l)
            counter[video_name][clip_id][start:end] = 1
        result = {video_name: {} for video_name in counter}
        for video_name in counter:
            for clip_id in counter[video_name]:
                result[video_name][clip_id] = self._get_intervals_from_ind(
                    np.where(counter[video_name][clip_id])[0]
                )
        return result, self.ids

    def get_unannotated_intervals(self, first_intervals=None) -> Dict:
        """
        Get a list of intervals in the original coordinates where there is no annotation

        Returns
        -------
        intervals : dict
            a nested dictionary where first-level keys are video ids, second-level keys are clip ids and
            values are lists of the intervals in `[start, end]` format
        """

        counter_value = 2
        if first_intervals is None:
            first_intervals = defaultdict(lambda: defaultdict(lambda: []))
            counter_value = 1
        counter = defaultdict(lambda: {})
        coordinates = self.input_store.get_original_coordinates()
        for i, coords in enumerate(coordinates):
            l = self.input_store.get_clip_length_from_coords(coords)
            ann = self.annotation_store[i]
            if (
                self.annotation_store.annotation_class()
                == "nonexclusive_classification"
            ):
                ann = ann[0, :]
            video_name = self.input_store.get_video_id(coords)
            clip_id = self.input_store.get_clip_id(coords)
            start, end = self.input_store.get_clip_start_end(coords)
            if clip_id not in counter[video_name]:
                counter[video_name][clip_id] = np.ones(l)
            counter[video_name][clip_id][start:end] = (ann[: end - start] == -100).int()
        result = {video_name: {} for video_name in counter}
        for video_name in counter:
            for clip_id in counter[video_name]:
                for start, end in first_intervals[video_name][clip_id]:
                    counter[video_name][clip_id][start:end] += 1
                result[video_name][clip_id] = self._get_intervals_from_ind(
                    np.where(counter[video_name][clip_id] == counter_value)[0]
                )
        return result

    def get_annotated_intervals(self) -> Dict:
        """
        Get a list of intervals in the original coordinates where there is no annotation

        Returns
        -------
        intervals : dict
            a nested dictionary where first-level keys are video ids, second-level keys are clip ids and
            values are lists of the intervals in `[start, end]` format
        """

        if self.annotation_type == "none":
            return []
        counter_value = 1
        counter = defaultdict(lambda: {})
        coordinates = self.input_store.get_original_coordinates()
        for i, coords in enumerate(coordinates):
            l = self.input_store.get_clip_length_from_coords(coords)
            ann = self.annotation_store[i]
            video_name = self.input_store.get_video_id(coords)
            clip_id = self.input_store.get_clip_id(coords)
            start, end = self.input_store.get_clip_start_end(coords)
            if clip_id not in counter[video_name]:
                counter[video_name][clip_id] = np.zeros(l)
            if (
                self.annotation_store.annotation_class()
                == "nonexclusive_classification"
            ):
                counter[video_name][clip_id][start:end] = (
                    torch.sum(ann[:, : end - start] != -100, dim=0) > 0
                ).int()
            else:
                counter[video_name][clip_id][start:end] = (
                    ann[: end - start] != -100
                ).int()
        result = {video_name: {} for video_name in counter}
        for video_name in counter:
            for clip_id in counter[video_name]:
                result[video_name][clip_id] = self._get_intervals_from_ind(
                    np.where(counter[video_name][clip_id] == counter_value)[0]
                )
        return result

    def get_ids(self) -> Dict:
        """
        Get a dictionary of all clip ids in the dataset

        Returns
        -------
        ids : dict
            a dictionary where keys are video ids and values are lists of clip ids
        """

        coordinates = self.input_store.get_original_coordinates()
        video_ids = np.array(self.input_store.get_video_id_order())
        id_set = set(video_ids)
        result = {}
        for video_id in id_set:
            coords = coordinates[video_ids == video_id]
            clip_ids = list({self.input_store.get_clip_id(c) for c in coords})
            result[video_id] = clip_ids
        return result

    def get_len(self, video_id: str, clip_id: str) -> int:
        """
        Get the length of a specific clip

        Parameters
        ----------
        video_id : str
            the video id
        clip_id : str
            the clip id

        Returns
        -------
        length : int
            the length
        """

        return self.input_store.get_clip_length(video_id, clip_id)

    def get_confusion_matrix(
        self, prediction: torch.Tensor, confusion_type: str = "recall"
    ) -> Tuple[ndarray, list]:
        """
        Get a confusion matrix

        Parameters
        ----------
        prediction : torch.Tensor
            a tensor of predicted class probabilities of shape `(#samples, #classes, #frames)`
        confusion_type : {"recall", "precision"}
            for datasets with non-exclusive annotation, if `type` is `"recall"`, only false positives are taken
            into account, and if `type` is `"precision"`, only false negatives

        Returns
        -------
        confusion_matrix : np.ndarray
            a confusion matrix of shape `(#classes, #classes)` where `A[i, j] = F_ij/N_i`, `F_ij` is the number of
            frames that have the i-th label in the ground truth and a false positive j-th label in the prediction,
            `N_i` is the number of frames that have the i-th label in the ground truth
        classes : list
            a list of classes
        """

        behaviors_dict = self.annotation_store.behaviors_dict()
        num_behaviors = len(behaviors_dict)
        confusion_matrix = np.zeros((num_behaviors, num_behaviors))
        if self.annotation_store.annotation_class() == "exclusive_classification":
            exclusive = True
            confusion_type = None
        elif self.annotation_store.annotation_class() == "nonexclusive_classification":
            exclusive = False
        else:
            raise RuntimeError(
                f"The {self.annotation_store.annotation_class()} annotation class is not recognized!"
            )
        for ann, p in zip(self.annotation_store, prediction):
            if exclusive:
                class_prediction = torch.max(p, dim=0)[1]
                for i in behaviors_dict.keys():
                    for j in behaviors_dict.keys():
                        confusion_matrix[i, j] += int(
                            torch.sum(class_prediction[ann == i] == j)
                        )
            else:
                class_prediction = (p > 0.5).int()
                for i in behaviors_dict.keys():
                    for j in behaviors_dict.keys():
                        if confusion_type == "recall":
                            pred = deepcopy(class_prediction[j])
                            if i != j:
                                pred[ann[j] == 1] = 0
                            confusion_matrix[i, j] += int(torch.sum(pred[ann[i] == 1]))
                        elif confusion_type == "precision":
                            annotation = deepcopy(ann[j])
                            if i != j:
                                annotation[class_prediction[j] == 1] = 0
                            confusion_matrix[i, j] += int(
                                torch.sum(annotation[class_prediction[i] == 1])
                            )
                        else:
                            raise ValueError(
                                f"The {confusion_type} type is not recognized; please choose from ['recall', 'precision']"
                            )
        counter = self.annotation_store.count_classes()
        for i in behaviors_dict.keys():
            if counter[i] != 0:
                if confusion_type == "recall" or confusion_type is None:
                    confusion_matrix[i, :] /= counter[i]
                else:
                    confusion_matrix[:, i] /= counter[i]
        return confusion_matrix, list(behaviors_dict.values()), confusion_type

    def annotation_class(self) -> str:
        """
        Get the type of annotation ('exclusive_classification', 'nonexclusive_classification', more coming soon)

        Returns
        -------
        annotation_class : str
            the type of annotation
        """

        return self.annotation_store.annotation_class()

    def set_normalization_stats(self, stats: Dict) -> None:
        """
        Set the stats to normalize data at runtime

        Parameters
        ----------
        stats : dict
            a nested dictionary where first-level keys are feature key names, second-level keys are 'mean' and 'std'
            and values are the statistics in `torch` tensors of shape `(#features, 1)`
        """

        self.stats = stats

    def get_min_max_frames(self, video_id) -> Tuple[Dict, Dict]:
        coords = self.input_store.get_original_coordinates()
        clips = set(
            [
                self.input_store.get_clip_id(c)
                for c in coords
                if self.input_store.get_video_id(c) == video_id
            ]
        )
        min_frames = {}
        max_frames = {}
        for clip in clips:
            start = self.input_store.get_clip_start(video_id, clip)
            end = start + self.input_store.get_clip_length(video_id, clip)
            min_frames[clip] = start
            max_frames[clip] = end - 1
        return min_frames, max_frames

    def get_normalization_stats(self, skip_keys=None) -> Dict:
        """
        Get mean and standard deviation for each key

        Returns
        -------
        stats : dict
            a nested dictionary where first-level keys are feature key names, second-level keys are 'mean' and 'std'
            and values are the statistics in `torch` tensors of shape `(#features, 1)`
        """

        stats = defaultdict(lambda: {})
        sums = defaultdict(lambda: 0)
        if skip_keys is None:
            skip_keys = []
        counter = defaultdict(lambda: 0)
        for sample in tqdm(self):
            for key, value in sample["input"].items():
                key_name = key.split("---")[0]
                if key_name not in skip_keys:
                    sums[key_name] += value[:, value.sum(0) != 0].sum(-1)
                counter[key_name] += torch.sum(value.sum(0) != 0)
        for key, value in sums.items():
            stats[key]["mean"] = (value / counter[key]).unsqueeze(-1)
        sums = defaultdict(lambda: 0)
        for sample in tqdm(self):
            for key, value in sample["input"].items():
                key_name = key.split("---")[0]
                if key_name not in skip_keys:
                    sums[key_name] += (
                        (value[:, value.sum(0) != 0] - stats[key_name]["mean"]) ** 2
                    ).sum(-1)
        for key, value in sums.items():
            stats[key]["std"] = np.sqrt(value.unsqueeze(-1) / counter[key])
        return stats
