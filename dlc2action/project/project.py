#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
"""
Project interface
"""
import copy
from email.policy import default
import os
from re import search
from typing import Dict, List, Tuple, Union, Set, Iterable, Any, Optional
import shutil

from numpy import ndarray
from ruamel.yaml import YAML
import pickle
import pandas as pd
from dlc2action.data.dataset import BehaviorDataset
from dlc2action.utils import apply_threshold
from collections.abc import Mapping
from collections import defaultdict

from dlc2action.task.task_dispatcher import TaskDispatcher
import warnings
from copy import deepcopy, copy
import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from itertools import product
from collections.abc import Iterable
import optuna
import plotly
import torch
from pathlib import Path
from dlc2action import options, __version__
from ruamel.yaml.comments import CommentedMap, CommentedSet
from tqdm import tqdm
from dlc2action.project.meta import (
    Searches,
    SavedStores,
    Run,
    SavedRuns,
    DecisionThresholds,
)


class Project:
    """
    A class to create and maintain the project files + keep track of experiments
    """

    def __init__(
        self,
        name: str,
        data_type: str = None,
        annotation_type: str = "none",
        projects_path: str = None,
        data_path: Union[str, List] = None,
        annotation_path: Union[str, List] = None,
        copy: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        name : str
            name of the project
        data_type : str, optional
            data type (run Project.data_types() to see available options; has to be provided if the project is being
            created)
        annotation_type : str, default 'none'
            annotation type (run Project.annotation_types() to see available options)
        projects_path : str, optional
            path to the projects folder (is filled with ~/DLC2Action by default)
        data_path : str, optional
            path to the folder containing input files for the project (has to be provided if the project is being
            created)
        annotation_path : str, optional
            path to the folder containing annotation files for the project
        copy : bool, default False
            if True, the files from annotation_path and data_path will be copied to the projects folder;
            otherwise they will be moved
        """

        if projects_path is None:
            projects_path = os.path.join(str(Path.home()), "DLC2Action")
        if not os.path.exists(projects_path):
            os.mkdir(projects_path)
        self.project_path = os.path.join(projects_path, name)
        self.name = name
        self.data_type = data_type
        self.annotation_type = annotation_type
        self.data_path = data_path
        self.annotation_path = annotation_path
        if not os.path.exists(self.project_path):
            if data_type is None:
                raise ValueError(
                    "The data_type parameter is necessary when creating a new project!"
                )
            self._initialize_project(
                data_type, annotation_type, data_path, annotation_path, copy
            )
        else:
            self.annotation_type, self.data_type = self._read_types()
            if data_type != self.data_type and data_type is not None:
                raise ValueError(
                    f"The project has already been initialized with data_type={self.data_type}!"
                )
            if annotation_type != self.annotation_type and annotation_type != "none":
                raise ValueError(
                    f"The project has already been initialized with annotation_type={self.annotation_type}!"
                )
            self.annotation_path, data_path = self._read_paths()
            if self.data_path is None:
                self.data_path = data_path
            # if data_path != self.data_path and data_path is not None:
            #     raise ValueError(
            #         f"The project has already been initialized with data_path={self.data_path}!"
            #     )
            if annotation_path != self.annotation_path and annotation_path is not None:
                raise ValueError(
                    f"The project has already been initialized with annotation_path={self.annotation_path}!"
                )
        self._update_configs()

    def _aggregate_predictions(
        self,
        prediction_name: str,
        episode_names: List,
        load_epochs: List = None,
        parameters_update: Dict = None,
        data_path: str = None,
        file_paths: Set = None,
        mode: str = "all",
        augment_n: int = 0,
        evaluate: bool = False,
        task: TaskDispatcher = None,
        embedding: bool = False,
    ) -> Tuple[TaskDispatcher, Dict, str, torch.Tensor]:
        """
        Generate a prediction
        """

        if load_epochs is None:
            load_epochs = [None for _ in episode_names]
        prediction = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        cnt = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        behs = set(self.get_behavior_dictionary(episode_names[0]).values())
        if not all(
            [
                set(self.get_behavior_dictionary(x).values()) == behs
                for x in episode_names
            ]
        ):
            raise ValueError(f"The behavior sets are different in {episode_names}")
        behaviors = set()
        for i, episode_name in enumerate(episode_names):
            task, parameters, data_mode, new_pred, _ = self._make_prediction(
                prediction_name,
                episode_names=[episode_name],
                load_epochs=[load_epochs[i]],
                parameters_update=parameters_update,
                data_path=data_path,
                file_paths=file_paths,
                mode=mode,
                augment_n=augment_n,
                evaluate=evaluate,
                task=task,
                embedding=embedding,
            )
            new_pred = task.dataset(data_mode).generate_full_length_prediction(new_pred)
            beh_dict = task.behaviors_dict()
            for video_id, video_values in new_pred.items():
                for clip_id, clip_prediction in video_values.items():
                    for beh_i in range(clip_prediction.shape[0]):
                        prediction[video_id][clip_id][
                            beh_dict[beh_i]
                        ] += clip_prediction[beh_i, :].unsqueeze(0)
                        cnt[video_id][clip_id][beh_dict[beh_i]] += 1
                        behaviors.add(beh_dict[beh_i])
        output = defaultdict(lambda: {})
        # behaviors = sorted(behaviors)
        behavior_indices = sorted(
            [x for x in task.behaviors_dict().keys() if x != -100]
        )
        behaviors = [task.behaviors_dict()[key] for key in behavior_indices]
        for video_id, video_values in prediction.items():
            for clip_id, clip_values in video_values.items():
                pred = torch.cat(
                    [
                        clip_values[beh] / cnt[video_id][clip_id][beh]
                        for beh in behaviors
                    ],
                    0,
                )
                output[video_id][clip_id] = pred
        return task, parameters, data_mode, dict(output), None

    def _make_prediction(
        self,
        prediction_name: str,
        episode_names: List,
        load_epochs: List = None,
        parameters_update: Dict = None,
        data_path: str = None,
        file_paths: Set = None,
        mode: str = "all",
        augment_n: int = 0,
        evaluate: bool = False,
        task: TaskDispatcher = None,
        embedding: bool = False,
    ) -> Tuple[TaskDispatcher, Dict, str, torch.Tensor]:
        """
        Generate a prediction
        """

        names = []
        epochs = []
        if load_epochs is None:
            load_epochs = [None for _ in episode_names]
        if len(load_epochs) != len(episode_names):
            raise ValueError(
                f"The length of load_epochs and the length of episode_names should be the same!"
            )
        for i, episode_name in enumerate(episode_names):
            names += self._episodes().get_runs(episode_name)
            epochs.append(load_epochs[i])
        if len(names) == 0:
            warnings.warn(f"None of the episodes {episode_names} exist!")
            names = [None]
        episodes = self._episodes()
        lengths = [
            episodes.load_parameters(name)["general"]["len_segment"] for name in names
        ]
        overlaps = [
            episodes.load_parameters(name)["general"]["overlap"] for name in names
        ]
        if not all([x == lengths[0] for x in lengths]):
            raise ValueError(f"Episodes {episode_names} have different segment lengths")
        if not all([x == overlaps[0] for x in overlaps]):
            raise ValueError(f"Episodes {episode_names} have different overlaps")
        load_epochs = epochs
        prediction = None
        decision_thresholds = None
        time_total = 0
        behavior_dicts = [
            self.get_behavior_dictionary(episode_name) for episode_name in names
        ]
        if not all(
            [
                set(d.values()) == set(behavior_dicts[0].values())
                for d in behavior_dicts[1:]
            ]
        ):
            raise ValueError(
                f"Episodes {episode_names} have different sets of behaviors!"
            )
        behavior_indices = [x for x in behavior_dicts[0].keys() if x != -100]
        behaviors = [behavior_dicts[0][i] for i in behavior_indices]
        cnt = defaultdict(lambda: 0)
        behavior_probs = defaultdict(lambda: 0)
        for episode_name, load_epoch, behavior_dict in zip(
            names, load_epochs, behavior_dicts
        ):
            print(f"episode {episode_name}")
            task, parameters, data_mode = self._make_task_prediction(
                prediction_name=prediction_name,
                load_episode=episode_name,
                parameters_update=parameters_update,
                load_epoch=load_epoch,
                data_path=data_path,
                mode=mode,
                file_paths=file_paths,
                task=task,
                decision_thresholds=decision_thresholds,
            )
            behavior_indices_cur = [x for x in behavior_dict.keys() if x != -100]
            behaviors_cur = [behavior_dict[i] for i in behavior_indices_cur]
            # data_mode = "train" if mode == "all" else mode
            time_start = time.time()
            new_pred = task.predict(
                data_mode,
                raw_output=True,
                apply_primary_function=True,
                augment_n=augment_n,
                embedding=embedding,
            )
            for j, beh in enumerate(behaviors_cur):
                cnt[beh] += 1
                behavior_probs[beh] += new_pred[:, j, :].unsqueeze(1)
            # indices = [
            #     behaviors.index(behavior_dict[i]) for i in range(new_pred.shape[1])
            # ]
            # new_pred = new_pred[:, indices, :]
            time_end = time.time()
            time_total += time_end - time_start
            if evaluate:
                _, metrics = task.evaluate_prediction(new_pred, data=data_mode)
                if mode == "val":
                    self._update_episode_metrics(episode_name, metrics)
            # if prediction is None:
            #     prediction = new_pred
            # else:
            #     prediction += new_pred
            print("\n")
        prediction = torch.cat([behavior_probs[beh] / cnt[beh] for beh in behaviors], 1)
        hours = int(time_total // 3600)
        time_total -= hours * 3600
        minutes = int(time_total // 60)
        time_total -= minutes * 60
        seconds = int(time_total)
        inference_time = f"{hours}:{minutes:02}:{seconds:02}"
        # prediction /= len(names)
        return task, parameters, data_mode, prediction, inference_time

    def _make_task_prediction(
        self,
        prediction_name: str,
        load_episode: str = None,
        parameters_update: Dict = None,
        load_epoch: int = None,
        data_path: str = None,
        mode: str = "val",
        file_paths: Set = None,
        decision_thresholds: List = None,
        task: TaskDispatcher = None,
    ) -> Tuple[TaskDispatcher, Dict, str]:
        """
        Make a `TaskDispatcher` object that will be used to generate a prediction
        """

        if parameters_update is None:
            parameters_update = {}
        parameters_update_second = {}
        if mode == "all" or data_path is not None or file_paths is not None:
            parameters_update_second["training"] = {
                "val_frac": 0,
                "test_frac": 0,
                "partition_method": "random",
                "save_split": False,
                "split_path": None,
            }
            mode = "train"
        if decision_thresholds is not None:
            if (
                len(decision_thresholds)
                == self._episode(load_episode).get_num_classes()
            ):
                parameters_update_second["general"] = {
                    "threshold_value": decision_thresholds
                }
            else:
                raise ValueError(
                    f"The length of the decision thresholds {decision_thresholds} "
                    f"must be equal to the length of the behaviors dictionary "
                    f"{self._episode(load_episode).get_behaviors_dict()}"
                )
        data_param_update = {}
        if data_path is not None:
            data_param_update = {"data_path": data_path}
        if file_paths is not None:
            data_param_update = {"data_path": None, "file_paths": file_paths}
        parameters_update = self._update(parameters_update, {"data": data_param_update})
        if data_path is not None or file_paths is not None:
            general_update = {
                "annotation_type": "none",
                "only_load_annotated": False,
            }
        else:
            general_update = {}
        parameters_update = self._update(parameters_update, {"general": general_update})
        task, parameters = self._make_task(
            episode_name=prediction_name,
            load_episode=load_episode,
            parameters_update=parameters_update,
            parameters_update_second=parameters_update_second,
            load_epoch=load_epoch,
            purpose="prediction",
            task=task,
            behaviors=self.get_behavior_dictionary(load_episode),
        )
        # if data_path is not None or file_paths is not None:
        #     print('SETTING')
        #     task.set_behaviors(self.get_behavior_dictionary(load_episode))
        if mode is None:
            if task.exists("test"):
                mode = "test"
            elif task.exists("val"):
                mode = "val"
            else:
                mode = "train"
        return task, parameters, mode

    def _make_task_training(
        self,
        episode_name: str,
        load_episode: str = None,
        parameters_update: Dict = None,
        load_epoch: int = None,
        load_search: str = None,
        load_parameters: list = None,
        round_to_binary: list = None,
        load_strict: bool = True,
        continuing: bool = False,
        task: TaskDispatcher = None,
        mask_name: str = None,
        throwaway: bool = False,
    ) -> Tuple[TaskDispatcher, Dict, str]:
        """
        Make a `TaskDispatcher` object that will be used to generate a prediction
        """

        if parameters_update is None:
            parameters_update = {}
        if continuing:
            purpose = "continuing"
        else:
            purpose = "training"
        if mask_name is not None:
            mask_name = os.path.join(self._mask_path(), f"{mask_name}.pickle")
        parameters_update_second = {"data": {"real_lens": mask_name}}
        if throwaway:
            parameters_update = self._update(
                parameters_update, {"training": {"normalize": False, "device": "cpu"}}
            )
        return self._make_task(
            episode_name,
            load_episode,
            parameters_update,
            parameters_update_second,
            load_epoch,
            load_search,
            load_parameters,
            round_to_binary,
            purpose,
            task,
            load_strict=load_strict,
        )

    def _make_parameters(
        self,
        episode_name: str,
        load_episode: str = None,
        parameters_update: Dict = None,
        parameters_update_second: Dict = None,
        load_epoch: int = None,
        load_search: str = None,
        load_parameters: list = None,
        round_to_binary: list = None,
        purpose: str = "train",
        load_strict: bool = True,
    ):
        """
        Construct a parameters dictionary
        """

        if parameters_update is None:
            parameters_update = {}
        pars_update = deepcopy(parameters_update)
        if parameters_update_second is None:
            parameters_update_second = {}
        if purpose == "prediction" and "model" in pars_update.keys():
            raise ValueError("Cannot change model parameters after training!")
        if purpose in ["continuing", "prediction"] and load_episode is not None:
            read_parameters = self._read_parameters()
            parameters = self._episodes().load_parameters(load_episode)
            parameters["metrics"] = self._update(
                read_parameters["metrics"], parameters["metrics"]
            )
            parameters["ssl"] = self._update(
                read_parameters["ssl"], parameters.get("ssl", {})
            )
        else:
            parameters = self._read_parameters()
        if "model" in pars_update:
            model_params = pars_update.pop("model")
        else:
            model_params = None
        if "features" in pars_update:
            feat_params = pars_update.pop("features")
        else:
            feat_params = None
        if "augmentations" in pars_update:
            aug_params = pars_update.pop("augmentations")
        else:
            aug_params = None
        parameters = self._update(parameters, pars_update)
        if pars_update.get("general", {}).get("model_name") is not None:
            model_name = parameters["general"]["model_name"]
            parameters["model"] = self._open_yaml(
                os.path.join(self.project_path, "config", "model", f"{model_name}.yaml")
            )
        if pars_update.get("general", {}).get("feature_extraction") is not None:
            feat_name = parameters["general"]["feature_extraction"]
            parameters["features"] = self._open_yaml(
                os.path.join(
                    self.project_path, "config", "features", f"{feat_name}.yaml"
                )
            )
            aug_name = options.extractor_to_transformer[
                parameters["general"]["feature_extraction"]
            ]
            parameters["augmentations"] = self._open_yaml(
                os.path.join(
                    self.project_path, "config", "augmentations", f"{aug_name}.yaml"
                )
            )
        if model_params is not None:
            parameters["model"] = self._update(parameters["model"], model_params)
        if feat_params is not None:
            parameters["features"] = self._update(parameters["features"], feat_params)
        if aug_params is not None:
            parameters["augmentations"] = self._update(
                parameters["augmentations"], aug_params
            )
        if load_search is not None:
            parameters = self._update_with_search(
                parameters, load_search, load_parameters, round_to_binary
            )
        parameters = self._fill(
            parameters,
            episode_name,
            load_episode,
            load_epoch=load_epoch,
            load_strict=load_strict,
            only_load_model=(purpose != "continuing"),
            continuing=(purpose in ["prediction", "continuing"]),
            enforce_split_parameters=(purpose == "prediction"),
        )
        parameters = self._update(parameters, parameters_update_second)
        return parameters

    def _make_task(
        self,
        episode_name: str,
        load_episode: str = None,
        parameters_update: Dict = None,
        parameters_update_second: Dict = None,
        load_epoch: int = None,
        load_search: str = None,
        load_parameters: list = None,
        round_to_binary: list = None,
        purpose: str = "train",
        task: TaskDispatcher = None,
        load_strict: bool = True,
        behaviors: Dict = None,
    ) -> Tuple[TaskDispatcher, Union[CommentedMap, dict]]:
        """
        Make a `TaskDispatcher` object

        The task parameters are read from the config files and then updated with the
        parameters_update dictionary. The model can be either initialized from scratch or loaded from one of the
        previous experiments. All parameters and results are saved in the meta files and can be accessed with the
        list_episodes() function. The train/test/validation split is saved and loaded from a file whenever the
        same split parameters are used. The pre-computed datasets are also saved and loaded whenever the same
        data parameters are used.

        Parameters
        ----------
        episode_name : str
            the name of the episode
        load_episode : str, optional
            the (previously run) episode name to load the model from
        parameters_update : dict, optional
            the dictionary used to update the parameters from the config
        parameters_update_second : dict, optional
            the dictionary used to update the parameters after the automatic fill-out
        load_epoch : int, optional
            the epoch to load (if load_episodes is not None); if not provided, the last epoch is used
        load_search : str, optional
            the hyperparameter search result to load
        load_parameters : list, optional
            a list of string names of the parameters to load from load_search (if not provided, all parameters
            are loaded)
        round_to_binary : list, optional
            a list of string names of the loaded parameters that should be rounded to the nearest power of two
        purpose : {"train", "continuing", "prediction"}
            the purpose of the task object (`"train"` for training from scratch, `"continuing"` for continuing
            the training of an interrupted episode, `"prediction"` for generating a prediction)
        task : TaskDispatcher, optional
            a pre-existing task; if provided, the method will update the task instead of creating a new one
            (this might save time, mainly on dataset loading)

        Returns
        -------
        task : TaskDispatcher
            the `TaskDispatcher` instance
        parameters : dict
            the parameters dictionary that describes the task
        """

        parameters = self._make_parameters(
            episode_name,
            load_episode,
            parameters_update,
            parameters_update_second,
            load_epoch,
            load_search,
            load_parameters,
            round_to_binary,
            purpose,
            load_strict=load_strict,
        )
        if parameters["data"].get("annotation_type", "none") == "none":
            parameters = self._update(
                parameters, {"data": {"behavior_dictionary": behaviors}}
            )
        if task is None:
            task = TaskDispatcher(parameters)
        else:
            task.update_task(parameters)
        self._save_stores(parameters)
        return task, parameters

    def run_episode(
        self,
        episode_name: str,
        load_episode: str = None,
        parameters_update: Dict = None,
        task: TaskDispatcher = None,
        load_epoch: int = None,
        load_search: str = None,
        load_parameters: list = None,
        round_to_binary: list = None,
        load_strict: bool = True,
        n_seeds: int = 1,
        force: bool = False,
        suppress_name_check: bool = False,
        remove_saved_features: bool = False,
        mask_name: str = None,
        autostop_metric: str = None,
        autostop_interval: int = 50,
        autostop_threshold: float = 0.001,
        loading_bar: bool = False,
        trial: Tuple = None,
    ) -> TaskDispatcher:
        """
        Run an episode

        The task parameters are read from the config files and then updated with the
        parameters_update dictionary. The model can be either initialized from scratch or loaded from one of the
        previous experiments. All parameters and results are saved in the meta files and can be accessed with the
        list_episodes() function. The train/test/validation split is saved and loaded from a file whenever the
        same split parameters are used. The pre-computed datasets are also saved and loaded whenever the same
        data parameters are used.

        You can use the autostop parameters to finish training when the parameters are not improving. It will be
        stopped if the average value of `autostop_metric` over the last `autostop_interval` epochs is smaller than
        the average over the previous `autostop_interval` epochs + `autostop_threshold`. For example, if the
        current epoch is 120 and `autostop_interval` is 50, the averages over epochs 70-120 and 20-70 will be compared.

        Parameters
        ----------
        episode_name : str
            the episode name
        load_episode : str, optional
            the (previously run) episode name to load the model from; if the episode has multiple runs,
            the new episode will have the same number of runs, each starting with one of the pre-trained models
        parameters_update : dict, optional
            the dictionary used to update the parameters from the config files
        task : TaskDispatcher, optional
            a pre-existing `TaskDispatcher` object (if provided, the method will update it instead of creating
            a new instance)
        load_epoch : int, optional
            the epoch to load (if load_episodes is not None); if not provided, the last epoch is used
        load_search : str, optional
            the hyperparameter search result to load
        load_parameters : list, optional
            a list of string names of the parameters to load from load_search (if not provided, all parameters
            are loaded)
        round_to_binary : list, optional
            a list of string names of the loaded parameters that should be rounded to the nearest power of two
        load_strict : bool, default True
            if `False`, matching weights will be loaded from `load_episode` and differences in parameter name lists and
            weight shapes will be ignored; otherwise mismatches will prompt a `RuntimeError`
        n_seeds : int, default 1
            the number of runs to perform with different random seeds; if `n_seeds > 1`, the episodes will be named
            `episode_name::seed_index`, e.g. `test_episode::0` and `test_episode::1`
        force : bool, default False
            if `True` and an episode with name `episode_name` already exists, it will be overwritten (use with caution!)
        suppress_name_check : bool, default False
            if `True`, episode names with a double colon are allowed (please don't use this option unless you understand
            why they are usually forbidden)
        remove_saved_features : bool, default False
            if `True`, the dataset will be deleted after training
        mask_name : str, optional
            the name of the real_lens to apply
        autostop_interval : int, default 50
            the number of epochs to average the autostop metric over
        autostop_threshold : float, default 0.001
            the autostop difference threshold
        autostop_metric : str, optional
            the autostop metric (can be any one of the tracked metrics of `'loss'`)
        """

        if type(n_seeds) is not int or n_seeds < 1:
            raise ValueError(
                f"The n_seeds parameter has to be an integer larger than 0; got {n_seeds}"
            )
        if n_seeds > 1 and mask_name is not None:
            raise ValueError("Cannot apply a real_lens with n_seeds > 1")
        self._check_episode_validity(
            episode_name, allow_doublecolon=suppress_name_check, force=force
        )
        load_runs = self._episodes().get_runs(load_episode)
        if len(load_runs) > 1:
            task = self.run_episodes(
                episode_names=[
                    f'{episode_name}::{run.split("::")[-1]}' for run in load_runs
                ],
                load_episodes=load_runs,
                parameters_updates=[parameters_update for _ in load_runs],
                load_epochs=[load_epoch for _ in load_runs],
                load_searches=[load_search for _ in load_runs],
                load_parameters=[load_parameters for _ in load_runs],
                round_to_binary=[round_to_binary for _ in load_runs],
                load_strict=[load_strict for _ in load_runs],
                suppress_name_check=True,
                force=force,
                remove_saved_features=False,
            )
            if remove_saved_features:
                self._remove_stores(
                    {
                        "general": task.general_parameters,
                        "data": task.data_parameters,
                        "features": task.feature_parameters,
                    }
                )
            if n_seeds > 1:
                warnings.warn(
                    f"The n_seeds parameter is disregarded since load_episode={load_episode} has multiple runs"
                )
        elif n_seeds > 1:
            self.run_episodes(
                episode_names=[f"{episode_name}::{i}" for i in range(n_seeds)],
                load_episodes=[load_episode for _ in range(n_seeds)],
                parameters_updates=[parameters_update for _ in range(n_seeds)],
                load_epochs=[load_epoch for _ in range(n_seeds)],
                load_searches=[load_search for _ in range(n_seeds)],
                load_parameters=[load_parameters for _ in range(n_seeds)],
                round_to_binary=[round_to_binary for _ in range(n_seeds)],
                load_strict=[load_strict for _ in range(n_seeds)],
                suppress_name_check=True,
                force=force,
                remove_saved_features=remove_saved_features,
            )
        else:
            print(f"TRAINING {episode_name}")
            try:
                task, parameters = self._make_task_training(
                    episode_name,
                    load_episode,
                    parameters_update,
                    load_epoch,
                    load_search,
                    load_parameters,
                    round_to_binary,
                    continuing=False,
                    task=task,
                    mask_name=mask_name,
                    load_strict=load_strict,
                )
                self._save_episode(
                    episode_name,
                    parameters,
                    task.behaviors_dict(),
                    norm_stats=task.get_normalization_stats(),
                )
                time_start = time.time()
                if trial is not None:
                    trial, metric = trial
                else:
                    trial, metric = None, None
                logs = task.train(
                    autostop_metric=autostop_metric,
                    autostop_interval=autostop_interval,
                    autostop_threshold=autostop_threshold,
                    loading_bar=loading_bar,
                    trial=trial,
                    optimized_metric=metric,
                )
                time_end = time.time()
                time_total = time_end - time_start
                hours = int(time_total // 3600)
                time_total -= hours * 3600
                minutes = int(time_total // 60)
                time_total -= minutes * 60
                seconds = int(time_total)
                training_time = f"{hours}:{minutes:02}:{seconds:02}"
                self._update_episode_results(episode_name, logs, training_time)
                if remove_saved_features:
                    self._remove_stores(parameters)
                print("\n")
                return task

            except Exception as e:
                if isinstance(e, optuna.exceptions.TrialPruned):
                    raise e
                else:
                    # if str(e) != f"The {episode_name} episode name is already in use!":
                    #     self.remove_episode(episode_name)
                    raise RuntimeError(f"Episode {episode_name} could not run")

    def run_episodes(
        self,
        episode_names: List,
        load_episodes: List = None,
        parameters_updates: List = None,
        load_epochs: List = None,
        load_searches: List = None,
        load_parameters: List = None,
        round_to_binary: List = None,
        load_strict: List = None,
        force: bool = False,
        suppress_name_check: bool = False,
        remove_saved_features: bool = False,
    ) -> TaskDispatcher:
        """
        Run multiple episodes in sequence (and re-use previously loaded information)

        For each episode, the task parameters are read from the config files and then updated with the
        parameter_update dictionary. The model can be either initialized from scratch or loaded from one of the
        previous experiments. All parameters and results are saved in the meta files and can be accessed with the
        list_episodes() function. The train/test/validation split is saved and loaded from a file whenever the
        same split parameters are used. The pre-computed datasets are also saved and loaded whenever the same
        data parameters are used.

        Parameters
        ----------
        episode_names : list
            a list of strings of episode names
        load_episodes : list, optional
            a list of strings of (previously run) episode names to load the model from; if the episode has multiple runs,
            the new episode will have the same number of runs, each starting with one of the pre-trained models
        parameters_updates : list, optional
            a list of dictionaries used to update the parameters from the config
        load_epochs : list, optional
            a list of integers used to specify the epoch to load (if load_episodes is not None)
        load_searches : list, optional
            a list of strings of hyperparameter search results to load
        load_parameters : list, optional
            a list of lists of string names of the parameters to load from the searches
        round_to_binary : list, optional
            a list of string names of the loaded parameters that should be rounded to the nearest power of two
        load_strict : list, optional
            a list of boolean values specifying weight loading policy: if `False`, matching weights will be loaded from
            the corresponding episode and differences in parameter name lists and
            weight shapes will be ignored; otherwise mismatches will prompt a `RuntimeError` (by default `True` for
            every episode)
        force : bool, default False
            if `True` and an episode name is already taken, it will be overwritten (use with caution!)
        suppress_name_check : bool, default False
            if `True`, episode names with a double colon are allowed (please don't use this option unless you understand
            why they are usually forbidden)
        remove_saved_features : bool, default False
            if `True`, the dataset will be deleted after training
        """

        task = None
        if load_searches is None:
            load_searches = [None for _ in episode_names]
        if load_episodes is None:
            load_episodes = [None for _ in episode_names]
        if parameters_updates is None:
            parameters_updates = [None for _ in episode_names]
        if load_parameters is None:
            load_parameters = [None for _ in episode_names]
        if load_epochs is None:
            load_epochs = [None for _ in episode_names]
        if load_strict is None:
            load_strict = [True for _ in episode_names]
        for (
            parameters_update,
            episode_name,
            load_episode,
            load_epoch,
            load_search,
            load_parameters_list,
            load_strict_value,
        ) in zip(
            parameters_updates,
            episode_names,
            load_episodes,
            load_epochs,
            load_searches,
            load_parameters,
            load_strict,
        ):
            task = self.run_episode(
                episode_name,
                load_episode,
                parameters_update,
                task,
                load_epoch,
                load_search,
                load_parameters_list,
                round_to_binary,
                load_strict_value,
                suppress_name_check=suppress_name_check,
                force=force,
                remove_saved_features=remove_saved_features,
            )
        return task

    def continue_episode(
        self,
        episode_name: str,
        num_epochs: int = None,
        task: TaskDispatcher = None,
        n_seeds: int = 1,
        remove_saved_features: bool = False,
        device: str = "cuda",
        num_cpus: int = None,
    ) -> TaskDispatcher:
        """
        Load an older episode and continue running from the latest checkpoint

        All parameters as well as the model and optimizer state dictionaries are loaded from the episode.

        Parameters
        ----------
        episode_name : str
            the name of the episode to continue
        num_epochs : int, optional
            the new number of epochs
        task : TaskDispatcher, optional
            a pre-existing task; if provided, the method will update the task instead of creating a new one
            (this might save time, mainly on dataset loading)
        result_average_interval : int, default 5
            the metric are averaged over the last result_average_interval to be stored in the episodes meta file
            and displayed by list_episodes() function (the full log is still always available)
        n_seeds : int, default 1
            the number of runs to perform; if `n_seeds > 1`, the episodes will be named `episode_name::run_index`, e.g.
            `test_episode::0` and `test_episode::1`
        remove_saved_features : bool, default False
            if `True`, pre-computed features will be deleted after the run
        device : str, default "cuda"
            the torch device to use
        """

        runs = self._episodes().get_runs(episode_name)
        for run in runs:
            print(f"TRAINING {run}")
            if num_epochs is None and not self._episode(run).unfinished():
                continue
            parameters_update = {
                "training": {
                    "num_epochs": num_epochs,
                    "device": device,
                },
                "general": {"num_cpus": num_cpus},
            }
            task, parameters = self._make_task_training(
                run,
                load_episode=run,
                parameters_update=parameters_update,
                continuing=True,
                task=task,
            )
            time_start = time.time()
            logs = task.train()
            time_end = time.time()
            old_time = self._training_time(run)
            if not np.isnan(old_time):
                time_end += old_time
                time_total = time_end - time_start
                hours = int(time_total // 3600)
                time_total -= hours * 3600
                minutes = int(time_total // 60)
                time_total -= minutes * 60
                seconds = int(time_total)
                training_time = f"{hours}:{minutes:02}:{seconds:02}"
            else:
                training_time = np.nan
            self._save_episode(
                run,
                parameters,
                task.behaviors_dict(),
                suppress_validation=True,
                training_time=training_time,
                norm_stats=task.get_normalization_stats(),
            )
            self._update_episode_results(run, logs)
            print("\n")
        if len(runs) < n_seeds:
            for i in range(len(runs), n_seeds):
                self.run_episode(
                    f"{episode_name}::{i}",
                    parameters_update=self._episodes().load_parameters(runs[0]),
                    task=task,
                    suppress_name_check=True,
                )
        if remove_saved_features:
            self._remove_stores(parameters)
        return task

    def run_default_hyperparameter_search(
        self,
        search_name: str,
        model_name: str = None,
        metric: str = "f1",
        best_n: int = 3,
        direction: str = "maximize",
        load_episode: str = None,
        load_epoch: int = None,
        load_strict: bool = True,
        prune: bool = True,
        force: bool = False,
        remove_saved_features: bool = False,
        overlap: float = 0,
        num_epochs: int = 50,
        test_frac: float = 0,
        n_trials=150,
        device: str = None,
    ):
        """
        Run an optuna hyperparameter search with default parameters for a model

        For the vast majority of cases, optimizing the default parameters should be enough.
        Check out `dlc2action.options.model_hyperparameters` for the lists of parameters.
        There are also options to set overlap, test fraction and number of epochs parameters for the search without
        modifying the project config files. However, if you want something more complex, look into
        `Project.run_hyperparameter_search`.

        The task parameters are read from the config files and updated with the parameters_update dictionary.
        The model can be either initialized from scratch or loaded from a previously run episode.
        For each trial, the objective metric is averaged over a few best epochs.

        Parameters
        ----------
        search_name : str
            the name of the search to store it in the meta files and load in run_episode
        model_name : str, optional
            the name of the model (by default loaded from the project settings, see `project.help('models')` for options)
        metric : str, default f1
            the metric to maximize/minimize (see direction); if the metric has an `"average"` parameter and it is set to
            `"none"` in the config files, it will be reset to `"macro"` for the search; see `project.help('metrics')` for options
        n_trials : int, default 20
            the number of optimization trials to run
        best_n : int, default 1
            the number of epochs to average the metric; if 0, the last value is taken
        parameters_update : dict, optional
            the parameters update dictionary
        direction : {'maximize', 'minimize'}
            optimization direction
        load_episode : str, optional
            the name of the episode to load the model from
        load_epoch : int, optional
            the epoch to load the model from (if not provided, the last checkpoint is used)
        prune : bool, default False
            if `True`, experiments where the optimized metric is improving too slowly will be terminated
            (with optuna HyperBand pruner)
        force : bool, default False
            if `True`, existing searches with the same name will be overwritten
        remove_saved_features : bool, default False
            if `True`, pre-computed features will be deleted after each run (if the data parameters change)
        device : str, optional
            cuda:{i} or cpu, if not given it is read from the default parameters

        Returns
        -------
        dict
            a dictionary of best parameters
        """

        if model_name is None:
            model_name = self._read_parameters()["general"]["model_name"]
        if model_name not in options.model_hyperparameters:
            raise ValueError(
                f"There is no default search space for {model_name}! Please choose from {options.model_hyperparameters.keys()} or try project.run_hyperparameter_search()"
            )
        pars = {
            "general": {
                "overlap": overlap,
                "model_name": model_name,
                "metric_functions": {metric},
            },
            "training": {"num_epochs": num_epochs},
        }
        if test_frac is not None:
            pars["training"]["test_frac"] = test_frac
        if not metric.split("_")[-1].isnumeric():
            project_pars = self._read_parameters()
            if project_pars["metrics"][metric].get("average") == "none":
                pars["metrics"] = {metric: {"average": "macro"}}
        if device is not None:
            pars["training"]["device"] = device
        return self.run_hyperparameter_search(
            search_name=search_name,
            search_space=options.model_hyperparameters[model_name],
            metric=metric,
            n_trials=n_trials,
            best_n=best_n,
            parameters_update=pars,
            direction=direction,
            load_episode=load_episode,
            load_epoch=load_epoch,
            load_strict=load_strict,
            prune=prune,
            force=force,
            remove_saved_features=remove_saved_features,
        )

    def run_hyperparameter_search(
        self,
        search_name: str,
        search_space: Dict,
        metric: str = "f1",
        n_trials: int = 20,
        best_n: int = 1,
        parameters_update: Dict = None,
        direction: str = "maximize",
        load_episode: str = None,
        load_epoch: int = None,
        load_strict: bool = True,
        prune: bool = False,
        force: bool = False,
        remove_saved_features: bool = False,
    ) -> Dict:
        """
        Run an optuna hyperparameter search

        For a simpler function that fits most use cases, check out `Project.run_default_hyperparameter_search()`.

        To use a default search space with this method, import `dlc2action.options.model_hyperparameters`. It is
        a dictionary where keys are model names and values are default search spaces.

        The task parameters are read from the config files and updated with the parameters_update dictionary.
        The model can be either initialized from scratch or loaded from a previously run episode.
        For each trial, the objective metric is averaged over a few best epochs.

        Parameters
        ----------
        search_name : str
            the name of the search to store it in the meta files and load in run_episode
        search_space : dict
            a dictionary representing the search space; of this general structure:
            {'group/param_name': ('float/int/float_log/int_log', start, end),
            'group/param_name': ('categorical', [choices])}, e.g.
            {'data/overlap': ('int', 5, 100), 'training/lr': ('float_log', 1e-4, 1e-2),
            'data/feature_extraction': ('categorical', ['kinematic', 'bones'])};
        metric : str, default f1
            the metric to maximize/minimize (see direction)
        n_trials : int, default 20
            the number of optimization trials to run
        best_n : int, default 1
            the number of epochs to average the metric; if 0, the last value is taken
        parameters_update : dict, optional
            the parameters update dictionary
        direction : {'maximize', 'minimize'}
            optimization direction
        load_episode : str, optional
            the name of the episode to load the model from
        load_epoch : int, optional
            the epoch to load the model from (if not provided, the last checkpoint is used)
        prune : bool, default False
            if `True`, experiments where the optimized metric is improving too slowly will be terminated
            (with optuna HyperBand pruner)
        force : bool, default False
            if `True`, existing searches with the same name will be overwritten
        remove_saved_features : bool, default False
            if `True`, pre-computed features will be deleted after each run (if the data parameters change)

        Returns
        -------
        dict
            a dictionary of best parameters
        """

        self._check_search_validity(search_name, force=force)
        print(f"SEARCH {search_name}")
        self.remove_episode(f"_{search_name}")
        if parameters_update is None:
            parameters_update = {}
        parameters_update = self._update(
            parameters_update, {"general": {"metric_functions": {metric}}}
        )
        parameters = self._make_parameters(
            f"_{search_name}",
            load_episode,
            parameters_update,
            parameters_update_second={"training": {"model_save_path": None}},
            load_epoch=load_epoch,
            load_strict=load_strict,
        )
        task = None

        if prune:
            pruner = optuna.pruners.HyperbandPruner()
        else:
            pruner = optuna.pruners.NopPruner()
        study = optuna.create_study(direction=direction, pruner=pruner)
        runner = _Runner(
            search_space=search_space,
            load_episode=load_episode,
            load_epoch=load_epoch,
            metric=metric,
            average=best_n,
            task=task,
            remove_saved_features=remove_saved_features,
            project=self,
            search_name=search_name,
        )
        study.optimize(lambda trial: runner.run(trial, parameters), n_trials=n_trials)
        search_path = self._search_path(search_name)
        os.mkdir(search_path)
        fig = optuna.visualization.plot_contour(study)
        plotly.offline.plot(
            fig, filename=os.path.join(search_path, f"{search_name}_contour.html")
        )
        fig = optuna.visualization.plot_param_importances(study)
        plotly.offline.plot(
            fig, filename=os.path.join(search_path, f"{search_name}_importances.html")
        )
        best_params = study.best_params
        best_value = study.best_value
        self._save_search(
            search_name,
            parameters,
            n_trials,
            best_params,
            best_value,
            metric,
            search_space,
        )
        self.remove_episode(f"_{search_name}")
        runner.clean()
        print(f"best parameters: {best_params}")
        print("\n")
        return best_params

    def run_prediction(
        self,
        prediction_name: str,
        episode_names: List,
        load_epochs: List = None,
        parameters_update: Dict = None,
        augment_n: int = 10,
        data_path: str = None,
        mode: str = "all",
        file_paths: Set = None,
        remove_saved_features: bool = False,
        submission: bool = False,
        frame_number_map_file: str = None,
        force: bool = False,
        embedding: bool = False,
    ) -> None:
        """
        Load models from previously run episodes to generate a prediction

        The probabilities predicted by the models are averaged.
        Unless `submission` is `True`, the prediction results are saved as a pickled dictionary in the project_name/results/predictions folder
        under the {episode_name}_{load_epoch}.pickle name. The file is a nested dictionary where the first-level
        keys are the video ids, the second-level keys are the clip ids (like individual names) and the values
        are the prediction arrays.

        Parameters
        ----------
        prediction_name : str
            the name of the prediction
        episode_names : list
            a list of string episode names to load the models from
        load_epochs : list, optional
            a list of integer epoch indices to load the model from; if None, the last ones are used
        parameters_update : dict, optional
            a dictionary of parameter updates
        augment_n : int, default 10
            the number of augmentations to average over
        data_path : str, optional
            the data path to run the prediction for
        mode : {'all', 'test', 'val', 'train'}
            the subset of the data to make the prediction for (forced to 'all' if data_path is not None)
        file_paths : set, optional
            a set of string file paths (data with all prefixes + feature files, in any order) to run the prediction
            for
        remove_saved_features : bool, default False
            if `True`, pre-computed features will be deleted
        submission : bool, default False
            if `True`, a MABe-22 style submission file is generated
        frame_number_map_file : str, optional
            path to the frame number map file
        force : bool, default False
            if `True`, existing prediction with this name will be overwritten
        """

        self._check_prediction_validity(prediction_name, force=force)
        print(f"PREDICTION {prediction_name}")
        if submission:
            task = ...
            # TODO: add submission option to _make_prediction
            predicted = task.generate_submission(
                frame_number_map_file=frame_number_map_file,
                dataset=mode,
                augment_n=augment_n,
            )
            folder = os.path.join(
                self.project_path,
                "results",
                "predictions",
                f"{prediction_name}",
            )
            filename = os.path.join(folder, f"{prediction_name}.npy")
            np.save(filename, predicted, allow_pickle=True)
        else:
            try:
                (
                    task,
                    parameters,
                    mode,
                    prediction,
                    inference_time,
                ) = self._make_prediction(
                    prediction_name,
                    episode_names,
                    load_epochs,
                    parameters_update,
                    data_path,
                    file_paths,
                    mode,
                    augment_n,
                    evaluate=False,
                    embedding=embedding,
                )
                predicted = task.dataset(mode).generate_full_length_prediction(
                    prediction
                )
            except ValueError:
                (
                    task,
                    parameters,
                    mode,
                    predicted,
                    inference_time,
                ) = self._aggregate_predictions(
                    prediction_name,
                    episode_names,
                    load_epochs,
                    parameters_update,
                    data_path,
                    file_paths,
                    mode,
                    augment_n,
                    evaluate=False,
                    embedding=embedding,
                )
            folder = self.prediction_path(prediction_name)
            os.mkdir(folder)
            for video_id, prediction in predicted.items():
                with open(
                    os.path.join(
                        folder, video_id + f"_{prediction_name}_prediction.pickle"
                    ),
                    "wb",
                ) as f:
                    prediction["min_frames"], prediction["max_frames"] = task.dataset(
                        mode
                    ).get_min_max_frames(video_id)
                    behavior_indices = sorted(
                        [key for key in task.behaviors_dict() if key != -100]
                    )
                    prediction["behaviors"] = [
                        task.behaviors_dict()[key] for key in behavior_indices
                    ]
                    pickle.dump(prediction, f)
        if remove_saved_features:
            self._remove_stores(parameters)
        self._save_prediction(
            prediction_name,
            parameters,
            task.behaviors_dict(),
            embedding,
            inference_time,
        )
        print("\n")

    def evaluate_prediction(
        self,
        prediction_name: str,
        parameters_update: Dict = None,
        data_path: str = None,
        file_paths: Set = None,
        mode: str = None,
        remove_saved_features: bool = False,
    ) -> Tuple[float, dict]:

        with open(
            os.path.join(
                self.project_path, "results", "predictions", f"{prediction_name}.pickle"
            ),
            "rb",
        ) as f:
            prediction = pickle.load(f)
        if parameters_update is None:
            parameters_update = {}
        parameters_update = self._update(
            self._predictions().load_parameters(prediction_name), parameters_update
        )
        parameters_update.pop("model")
        task, parameters, mode = self._make_task_prediction(
            "_",
            load_episode=None,
            parameters_update=parameters_update,
            data_path=data_path,
            file_paths=file_paths,
            mode=mode,
        )
        results = task.evaluate_prediction(prediction, data=mode)
        if remove_saved_features:
            self._remove_stores(parameters)
        print("\n")
        return results

    def evaluate(
        self,
        episode_names: List,
        load_epochs: List = None,
        augment_n: int = 0,
        data_path: str = None,
        file_paths: Set = None,
        mode: str = None,
        parameters_update: Dict = None,
        multiple_episode_policy: str = "average",
        remove_saved_features: bool = False,
        skip_updating_meta: bool = True,
    ) -> Dict:
        """
        Load one or several models from previously run episodes to make an evaluation

        By default it will run on the test (or validation, if there is no test) subset of the project dataset.

        Parameters
        ----------
        episode_names : list
            a list of string episode names to load the models from
        load_epochs : list, optional
            a list of integer epoch indices to load the model from; if None, the last ones are used
        augment_n : int, default 0
            the number of augmentations to average over
        data_path : str, optional
            the data path to run the prediction for
        file_paths : set, optional
            a set of files to run the prediction for
        mode : {'test', 'val', 'train', 'all'}
            the subset of the data to make the prediction for (forced to 'all' if data_path is not None;
            by default 'test' if test subset is not empty and 'val' otherwise)
        parameters_update : dict, optional
            a dictionary with parameter updates (cannot change model parameters)
        remove_saved_features : bool, default False
            if `True`, the dataset will be deleted

        Returns
        -------
        metric : dict
            a dictionary of average values of metric functions
        """

        names = []
        for episode_name in episode_names:
            names += self._episodes().get_runs(episode_name)
        if len(set(episode_names)) == 1:
            print(f"EVALUATION {episode_names[0]}")
        else:
            print(f"EVALUATION {episode_names}")
        if len(names) > 1:
            evaluate = True
        else:
            evaluate = False
        if multiple_episode_policy == "average":
            try:
                (
                    task,
                    parameters,
                    mode,
                    prediction,
                    inference_time,
                ) = self._make_prediction(
                    "_",
                    episode_names,
                    load_epochs,
                    parameters_update,
                    mode=mode,
                    data_path=data_path,
                    file_paths=file_paths,
                    augment_n=augment_n,
                    evaluate=evaluate,
                )
            except:
                (
                    task,
                    parameters,
                    mode,
                    prediction,
                    inference_time,
                ) = self._aggregate_predictions(
                    "_",
                    episode_names,
                    load_epochs,
                    parameters_update,
                    mode=mode,
                    data_path=data_path,
                    file_paths=file_paths,
                    augment_n=augment_n,
                    evaluate=evaluate,
                )
            print("AGGREGATED:")
            _, results = task.evaluate_prediction(prediction, data=mode)
            if len(names) == 1 and mode == "val" and not skip_updating_meta:
                self._update_episode_metrics(names[0], results)
        elif multiple_episode_policy == "statistics":
            values = defaultdict(lambda: [])
            task = None
            for name in names:
                (
                    task,
                    parameters,
                    mode,
                    prediction,
                    inference_time,
                ) = self._make_prediction(
                    "_",
                    [name],
                    load_epochs,
                    parameters_update,
                    mode=mode,
                    data_path=data_path,
                    file_paths=file_paths,
                    augment_n=augment_n,
                    evaluate=evaluate,
                    task=task,
                )
                _, metrics = task.evaluate_prediction(prediction, data=mode)
                for name, value in metrics.items():
                    values[name].append(value)
                if mode == "val" and not skip_updating_meta:
                    self._update_episode_metrics(name, metrics)
            results = defaultdict(lambda: {})
            mean_string = ""
            std_string = ""
            for key, value_list in values.items():
                results[key]["mean"] = np.mean(value_list)
                results[key]["std"] = np.std(value_list)
                mean_string += f"{key} {np.mean(value_list):.3f}, "
                std_string += f"{key} {np.std(value_list):.3f}, "
            print("MEAN:")
            print(mean_string)
            print("STD:")
            print(std_string)
        else:
            raise ValueError(
                f"The {multiple_episode_policy} multiple episode policy is not recognized; please choose "
                f"from ['average', 'statistics']"
            )
        if len(names) > 0 and remove_saved_features:
            self._remove_stores(parameters)
        print(f"Inference time: {inference_time}")
        print("\n")
        return results

    def _generate_similarity_score(
        self,
        prediction_name: str,
        target_video_id: str,
        target_clip: str,
        target_start: int,
        target_end: int,
    ) -> Dict:
        with open(
            os.path.join(
                self.project_path,
                "results",
                "predictions",
                f"{prediction_name}.pickle",
            ),
            "rb",
        ) as f:
            prediction = pickle.load(f)
        target = prediction[target_video_id][target_clip][:, target_start:target_end]
        score_dict = defaultdict(lambda: {})
        for video_id in prediction:
            for clip_id in prediction[video_id]:
                score_dict[video_id][clip_id] = torch.cdist(
                    target.T, prediction[video_id][score_dict].T
                ).min(0)
        return score_dict

    def _suggest_intervals_from_dict(self, score_dict, min_length, n_intervals) -> Dict:
        interval_address = {}
        interval_value = {}
        s = 0
        n = 0
        for video_id, video_dict in score_dict.items():
            for clip_id, value in video_dict.items():
                s += value.mean()
                n += 1
        mean_value = s / n
        alpha = 1.75
        for it in range(10):
            id = 0
            interval_address = {}
            interval_value = {}
            for video_id, video_dict in score_dict.items():
                for clip_id, value in video_dict.items():
                    res_indices_start, res_indices_end = apply_threshold(
                        value,
                        threshold=(2 - alpha * (0.9**it)) * mean_value,
                        low=True,
                        error_mask=None,
                        min_frames=min_length,
                        smooth_interval=0,
                    )
                    for start, end in zip(res_indices_start, res_indices_end):
                        interval_address[id] = [video_id, clip_id, start, end]
                        interval_value[id] = score_dict[video_id][clip_id][
                            start:end
                        ].mean()
                        id += 1
            if len(interval_address) >= n_intervals:
                break
        if len(interval_address) < n_intervals:
            warnings.warn(
                f"Could not get {n_intervals} intervals from the data, saving the result with {len(interval_address)} intervals"
            )
        sorted_intervals = sorted(
            interval_value.items(), key=lambda x: x[1], reverse=True
        )
        output_intervals = [
            interval_address[x[0]]
            for x in sorted_intervals[: min(len(sorted_intervals), n_intervals)]
        ]
        output = defaultdict(lambda: [])
        for video_id, clip_id, start, end in output_intervals:
            output[video_id].append([start, end, clip_id])
        return output

    def list_episodes(
        self,
        episode_names: List = None,
        value_filter: str = "",
        display_parameters: List = None,
        print_results: bool = True,
    ) -> pd.DataFrame:
        """
        Get a filtered pandas dataframe with episode metadata

        Parameters
        ----------
        episode_names : list
            a list of strings of episode names
        value_filter : str
            a string of filters to apply; of this general structure:
            'group_name1/par_name1::(</>/<=/>=/=)value1,group_name2/par_name2::(</>/<=/>=/=)value2', e.g.
            'data/overlap::=50,results/recall::>0.5,data/feature_extraction::=kinematic,meta/training_time::>=00:00:10'
        display_parameters : list
            list of parameters to display (e.g. ['data/overlap', 'results/recall'])
        print_results : bool, default True
            if True, the result will be printed to standard output

        Returns
        -------
        pd.DataFrame
            the filtered dataframe
        """

        episodes = self._episodes().list_episodes(
            episode_names, value_filter, display_parameters
        )
        if print_results:
            print("TRAINING EPISODES")
            print(episodes)
            print("\n")
        return episodes

    def list_predictions(
        self,
        episode_names: List = None,
        value_filter: str = "",
        display_parameters: List = None,
        print_results: bool = True,
    ) -> pd.DataFrame:
        """
        Get a filtered pandas dataframe with prediction metadata

        Parameters
        ----------
        episode_names : list
            a list of strings of episode names
        value_filter : str
            a string of filters to apply; of this general structure:
            'group_name1/par_name1:(<>=)value1,group_name2/par_name2:(<>=)value2', e.g.
            'data/overlap:=50,results/recall:>0.5,data/feature_extraction:=kinematic'
        display_parameters : list
            list of parameters to display (e.g. ['data/overlap', 'results/recall'])
        print_results : bool, default True
            if True, the result will be printed to standard output

        Returns
        -------
        pd.DataFrame
            the filtered dataframe
        """

        predictions = self._predictions().list_episodes(
            episode_names, value_filter, display_parameters
        )
        if print_results:
            print("PREDICTIONS")
            print(predictions)
            print("\n")
        return predictions

    def list_searches(
        self,
        search_names: List = None,
        value_filter: str = "",
        display_parameters: List = None,
        print_results: bool = True,
    ) -> pd.DataFrame:
        """
        Get a filtered pandas dataframe with hyperparameter search metadata

        Parameters
        ----------
        search_names : list
            a list of strings of search names
        value_filter : str
            a string of filters to apply; of this general structure:
            'group_name1/par_name1:(<>=)value1,group_name2/par_name2:(<>=)value2', e.g.
            'data/overlap:=50,results/recall:>0.5,data/feature_extraction:=kinematic'
        display_parameters : list
            list of parameters to display (e.g. ['data/overlap', 'results/recall'])
        print_results : bool, default True
            if True, the result will be printed to standard output

        Returns
        -------
        pd.DataFrame
            the filtered dataframe
        """

        searches = self._searches().list_episodes(
            search_names, value_filter, display_parameters
        )
        if print_results:
            print("SEARCHES")
            print(searches)
            print("\n")
        return searches

    def get_best_parameters(
        self,
        search_name: str,
        round_to_binary: List = None,
    ):
        params, model = self._searches().get_best_params(
            search_name, round_to_binary=round_to_binary
        )
        params = self._update(params, {"general": {"model_name": model}})
        return params

    def list_best_parameters(
        self, search_name: str, print_results: bool = True
    ) -> Dict:
        """
        Get the raw dictionary of best parameters found by a search

        Parameters
        ----------
        search_name : str
            the name of the search
        print_results : bool, default True
            if True, the result will be printed to standard output

        Returns
        -------
        best_params : dict
            a dictionary of the best parameters where the keys are in '{group}/{name}' format
        """

        params = self._searches().get_best_params_raw(search_name)
        if print_results:
            print(f"SEARCH RESULTS {search_name}")
            for k, v in params.items():
                print(f"{k}: {v}")
            print("\n")
        return params

    def plot_episodes(
        self,
        episode_names: List,
        metrics: List,
        modes: List = None,
        title: str = None,
        episode_labels: List = None,
        save_path: str = None,
        add_hlines: List = None,
        epoch_limits: List = None,
        colors: List = None,
        add_highpoint_hlines: bool = False,
    ) -> None:
        """
        Plot episode training curves

        Parameters
        ----------
        episode_names : list
            a list of episode names to plot; to plot to episodes in one line combine them in a list
            (e.g. ['episode1', ['episode2', 'episode3']] to plot episode2 and episode3 as one experiment)
        metrics : list
            a list of metric to plot
        modes : list, optional
            a list of modes to plot ('train' and/or 'val'; `['val']` by default)
        title : str, optional
            title for the plot
        episode_labels : list, optional
            a list of strings used to label the curves (has to be the same length as episode_names)
        save_path : str, optional
            the path to save the resulting plot
        add_hlines : list, optional
            a list of float values (or (value, label) tuples) to mark with horizontal lines
        colors: list, optional
            a list of matplotlib colors
        add_highpoint_hlines : bool, default False
            if `True`, horizontal lines will be added at the highest value of each episode
        """

        if modes is None:
            modes = ["val"]
        if add_hlines is None:
            add_hlines = []
        logs = []
        epochs = []
        labels = []
        if episode_labels is not None:
            assert len(episode_labels) == len(episode_names)
        for name_i, name in enumerate(episode_names):
            log_params = product(metrics, modes)
            for metric, mode in log_params:
                if episode_labels is not None:
                    label = episode_labels[name_i]
                else:
                    label = deepcopy(name)
                if len(modes) != 1:
                    label += f"_{mode}"
                if len(metrics) != 1:
                    label += f"_{metric}"
                labels.append(label)
                if isinstance(name, Iterable) and not isinstance(name, str):
                    epoch_list = defaultdict(lambda: [])
                    multi_logs = defaultdict(lambda: [])
                    for i, n in enumerate(name):
                        runs = self._episodes().get_runs(n)
                        if len(runs) > 1:
                            for run in runs:
                                index = run.split("::")[-1]
                                if multi_logs[index] == []:
                                    if multi_logs["null"] is None:
                                        raise RuntimeError(
                                            "The run indices are not consistent across episodes!"
                                        )
                                    else:
                                        multi_logs[index] += multi_logs["null"]
                                multi_logs[index] += list(
                                    self._episode(run).get_metric_log(mode, metric)
                                )
                                start = (
                                    0
                                    if len(epoch_list[index]) == 0
                                    else epoch_list[index][-1]
                                )
                                epoch_list[index] += [
                                    x + start
                                    for x in self._episode(run).get_epoch_list(mode)
                                ]
                            multi_logs["null"] = None
                        else:
                            if len(multi_logs.keys()) > 1:
                                raise RuntimeError(
                                    "Cannot plot a single-run episode after a multi-run episode!"
                                )
                            multi_logs["null"] += list(
                                self._episode(n).get_metric_log(mode, metric)
                            )
                            start = (
                                0
                                if len(epoch_list["null"]) == 0
                                else epoch_list["null"][-1]
                            )
                            epoch_list["null"] += [
                                x + start for x in self._episode(n).get_epoch_list(mode)
                            ]
                    if len(multi_logs.keys()) == 1:
                        log = multi_logs["null"]
                        epochs.append(epoch_list["null"])
                    else:
                        log = tuple([v for k, v in multi_logs.items() if k != "null"])
                        epochs.append(
                            tuple([v for k, v in epoch_list.items() if k != "null"])
                        )
                else:
                    runs = self._episodes().get_runs(name)
                    if len(runs) > 1:
                        log = []
                        for run in runs:
                            tracked_metrics = self._episode(run).get_metrics()
                            if metric in tracked_metrics:
                                log.append(
                                    list(
                                        self._episode(run).get_metric_log(mode, metric)
                                    )
                                )
                            else:
                                relevant = []
                                for m in tracked_metrics:
                                    m_split = m.split("_")
                                    if (
                                        "_".join(m_split[:-1]) == metric
                                        and m_split[-1].isnumeric()
                                    ):
                                        relevant.append(m)
                                if len(relevant) == 0:
                                    raise ValueError(
                                        f"The {metric} metric was not tracked at {run}"
                                    )
                                arr = 0
                                for m in relevant:
                                    arr += self._episode(run).get_metric_log(mode, m)
                                arr /= len(relevant)
                                log.append(list(arr))
                        log = tuple(log)
                        epochs.append(
                            tuple(
                                [
                                    self._episode(run).get_epoch_list(mode)
                                    for run in runs
                                ]
                            )
                        )
                    else:
                        tracked_metrics = self._episode(name).get_metrics()
                        if metric in tracked_metrics:
                            log = list(self._episode(name).get_metric_log(mode, metric))
                        else:
                            relevant = []
                            for m in tracked_metrics:
                                m_split = m.split("_")
                                if (
                                    "_".join(m_split[:-1]) == metric
                                    and m_split[-1].isnumeric()
                                ):
                                    relevant.append(m)
                            if len(relevant) == 0:
                                raise ValueError(
                                    f"The {metric} metric was not tracked at {name}"
                                )
                            arr = 0
                            for m in relevant:
                                arr += self._episode(name).get_metric_log(mode, m)
                            arr /= len(relevant)
                            log = list(arr)
                        epochs.append(self._episode(name).get_epoch_list(mode))
                logs.append(log)
        # if episode_labels is not None:
        #     print(f'{len(episode_labels)=}, {len(logs)=}')
        #     if len(episode_labels) != len(logs):

        #         raise ValueError(
        #             f"The length of episode_labels ({len(episode_labels)}) has to be equal to the length of "
        #             f"curves ({len(logs)})!"
        #         )
        #     else:
        #         labels = episode_labels
        if colors is None:
            colors = cm.rainbow(np.linspace(0, 1, len(logs)))
        if len(colors) != len(logs):
            raise ValueError(
                "The length of colors has to be equal to the length of curves (metrics * modes * episode_names)!"
            )
        plt.figure()
        length = 0
        for log, label, color, epoch_list in zip(logs, labels, colors, epochs):
            if type(log) is list:
                if len(log) > length:
                    length = len(log)
                plt.plot(
                    epoch_list,
                    log,
                    label=label,
                    color=color,
                )
                if add_highpoint_hlines:
                    plt.axhline(np.max(log), linestyle="dashed", color=color)
            else:
                for l, xx in zip(log, epoch_list):
                    if len(l) > length:
                        length = len(l)
                    plt.plot(
                        xx,
                        l,
                        color=color,
                        alpha=0.2,
                    )
                if not all([len(x) == len(log[0]) for x in log]):
                    warnings.warn(
                        f"Got logs with unequal lengths in parallel runs for {label}"
                    )
                    log = list(log)
                    epoch_list = list(epoch_list)
                    for i, x in enumerate(epoch_list):
                        to_remove = []
                        for j, y in enumerate(x[1:]):
                            if y <= x[j - 1]:
                                y_ind = x.index(y)
                                to_remove += list(range(y_ind, j))
                        epoch_list[i] = [
                            y for j, y in enumerate(x) if j not in to_remove
                        ]
                        log[i] = [y for j, y in enumerate(log[i]) if j not in to_remove]
                    length = min([len(x) for x in log])
                    for i in range(len(log)):
                        log[i] = log[i][:length]
                        epoch_list[i] = epoch_list[i][:length]
                    if not all([x == epoch_list[0] for x in epoch_list]):
                        raise RuntimeError(
                            f"Got different epoch indices in parallel runs for {label}"
                        )
                mean = np.array(log).mean(0)
                plt.plot(
                    epoch_list[0],
                    mean,
                    label=label,
                    color=color,
                )
                if add_highpoint_hlines:
                    plt.axhline(np.max(mean), linestyle="dashed", color=color)
        for x in add_hlines:
            label = None
            if isinstance(x, Iterable):
                x, label = x
            plt.axhline(x, label=label)
            plt.xlim((0, length))

        plt.legend()
        plt.xlabel("epochs")
        if len(metrics) == 1:
            plt.ylabel(metrics[0])
        else:
            plt.ylabel("value")
        if title is None:
            if len(episode_names) == 1:
                title = episode_names[0]
            elif len(metrics) == 1:
                title = metrics[0]
        if epoch_limits is not None:
            plt.xlim(epoch_limits)
        if title is not None:
            plt.title(title)
        plt.show()
        if save_path is not None:
            plt.savefig(save_path)

    def update_parameters(
        self,
        parameters_update: Dict = None,
        load_search: str = None,
        load_parameters: List = None,
        round_to_binary: List = None,
    ) -> None:
        """
        Update the parameters in the project config files

        Parameters
        ----------
        parameters_update : dict, optional
            a dictionary of parameter updates
        load_search : str, optional
            the name of hyperparameter search results to load to config
        load_parameters : list, optional
            a list of lists of string names of the parameters to load from the searches
        round_to_binary : list, optional
            a list of string names of the loaded parameters that should be rounded to the nearest power of two
        """

        keys = [
            "general",
            "losses",
            "metrics",
            "ssl",
            "training",
            "data",
        ]
        parameters = self._read_parameters(catch_blanks=False)
        if parameters_update is not None:
            if "model" in parameters_update:
                model_params = parameters_update.pop("model")
            else:
                model_params = None
            if "features" in parameters_update:
                feat_params = parameters_update.pop("features")
            else:
                feat_params = None
            if "augmentations" in parameters_update:
                aug_params = parameters_update.pop("augmentations")
            else:
                aug_params = None
            parameters = self._update(parameters, parameters_update)
            model_name = parameters["general"]["model_name"]
            parameters["model"] = self._open_yaml(
                os.path.join(self.project_path, "config", "model", f"{model_name}.yaml")
            )
            if model_params is not None:
                parameters["model"] = self._update(parameters["model"], model_params)
            feat_name = parameters["general"]["feature_extraction"]
            parameters["features"] = self._open_yaml(
                os.path.join(
                    self.project_path, "config", "features", f"{feat_name}.yaml"
                )
            )
            if feat_params is not None:
                parameters["features"] = self._update(
                    parameters["features"], feat_params
                )
            aug_name = options.extractor_to_transformer[
                parameters["general"]["feature_extraction"]
            ]
            parameters["augmentations"] = self._open_yaml(
                os.path.join(
                    self.project_path, "config", "augmentations", f"{aug_name}.yaml"
                )
            )
            if aug_params is not None:
                parameters["augmentations"] = self._update(
                    parameters["augmentations"], aug_params
                )
        if load_search is not None:
            parameters_update, model_name = self._searches().get_best_params(
                load_search, load_parameters, round_to_binary
            )
            parameters["general"]["model_name"] = model_name
            parameters["model"] = self._open_yaml(
                os.path.join(self.project_path, "config", "model", f"{model_name}.yaml")
            )
            parameters = self._update(parameters, parameters_update)
        for key in keys:
            with open(
                os.path.join(self.project_path, "config", f"{key}.yaml"), "w", encoding="utf-8"
            ) as f:
                YAML().dump(parameters[key], f)
        model_name = parameters["general"]["model_name"]
        model_path = os.path.join(
            self.project_path, "config", "model", f"{model_name}.yaml"
        )
        with open(model_path, "w", encoding="utf-8") as f:
            YAML().dump(parameters["model"], f)
        features_name = parameters["general"]["feature_extraction"]
        features_path = os.path.join(
            self.project_path, "config", "features", f"{features_name}.yaml"
        )
        with open(features_path, "w", encoding="utf-8") as f:
            YAML().dump(parameters["features"], f)
        aug_name = options.extractor_to_transformer[features_name]
        aug_path = os.path.join(
            self.project_path, "config", "augmentations", f"{aug_name}.yaml"
        )
        with open(aug_path, "w", encoding="utf-8") as f:
            YAML().dump(parameters["augmentations"], f)

    def get_summary(
        self,
        episode_names: list,
        method: str = "last",
        average: int = 1,
        metrics: List = None,
    ) -> Dict:
        """
        Get a summary of episode statistics

        If the episode has multiple runs, the statistics will be aggregated over all of them.

        Parameters
        ----------
        episode_name : str
            the name of the episode
        method : ["best", "last"]
            the method for choosing the epochs
        average : int, default 1
            the number of epochs to average over (for each run)
        metrics : list, optional
            a list of metrics

        Returns
        -------
        statistics : dict
            a nested dictionary where first-level keys are metric names and second-level keys are 'mean' for the mean
            and 'std' for the standard deviation
        """

        runs = []
        for episode_name in episode_names:
            runs_ep = self._episodes().get_runs(episode_name)
            if len(runs_ep) == 0:
                raise RuntimeError(
                    f"There is no {episode_name} episode in the project memory"
                )
            runs += runs_ep
        if metrics is None:
            metrics = self._episode(runs[0]).get_metrics()

        values = {m: [] for m in metrics}
        for run in runs:
            for m in metrics:
                log = self._episode(run).get_metric_log(mode="val", metric_name=m)
                if method == "best":
                    log = sorted(log)
                    values[m] += list(log[-average:])
                elif method == "last":
                    if len(log) == 0:
                        episodes = self._episodes().data
                        if average == 1 and ("results", m) in episodes.columns:
                            values[m] += [episodes.loc[run, ("results", m)]]
                        else:
                            raise RuntimeError(f"Did not find {m} metric for {run} run")
                    values[m] += list(log[-average:])
                elif method.startswith("epoch"):
                    epoch = int(method[5:]) - 1
                    pars = self._episodes().load_parameters(run)
                    step = int(pars["training"]["validation_interval"])
                    values[m] += [log[epoch // step]]
                else:
                    raise ValueError(
                        f"The {method} method is not recognized! Please choose from ['last', 'best', 'epoch...']"
                    )
        statistics = defaultdict(lambda: {})
        for m, v in values.items():
            statistics[m]["mean"] = np.mean(v)
            statistics[m]["std"] = np.std(v)
        print(f"SUMMARY {episode_names}")
        for m, v in statistics.items():
            print(f'{m}: mean {v["mean"]:.3f}, std {v["std"]:.3f}')
        print("\n")
        return dict(statistics)

    @staticmethod
    def remove_project(name: str, projects_path: str = None) -> None:
        """
        Remove all project files and experiment records and results
        """

        if projects_path is None:
            projects_path = os.path.join(str(Path.home()), "DLC2Action")
        project_path = os.path.join(projects_path, name)
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def remove_saved_features(
        self,
        dataset_names: List = None,
        exceptions: List = None,
        remove_active: bool = False,
    ) -> None:
        """
        Remove saved pre-computed dataset files

        By default, all pre-computed features will be deleted.
        No essential information can get lost, storing them only saves time. Be careful with deleting datasets
        while training or inference is happening though.

        Parameters
        ----------
        dataset_names : list, optional
            a list of dataset names to delete (by default all names are added)
        exceptions : list, optional
            a list of dataset names to not be deleted
        remove_active : bool, default False
            if `False`, datasets used by unfinished episodes will not be deleted
        """

        print("Removing datasets...")
        if dataset_names is None:
            dataset_names = []
        if exceptions is None:
            exceptions = []
        if not remove_active:
            exceptions += self._episodes().get_active_datasets()
        dataset_path = os.path.join(self.project_path, "saved_datasets")
        if os.path.exists(dataset_path):
            if dataset_names == []:
                dataset_names = set([f.split(".")[0] for f in os.listdir(dataset_path)])

            to_remove = [
                x
                for x in dataset_names
                if os.path.exists(os.path.join(dataset_path, x)) and x not in exceptions
            ]
            if len(to_remove) > 2:
                to_remove = tqdm(to_remove)
            for dataset in to_remove:
                shutil.rmtree(os.path.join(dataset_path, dataset))
            to_remove = [
                f"{x}.pickle"
                for x in dataset_names
                if os.path.exists(os.path.join(dataset_path, f"{x}.pickle"))
                and x not in exceptions
            ]
            for dataset in to_remove:
                os.remove(os.path.join(dataset_path, dataset))
            names = self._saved_datasets().dataset_names()
            self._saved_datasets().remove(names)
        print("\n")

    def remove_extra_checkpoints(
        self, episode_names: List = None, exceptions: List = None
    ) -> None:
        """
        Remove intermediate model checkpoint files (only leave the results of the last epoch)

        By default, all intermediate checkpoints will be deleted.
        Files in the model folder that are not associated with any record in the meta files are also deleted.

        Parameters
        ----------
        episode_names : list, optional
            a list of episode names to clean (by default all names are added)
        exceptions : list, optional
            a list of episode names to not clean
        """

        model_path = os.path.join(self.project_path, "results", "model")
        try:
            all_names = self._episodes().data.index
        except:
            all_names = os.listdir(model_path)
        if episode_names is None:
            episode_names = all_names
        if exceptions is None:
            exceptions = []
        to_remove = [x for x in episode_names if x not in exceptions]
        folders = os.listdir(model_path)
        for folder in folders:
            if folder not in all_names:
                shutil.rmtree(os.path.join(model_path, folder))
            elif folder in to_remove:
                files = os.listdir(os.path.join(model_path, folder))
                for file in sorted(files)[:-1]:
                    os.remove(os.path.join(model_path, folder, file))

    def remove_search(self, search_name: str) -> None:
        """
        Remove a hyperparameter search record

        Parameters
        ----------
        search_name : str
            the name of the search to remove
        """

        self._searches().remove_episode(search_name)
        graph_path = os.path.join(self.project_path, "results", "searches", search_name)
        if os.path.exists(graph_path):
            shutil.rmtree(graph_path)

    def remove_prediction(self, prediction_name: str) -> None:
        """
        Remove a prediction record

        Parameters
        ----------
        prediction_name : str
            the name of the prediction to remove
        """

        self._predictions().remove_episode(prediction_name)
        prediction_path = os.path.join(
            self.project_path, "results", "predictions", prediction_name
        )
        if os.path.exists(prediction_path):
            shutil.rmtree(prediction_path)

    def remove_episode(self, episode_name: str) -> None:
        """
        Remove all model, logs and metafile records related to an episode

        Parameters
        ----------
        episode_name : str
            the name of the episode to remove
        """

        runs = self._episodes().get_runs(episode_name)
        runs.append(episode_name)
        for run in runs:
            self._episodes().remove_episode(run)
            model_path = os.path.join(self.project_path, "results", "model", run)
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
            log_path = os.path.join(self.project_path, "results", "logs", f"{run}.txt")
            if os.path.exists(log_path):
                os.remove(log_path)

    def prune_unfinished(self, exceptions: List = None) -> None:
        """
        Remove all interrupted episodes

        Remove all episodes that either don't have a log file or have less epochs in the log file than in
        the training parameters or have a model folder but not a record. Note that it can remove episodes that are
        currently running!

        Parameters
        ----------
        exceptions : list
            the episodes to keep even if they are interrupted

        Returns
        -------
        pruned : list
            a list of the episode names that were pruned
        """

        if exceptions is None:
            exceptions = []
        unfinished = self._episodes().unfinished_episodes()
        unfinished = [x for x in unfinished if x not in exceptions]
        model_folders = os.listdir(os.path.join(self.project_path, "results", "model"))
        unfinished += [
            x for x in model_folders if x not in self._episodes().list_episodes().index
        ]
        print(f"PRUNING {unfinished}")
        for episode_name in unfinished:
            self.remove_episode(episode_name)
        print(f"\n")
        return unfinished

    def prediction_path(self, prediction_name: str) -> str:
        """
        Get the path where prediction files are saved

        Parameters
        ----------
        prediction_name : str
            name of the prediction

        Returns
        -------
        prediction_path : str
            the file path
        """

        return os.path.join(
            self.project_path, "results", "predictions", f"{prediction_name}"
        )

    @classmethod
    def print_data_types(cls):
        print("DATA TYPES:")
        for key, value in cls.data_types().items():
            print(f"{key}:")
            print(value.__doc__)

    @classmethod
    def print_annotation_types(cls):
        print("ANNOTATION TYPES:")
        for key, value in cls.annotation_types().items():
            print(f"{key}:")
            print(value.__doc__)

    @staticmethod
    def data_types() -> List:
        """
        Get available data types

        Returns
        -------
        list
            available data types
        """

        return options.input_stores

    @staticmethod
    def annotation_types() -> List:
        """
        Get available annotation types

        Returns
        -------
        list
            available annotation types
        """

        return options.annotation_stores

    def _save_mask(self, file: Dict, mask_name: str):
        """
        Save a mask file
        """

        if not os.path.exists(self._mask_path()):
            os.mkdir(self._mask_path())
        with open(os.path.join(self._mask_path(), mask_name + ".pickle"), "wb") as f:
            pickle.dump(file, f)

    def _load_mask(self, mask_name: str) -> Dict:
        """
        Load a mask file
        """

        with open(os.path.join(self._mask_path(), mask_name + ".pickle"), "rb") as f:
            data = pickle.load(f)
        return data

    def _thresholds(self) -> DecisionThresholds:
        """
        Get the decision thresholds meta object
        """

        return DecisionThresholds(self._thresholds_path())

    def _episodes(self) -> SavedRuns:
        """
        Get the episodes meta object

        Returns
        -------
        episodes : SavedRuns
            the episodes meta object
        """

        try:
            return SavedRuns(self._episodes_path(), self.project_path)
        except:
            self.load_metadata_backup()
            return SavedRuns(self._episodes_path(), self.project_path)

    def _predictions(self) -> SavedRuns:
        """
        Get the predictions meta object

        Returns
        -------
        predictions : SavedRuns
            the predictions meta object
        """

        try:
            return SavedRuns(self._predictions_path(), self.project_path)
        except:
            self.load_metadata_backup()
            return SavedRuns(self._predictions_path(), self.project_path)

    def _saved_datasets(self) -> SavedStores:
        """
        Get the datasets meta object

        Returns
        -------
        datasets : SavedStores
            the datasets meta object
        """

        try:
            return SavedStores(self._saved_datasets_path())
        except:
            self.load_metadata_backup()
            return SavedStores(self._saved_datasets_path())

    def _prediction(self, name: str) -> Run:
        """
        Get a prediction meta object

        Parameters
        ----------
        name : str
            episode name

        Returns
        -------
        prediction : Run
            the prediction meta object
        """

        try:
            return Run(name, self.project_path, meta_path=self._predictions_path())
        except:
            self.load_metadata_backup()
            return Run(name, self.project_path, meta_path=self._predictions_path())

    def _episode(self, name: str) -> Run:
        """
        Get an episode meta object

        Parameters
        ----------
        name : str
            episode name

        Returns
        -------
        episode : Run
            the episode meta object
        """

        try:
            return Run(name, self.project_path, meta_path=self._episodes_path())
        except:
            self.load_metadata_backup()
            return Run(name, self.project_path, meta_path=self._episodes_path())

    def _searches(self) -> Searches:
        """
        Get the hyperparameter search meta object

        Returns
        -------
        searches : Searches
            the searches meta object
        """

        try:
            return Searches(self._searches_path(), self.project_path)
        except:
            self.load_metadata_backup()
            return Searches(self._searches_path(), self.project_path)

    def _update_configs(self) -> None:
        """
        Update the project config files with newly added files and parameters
        """

        self.update_parameters({"data": {"data_path": self.data_path}})
        folders = ["augmentations", "features", "model"]
        original_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config"
        )
        project_path = os.path.join(self.project_path, "config")
        filenames = [x for x in os.listdir(original_path) if x.endswith("yaml")]
        for folder in folders:
            filenames += [
                os.path.join(folder, x)
                for x in os.listdir(os.path.join(original_path, folder))
            ]
        filenames.append(os.path.join("data", f"{self.data_type}.yaml"))
        if self.annotation_type != "none":
            filenames.append(os.path.join("annotation", f"{self.annotation_type}.yaml"))
        for file in filenames:
            filepath_original = os.path.join(original_path, file)
            if file.startswith("data") or file.startswith("annotation"):
                file = os.path.basename(file)
            filepath_project = os.path.join(project_path, file)
            if not os.path.exists(filepath_project):
                shutil.copy(filepath_original, filepath_project)
            else:
                original_pars = self._open_yaml(filepath_original)
                project_pars = self._open_yaml(filepath_project)
                to_remove = []
                for key, value in project_pars.items():
                    if key not in original_pars:
                        if key not in ["data_type", "annotation_type"]:
                            to_remove.append(key)
                for key in to_remove:
                    project_pars.pop(key)
                to_remove = []
                for key, value in original_pars.items():
                    if key in project_pars:
                        to_remove.append(key)
                for key in to_remove:
                    original_pars.pop(key)
                project_pars = self._update(project_pars, original_pars)
                with open(filepath_project, "w", encoding="utf-8") as f:
                    YAML().dump(project_pars, f)

    def _update_project(self) -> None:
        """
        Update project files with the current version
        """

        version_file = self._version_path()
        ok = True
        if not os.path.exists(version_file):
            ok = False
        else:
            with open(version_file) as f:
                project_version = f.read()
            if project_version < __version__:
                ok = False
            elif project_version > __version__:
                warnings.warn(
                    f"The project expects a higher dlc2action version ({project_version}), please update!"
                )
        if not ok:
            project_config_path = os.path.join(self.project_path, "config")
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__path__)), "config"
            )
            episodes = self._episodes()
            folders = ["annotation", "augmentations", "data", "features", "model"]

            project_annotation_configs = os.listdir(
                os.path.join(project_config_path, "annotation")
            )
            annotation_configs = os.listdir(os.path.join(config_path, "annotation"))
            for ann_config in annotation_configs:
                if ann_config not in project_annotation_configs:
                    shutil.copytree(
                        os.path.join(config_path, "annotation", ann_config),
                        os.path.join(project_config_path, "annotation", ann_config),
                        dirs_exist_ok=True,
                    )
                else:
                    project_pars = self._open_yaml(
                        os.path.join(project_config_path, "annotation", ann_config)
                    )
                    pars = self._open_yaml(
                        os.path.join(config_path, "annotation", ann_config)
                    )
                    new_keys = set(pars.keys()) - set(project_pars.keys())
                    for key in new_keys:
                        project_pars[key] = pars[key]
                        c = self._get_comment(pars.ca.items.get(key))
                        project_pars.yaml_add_eol_comment(c, key=key)
                        episodes.update(
                            condition=f"general/annotation_type::={ann_config}",
                            update={f"data/{key}": pars[key]},
                        )

    def _initialize_project(
        self,
        data_type: str,
        annotation_type: str = None,
        data_path: str = None,
        annotation_path: str = None,
        copy: bool = True,
    ) -> None:
        """
        Initialize a new project
        """

        if data_type not in self.data_types():
            raise ValueError(
                f"The {data_type} data type is not available. "
                f"Please choose from {self.data_types()}"
            )
        if annotation_type not in self.annotation_types():
            raise ValueError(
                f"The {annotation_type} annotation type is not available. "
                f"Please choose from {self.annotation_types()}"
            )
        os.mkdir(self.project_path)
        folders = ["results", "saved_datasets", "meta", "config"]
        for f in folders:
            os.mkdir(os.path.join(self.project_path, f))
        results_subfolders = [
            "model",
            "logs",
            "predictions",
            "splits",
            "searches",
        ]
        for sf in results_subfolders:
            os.mkdir(os.path.join(self.project_path, "results", sf))
        if data_path is not None:
            if copy:
                os.mkdir(os.path.join(self.project_path, "data"))
                shutil.copytree(
                    data_path,
                    os.path.join(self.project_path, "data"),
                    dirs_exist_ok=True,
                )
                data_path = os.path.join(self.project_path, "data")
        if annotation_path is not None:
            if copy:
                os.mkdir(os.path.join(self.project_path, "annotation"))
                shutil.copytree(
                    annotation_path,
                    os.path.join(self.project_path, "annotation"),
                    dirs_exist_ok=True,
                )
                annotation_path = os.path.join(self.project_path, "annotation")
        self._generate_config(
            data_type,
            annotation_type,
            data_path=data_path,
            annotation_path=annotation_path,
        )
        self._generate_meta()

    def _read_types(self) -> Tuple[str, str]:
        """
        Get data type and annotation type from existing project files
        """

        config_path = os.path.join(self.project_path, "config", "general.yaml")
        with open(config_path) as f:
            pars = YAML().load(f)
        data_type = pars["data_type"]
        annotation_type = pars["annotation_type"]
        return annotation_type, data_type

    def _read_paths(self) -> Tuple[str, str]:
        """
        Get data type and annotation type from existing project files
        """

        config_path = os.path.join(self.project_path, "config", "data.yaml")
        with open(config_path) as f:
            pars = YAML().load(f)
        data_path = pars["data_path"]
        annotation_path = pars["annotation_path"]
        return annotation_path, data_path

    def _generate_config(
        self, data_type: str, annotation_type: str, data_path: str, annotation_path: str
    ) -> None:
        """
        Initialize the config files
        """

        default_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config"
        )
        config_path = os.path.join(self.project_path, "config")
        files = ["losses", "metrics", "ssl", "training"]
        for f in files:
            shutil.copy(os.path.join(default_path, f"{f}.yaml"), config_path)
        shutil.copytree(
            os.path.join(default_path, "model"), os.path.join(config_path, "model")
        )
        shutil.copytree(
            os.path.join(default_path, "features"),
            os.path.join(config_path, "features"),
        )
        shutil.copytree(
            os.path.join(default_path, "augmentations"),
            os.path.join(config_path, "augmentations"),
        )
        yaml = YAML()
        data_param_path = os.path.join(default_path, "data", f"{data_type}.yaml")
        if os.path.exists(data_param_path):
            with open(data_param_path, encoding="utf-8") as f:
                data_params = yaml.load(f)
        if data_params is None:
            data_params = {}
        if annotation_type is None:
            ann_params = {}
        else:
            ann_param_path = os.path.join(
                default_path, "annotation", f"{annotation_type}.yaml"
            )
            if os.path.exists(ann_param_path):
                ann_params = self._open_yaml(ann_param_path)
            elif annotation_type == "none":
                ann_params = {}
            else:
                raise ValueError(
                    f"The {annotation_type} data type is not available. "
                    f"Please choose from {BehaviorDataset.annotation_types()}"
                )
        if ann_params is None:
            ann_params = {}
        data_params = self._update(data_params, ann_params)
        data_params["data_path"] = data_path
        data_params["annotation_path"] = annotation_path
        with open(os.path.join(config_path, "data.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(data_params, f)
        with open(os.path.join(default_path, "general.yaml"), encoding="utf-8") as f:
            general_params = yaml.load(f)
        general_params["data_type"] = data_type
        general_params["annotation_type"] = annotation_type
        with open(os.path.join(config_path, "general.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(general_params, f)

    def _generate_meta(self) -> None:
        """
        Initialize the meta files
        """

        config_file = os.path.join(self.project_path, "config")
        meta_fields = ["time"]
        columns = [("meta", field) for field in meta_fields]
        episodes = pd.DataFrame(columns=pd.MultiIndex.from_tuples(columns))
        episodes.to_pickle(self._episodes_path())
        meta_fields = ["time", "objective"]
        result_fields = ["best_params", "best_value"]
        columns = [("meta", field) for field in meta_fields] + [
            ("results", field) for field in result_fields
        ]
        searches = pd.DataFrame(columns=pd.MultiIndex.from_tuples(columns))
        searches.to_pickle(self._searches_path())
        meta_fields = ["time"]
        columns = [("meta", field) for field in meta_fields]
        predictions = pd.DataFrame(columns=pd.MultiIndex.from_tuples(columns))
        predictions.to_pickle(self._predictions_path())
        with open(os.path.join(config_file, "data.yaml")) as f:
            data_keys = list(YAML().load(f).keys())
        saved_data = pd.DataFrame(columns=data_keys)
        saved_data.to_pickle(self._saved_datasets_path())
        pd.DataFrame().to_pickle(self._thresholds_path())
        # with open(self._version_path()) as f:
        #     f.write(__version__)

    def _open_yaml(self, path: str) -> CommentedMap:
        """
        Load a parameter dictionary from a .yaml file
        """

        with open(path, encoding="utf-8") as f:
            data = YAML().load(f)
        if data is None:
            data = {}
        return data

    def _compare(self, d: Dict, u: Dict, allow_diff: float = 1e-7):
        """
        Compare nested dictionaries with 'almost equal' condition
        """

        ok = True
        if u.keys() != d.keys():
            ok = False
        else:
            for k, v in u.items():
                if isinstance(v, Mapping):
                    ok = self._compare(d[k], v, allow_diff=allow_diff)
                else:
                    if isinstance(v, float) or isinstance(d[k], float):
                        if not isinstance(d[k], float) and not isinstance(d[k], int):
                            ok = False
                        elif not isinstance(v, float) and not isinstance(v, int):
                            ok = False
                        elif np.abs(v - d[k]) > allow_diff:
                            ok = False
                    elif v != d[k]:
                        ok = False
        return ok

    def _check_comment(self, comment_sequence: List) -> bool:
        """
        Check if a comment already exists in a ruamel.yaml comment sequence
        """

        if comment_sequence is None:
            return False
        c = self._get_comment(comment_sequence)
        if c != "":
            return True
        else:
            return False

    def _get_comment(self, comment_sequence: List, strip=True) -> str:
        """
        Get the comment string from a ruamel.yaml comment sequence
        """

        if comment_sequence is None:
            return ""
        c = ""
        for cm in comment_sequence:
            if cm is not None:
                if isinstance(cm, Iterable):
                    for c in cm:
                        if c is not None:
                            c = c.value
                            break
                    break
                else:
                    c = cm.value
                    break
        if strip:
            c = c.strip()
        return c

    def _update(self, d: Union[CommentedMap, Dict], u: Union[CommentedMap, Dict]):
        """
        Update a nested dictionary
        """

        if "general" in u and "model_name" in u["general"] and "model" in d:
            model_name = u["general"]["model_name"]
            if d["general"]["model_name"] != model_name:
                d["model"] = self._open_yaml(
                    os.path.join(
                        self.project_path, "config", "model", f"{model_name}.yaml"
                    )
                )
        d_copied = deepcopy(d)
        for k, v in u.items():
            if (
                k in d_copied
                and isinstance(d_copied[k], list)
                and isinstance(v, Mapping)
                and all([isinstance(x, int) for x in v.keys()])
            ):
                for kk, vv in v.items():
                    d_copied[k][kk] = vv
            elif (
                isinstance(v, Mapping)
                and k in d_copied
                and isinstance(d_copied[k], Mapping)
            ):
                if d_copied[k] is None:
                    d_k = CommentedMap()
                else:
                    d_k = d_copied[k]
                d_copied[k] = self._update(d_k, v)
            else:
                d_copied[k] = v
                if isinstance(u, CommentedMap) and u.ca.items.get(k) is not None:
                    c = self._get_comment(u.ca.items.get(k), strip=False)
                    if isinstance(d_copied, CommentedMap) and not self._check_comment(
                        d_copied.ca.items.get(k)
                    ):
                        d_copied.yaml_add_eol_comment(c, key=k)
        return d_copied

    def _update_with_search(
        self,
        d: Dict,
        search_name: str,
        load_parameters: list = None,
        round_to_binary: list = None,
    ):
        """
        Update a dictionary with best parameters from a hyperparameter search
        """

        u, _ = self._searches().get_best_params(
            search_name, load_parameters, round_to_binary
        )
        return self._update(d, u)

    def _read_parameters(self, catch_blanks=True) -> Dict:
        """
        Compose a parameter dictionary to create a task from the config files
        """

        config_path = os.path.join(self.project_path, "config")
        keys = [
            "data",
            "general",
            "losses",
            "metrics",
            "ssl",
            "training",
        ]
        parameters = {}
        for key in keys:
            parameters[key] = self._open_yaml(os.path.join(config_path, f"{key}.yaml"))
        features = parameters["general"]["feature_extraction"]
        parameters["features"] = self._open_yaml(
            os.path.join(config_path, "features", f"{features}.yaml")
        )
        transformer = options.extractor_to_transformer[features]
        parameters["augmentations"] = self._open_yaml(
            os.path.join(config_path, "augmentations", f"{transformer}.yaml")
        )
        model = parameters["general"]["model_name"]
        parameters["model"] = self._open_yaml(
            os.path.join(config_path, "model", f"{model}.yaml")
        )
        # input = parameters["general"]["input"]
        # parameters["model"] = self._open_yaml(
        #     os.path.join(config_path, "model", f"{model}.yaml")
        # )
        if catch_blanks:
            blanks = self._get_blanks()
            if len(blanks) > 0:
                self.list_blanks()
                raise ValueError(
                    f"Please fill in all the blanks before running experiments"
                )
        return parameters

    def set_main_parameters(self, model_name: str = None, metric_names: List = None):
        """
        Select the model and the metrics

        Parameters
        ----------
        model_name : str, optional
            model name; run `project.help("model") to find out more
        metric_names : list, optional
            a list of metric function names; run `project.help("metrics") to find out more
        """

        pars = {"general": {}}
        if model_name is not None:
            assert model_name in options.models
            pars["general"]["model_name"] = model_name
        if metric_names is not None:
            for metric in metric_names:
                assert metric in options.metrics
            pars["general"]["metric_functions"] = metric_names
        self.update_parameters(pars)

    def help(self, keyword: str = None):
        """
        Get information on available options

        Parameters
        ----------
        keyword : str, optional
            the keyword for options (run without arguments to see which keywords are available)

        """

        if keyword is None:
            print("AVAILABLE HELP FUNCTIONS:")
            print("- Try running `project.help(keyword)` with the following keywords:")
            print("    - model: to get more information on available models,")
            print(
                "    - features: to get more information on available feature extraction modes,"
            )
            print(
                "    - partition_method: to get more information on available train/test/val partitioning methods,"
            )
            print("    - metrics: to see a list of available metric functions.")
            print("    - data: to see help for expected data structure")
            print(
                "- To start working with this project, first run `project.list_blanks()` to check which parameters need to be filled in."
            )
            print(
                "- After a model and metrics are set, run `project.list_basic_parameters()` to see a list of the most important parameters that you might want to modify"
            )
            print(
                f"- If you want to dig deeper, get the full dictionary with project._read_parameters() (it is a `ruamel.yaml.comments.CommentedMap` instance)."
            )
        elif keyword == "model":
            print("MODELS:")
            for key, model in options.models.items():
                print(f"{key}:")
                print(model.__doc__)
        elif keyword == "features":
            print("FEATURE EXTRACTORS:")
            for key, extractor in options.feature_extractors.items():
                print(f"{key}:")
                print(extractor.__doc__)
        elif keyword == "partition_method":
            print("PARTITION METHODS:")
            print(
                BehaviorDataset.partition_train_test_val.__doc__.split(
                    "The partitioning method:"
                )[1].split("val_frac :")[0]
            )
        elif keyword == "metrics":
            print("METRICS:")
            for key, metric in options.metrics.items():
                print(f"{key}:")
                print(metric.__doc__)
        elif keyword == "data":
            print("DATA:")
            print(f"Video data: {self.data_type}")
            print(options.input_stores[self.data_type].__doc__)
            print(f"Annotation data: {self.annotation_type}")
            print(options.annotation_stores[self.annotation_type].__doc__)
            print(
                "Annotation path and data path don't have to be separate, you can keep everything in one folder."
            )
        else:
            raise ValueError(f"The {keyword} keyword is not recognized")
        print("\n")

    def _process_value(self, value):
        if isinstance(value, str):
            value = f'"{value}"'
        elif isinstance(value, CommentedSet):
            value = {x for x in value}
        return value

    def _get_blanks(self):
        caught = []
        parameters = self._read_parameters(catch_blanks=False)
        for big_key, big_value in parameters.items():
            for key, value in big_value.items():
                if value == "???":
                    caught.append(
                        (big_key, key, self._get_comment(big_value.ca.items.get(key)))
                    )
        return caught

    def list_blanks(self, blanks=None):
        """
        List parameters that need to be filled in

        Parameters
        ----------
        blanks : list, optional
            a list of the parameters to list, if already known
        """

        if blanks is None:
            blanks = self._get_blanks()
        if len(blanks) > 0:
            to_update = defaultdict(lambda: [])
            for b, k, c in blanks:
                to_update[b].append((k, c))
            print("Before running experiments, please update all the blanks.")
            print("To do that, you can run this.")
            print("--------------------------------------------------------")
            print(f"project.update_parameters(")
            print(f"    {{")
            for big_key, keys in to_update.items():
                print(f'        "{big_key}": {{')
                for key, comment in keys:
                    print(f'            "{key}": ..., {comment}')
                print(f"        }},")
            print(f"    }}")
            print(")")
            print("--------------------------------------------------------")
            print("Replace ... with relevant values.")
        else:
            print("There is no blanks left!")

    def list_basic_parameters(
        self,
    ):
        """
        Get a list of most relevant parameters and code to modify them
        """

        parameters = self._read_parameters()
        print("BASIC PARAMETERS:")
        model_name = parameters["general"]["model_name"]
        metric_names = parameters["general"]["metric_functions"]
        loss_name = parameters["general"]["loss_function"]
        feature_extraction = parameters["general"]["feature_extraction"]
        print("Here is a list of current parameters.")
        print(
            "You can copy this code, change the parameters you want to set and run it to update the project config."
        )
        print("--------------------------------------------------------")
        print("project.update_parameters(")
        print("    {")
        for group in ["general", "data", "training"]:
            print(f'        "{group}": {{')
            for key in options.basic_parameters[group]:
                if key in parameters[group]:
                    print(
                        f'            "{key}": {self._process_value(parameters[group][key])}, {self._get_comment(parameters[group].ca.items.get(key))}'
                    )
            print("        },")
        print('        "losses": {')
        print(f'            "{loss_name}": {{')
        for key in options.basic_parameters["losses"][loss_name]:
            if key in parameters["losses"][loss_name]:
                print(
                    f'                "{key}": {self._process_value(parameters["losses"][loss_name][key])}, {self._get_comment(parameters["losses"][loss_name].ca.items.get(key))}'
                )
        print("            },")
        print("        },")
        print('        "metrics": {')
        for metric in metric_names:
            print(f'            "{metric}": {{')
            for key in parameters["metrics"][metric]:
                print(
                    f'                "{key}": {self._process_value(parameters["metrics"][metric][key])}, {self._get_comment(parameters["metrics"][metric].ca.items.get(key))}'
                )
            print("            },")
        print("        },")
        print('        "model": {')
        for key in options.basic_parameters["model"][model_name]:
            if key in parameters["model"]:
                print(
                    f'            "{key}": {self._process_value(parameters["model"][key])}, {self._get_comment(parameters["model"].ca.items.get(key))}'
                )

        print("        },")
        print('        "features": {')
        for key in options.basic_parameters["features"][feature_extraction]:
            if key in parameters["features"]:
                print(
                    f'            "{key}": {self._process_value(parameters["features"][key])}, {self._get_comment(parameters["features"].ca.items.get(key))}'
                )

        print("        },")
        print('        "augmentations": {')
        for key in options.basic_parameters["augmentations"][feature_extraction]:
            if key in parameters["augmentations"]:
                print(
                    f'            "{key}": {self._process_value(parameters["augmentations"][key])}, {self._get_comment(parameters["augmentations"].ca.items.get(key))}'
                )
        print("        },")
        print("    },")
        print(")")
        print("--------------------------------------------------------")
        print("\n")

    def _create_record(
        self,
        episode_name: str,
        behaviors_dict: Dict,
        load_episode: str = None,
        parameters_update: Dict = None,
        task: TaskDispatcher = None,
        load_epoch: int = None,
        load_search: str = None,
        load_parameters: list = None,
        round_to_binary: list = None,
        load_strict: bool = True,
        n_seeds: int = 1,
    ) -> TaskDispatcher:
        """
        Create a meta data episode record
        """

        if episode_name in self._episodes().data.index:
            return
        if type(n_seeds) is not int or n_seeds < 1:
            raise ValueError(
                f"The n_seeds parameter has to be an integer larger than 0; got {n_seeds}"
            )
        if parameters_update is None:
            parameters_update = {}
        parameters = self._read_parameters()
        parameters = self._update(parameters, parameters_update)
        if load_search is not None:
            parameters = self._update_with_search(
                parameters, load_search, load_parameters, round_to_binary
            )
        parameters = self._fill(
            parameters,
            episode_name,
            load_episode,
            load_epoch=load_epoch,
            only_load_model=True,
            load_strict=load_strict,
            continuing=True,
        )
        self._save_episode(episode_name, parameters, behaviors_dict)
        return task

    def _save_thresholds(
        self,
        episode_names: List,
        metric_name: str,
        parameters: Dict,
        thresholds: List,
        load_epochs: List,
    ):
        """
        Save optimal decision thresholds in the meta records
        """

        metric_parameters = parameters["metrics"][metric_name]
        self._thresholds().save_thresholds(
            episode_names, load_epochs, metric_name, metric_parameters, thresholds
        )

    def _save_episode(
        self,
        episode_name: str,
        parameters: Dict,
        behaviors_dict: Dict,
        suppress_validation: bool = False,
        training_time: str = None,
        norm_stats: Dict = None,
    ) -> None:
        """
        Save an episode in the meta files
        """

        try:
            split_info = self._split_info_from_filename(
                parameters["training"]["split_path"]
            )
            parameters["training"]["partition_method"] = split_info["partition_method"]
        except:
            pass
        if norm_stats is not None:
            norm_stats = dict(norm_stats)
        parameters["training"]["stats"] = norm_stats
        self._episodes().save_episode(
            episode_name,
            parameters,
            behaviors_dict,
            suppress_validation=suppress_validation,
            training_time=training_time,
        )

    def _update_episode_results(
        self,
        episode_name: str,
        logs: Tuple,
        training_time: str = None,
    ) -> None:
        """
        Save the results of a run in the meta files
        """

        self._episodes().update_episode_results(episode_name, logs, training_time)

    def _save_prediction(
        self,
        episode_name: str,
        parameters: Dict,
        behaviors_dict: Dict,
        embedding: bool = False,
        inference_time: str = None,
    ) -> None:
        """
        Save a prediction in the meta files
        """

        parameters = self._update(
            parameters,
            {"meta": {"embedding": embedding, "inference_time": inference_time}},
        )
        self._predictions().save_episode(episode_name, parameters, behaviors_dict)

    def _save_search(
        self,
        search_name: str,
        parameters: Dict,
        n_trials: int,
        best_params: Dict,
        best_value: float,
        metric: str,
        search_space: Dict,
    ) -> None:
        """
        Save a hyperparameter search in the meta files
        """

        self._searches().save_search(
            search_name,
            parameters,
            n_trials,
            best_params,
            best_value,
            metric,
            search_space,
        )

    def _save_stores(self, parameters: Dict) -> None:
        """
        Save a pickled dataset in the meta files
        """

        name = os.path.basename(parameters["data"]["feature_save_path"])
        self._saved_datasets().save_store(name, self._get_data_pars(parameters))
        self.create_metadata_backup()

    def _remove_stores(self, parameters: Dict, remove_active: bool = False) -> None:
        """
        Remove the pre-computed features folder
        """

        name = os.path.basename(parameters["data"]["feature_save_path"])
        if remove_active or name not in self._episodes().get_active_datasets():
            self.remove_saved_features([name])

    def _check_episode_validity(
        self, episode_name: str, allow_doublecolon: bool = False, force: bool = False
    ) -> None:
        """
        Check whether the episode name is valid
        """

        if episode_name.startswith("_"):
            raise ValueError(
                "Names starting with an underscore are reserved by dlc2action and cannot be used!"
            )
        elif "." in episode_name:
            raise ValueError("Names containing '.' cannot be used!")
        if not allow_doublecolon and "::" in episode_name:
            raise ValueError(
                "Names containing '::' are reserved by dlc2action and cannot be used!"
            )
        if force:
            self.remove_episode(episode_name)
        elif not self._episodes().check_name_validity(episode_name):
            raise ValueError(
                f"The {episode_name} name is already taken! Set force=True to overwrite."
            )

    def _check_search_validity(self, search_name: str, force: bool = False) -> None:
        """
        Check whether the search name is valid
        """

        if search_name.startswith("_"):
            raise ValueError(
                "Names starting with an underscore are reserved by dlc2action and cannot be used!"
            )
        elif "." in search_name:
            raise ValueError("Names containing '.' cannot be used!")
        if force:
            self.remove_search(search_name)
        elif not self._searches().check_name_validity(search_name):
            raise ValueError(f"The {search_name} name is already taken!")

    def _check_prediction_validity(
        self, prediction_name: str, force: bool = False
    ) -> None:
        """
        Check whether the prediction name is valid
        """

        if prediction_name.startswith("_"):
            raise ValueError(
                "Names starting with an underscore are reserved by dlc2action and cannot be used!"
            )
        elif "." in prediction_name:
            raise ValueError("Names containing '.' cannot be used!")
        if force:
            self.remove_prediction(prediction_name)
        elif not self._predictions().check_name_validity(prediction_name):
            raise ValueError(f"The {prediction_name} name is already taken!")

    def _training_time(self, episode_name: str) -> int:
        """
        Get the training time of an episode in seconds
        """

        return self._episode(episode_name).training_time()

    def _mask_path(self) -> str:
        """
        Get the path to the masks folder
        """

        return os.path.join(self.project_path, "results", "masks")

    def _thresholds_path(self) -> str:
        """
        Get the path to the thresholds meta file
        """

        return os.path.join(self.project_path, "meta", "thresholds.pickle")

    def _episodes_path(self) -> str:
        """
        Get the path to the episodes meta file
        """

        return os.path.join(self.project_path, "meta", "episodes.pickle")

    def _saved_datasets_path(self) -> str:
        """
        Get the path to the datasets meta file
        """

        return os.path.join(self.project_path, "meta", "saved_datasets.pickle")

    def _predictions_path(self) -> str:
        """
        Get the path to the predictions meta file
        """

        return os.path.join(self.project_path, "meta", "predictions.pickle")

    def _dataset_store_path(self, name: str) -> str:
        """
        Get the path to a specific pickled dataset
        """

        return os.path.join(self.project_path, "saved_datasets", f"{name}.pickle")

    def _searches_path(self) -> str:
        """
        Get the path to the hyperparameter search meta file
        """

        return os.path.join(self.project_path, "meta", "searches.pickle")

    def _search_path(self, name: str) -> str:
        """
        Get the default path to the graph folder for a specific hyperparameter search
        """

        return os.path.join(self.project_path, "results", "searches", name)

    def _version_path(self) -> str:
        """
        Get the path to the version file
        """

        return os.path.join(self.project_path, "meta", "version.txt")

    def _default_split_file(self, split_info: Dict) -> Optional[str]:
        """
        Generate a path to a split file from split parameters
        """

        if split_info["partition_method"].startswith("time"):
            return None
        val_frac = split_info["val_frac"]
        test_frac = split_info["test_frac"]
        split_name = f'{split_info["partition_method"]}_val{val_frac * 100}%_test{test_frac * 100}%_len{split_info["len_segment"]}_overlap{split_info["overlap"]}'
        if not split_info["only_load_annotated"]:
            split_name += "_all"
        split_name += ".txt"
        return os.path.join(self.project_path, "results", "splits", split_name)

    def _split_info_from_filename(self, split_name: str) -> Dict:
        """
        Get split parameters from default path to a split file
        """

        if split_name is None:
            return {}
        try:
            name = os.path.basename(split_name)[:-4]
            split = name.split("_")
            if len(split) == 6:
                only_load_annotated = False
            else:
                only_load_annotated = True
            len_segment = int(split[3][3:])
            overlap = int(split[4][7:])
            method, val, test = split[:3]
            val = float(val[3:-1]) / 100
            test = float(test[4:-1]) / 100
            return {
                "partition_method": method,
                "val_frac": val,
                "test_frac": test,
                "only_load_annotated": only_load_annotated,
                "len_segment": len_segment,
                "overlap": overlap,
            }
        except:
            return {"partition_method": "file"}

    def _fill(
        self,
        parameters: Dict,
        episode_name: str,
        load_experiment: str = None,
        load_epoch: int = None,
        load_strict: bool = True,
        only_load_model: bool = False,
        continuing: bool = False,
        enforce_split_parameters: bool = False,
    ) -> Dict:
        """
        Update the parameters from the config files with project specific information

        Fill in the constant file path parameters and generate a unique log file and a model folder.
        Fill in the split file if the same split has been run before in the project and change partition method to
        from_file.
        Fill in saved data path if a dataset with the same data parameters already exists in the project.
        If load_experiment is not None, fill in the checkpoint path as well.
        The only_load_model training parameter is defined by the corresponding argument.
        If continuing is True, new files are not created and all information is loaded from load_experiment.
        If prediction is True, log and model files are not created.
        The enforce_split_parameters parameter is used to resolve conflicts
        between split file path and split parameters when they arise.
        """

        pars = deepcopy(parameters)
        if episode_name == "_":
            self.remove_episode("_")
        log = os.path.join(self.project_path, "results", "logs", f"{episode_name}.txt")
        model_save_path = os.path.join(
            self.project_path, "results", "model", episode_name
        )
        if not continuing and (os.path.exists(log) or os.path.exists(model_save_path)):
            raise ValueError(
                f"The {episode_name} episode name is already in use! Set force=True to overwrite."
            )
        keys = ["val_frac", "test_frac", "partition_method"]
        if "len_segment" not in pars["general"] and "len_segment" in pars["data"]:
            pars["general"]["len_segment"] = pars["data"]["len_segment"]
        if "overlap" not in pars["general"] and "overlap" in pars["data"]:
            pars["general"]["overlap"] = pars["data"]["overlap"]
        if "len_segment" in pars["data"]:
            pars["data"].pop("len_segment")
        if "overlap" in pars["data"]:
            pars["data"].pop("overlap")
        split_info = {k: pars["training"][k] for k in keys}
        split_info["only_load_annotated"] = pars["general"]["only_load_annotated"]
        split_info["len_segment"] = pars["general"]["len_segment"]
        split_info["overlap"] = pars["general"]["overlap"]
        pars["training"]["log_file"] = log
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)
        pars["training"]["model_save_path"] = model_save_path
        if load_experiment is not None:
            if load_experiment not in self._episodes().data.index:
                raise ValueError(f"The {load_experiment} episode does not exist!")
            old_episode = self._episode(load_experiment)
            old_file = old_episode.split_file()
            old_info = self._split_info_from_filename(old_file)
            if len(old_info) == 0:
                old_info = old_episode.split_info()
            if enforce_split_parameters:
                if split_info["partition_method"] != "file":
                    pars["training"]["split_path"] = self._default_split_file(
                        split_info
                    )
            else:
                equal = True
                if old_info["partition_method"] != split_info["partition_method"]:
                    equal = False
                if old_info["partition_method"] != "file":
                    if (
                        old_info["val_frac"] != split_info["val_frac"]
                        or old_info["test_frac"] != split_info["test_frac"]
                    ):
                        equal = False
                if not continuing and not equal:
                    warnings.warn(
                        f"The partitioning parameters in the loaded experiment ({old_info}) "
                        f"are not equal to the current partitioning parameters ({split_info}). "
                        f"The current parameters are replaced."
                    )
                pars["training"]["split_path"] = old_file
            pars["training"]["checkpoint_path"] = old_episode.model_file(load_epoch)
            pars["training"]["load_strict"] = load_strict
        else:
            pars["training"]["checkpoint_path"] = None
            if pars["training"]["partition_method"] == "file":
                if (
                    "split_path" not in pars["training"]
                    or pars["training"]["split_path"] is None
                ):
                    raise ValueError(
                        "The partition_method parameter is set to file but the "
                        "split_path parameter is not set!"
                    )
                elif not os.path.exists(pars["training"]["split_path"]):
                    raise ValueError(
                        f'The {pars["training"]["split_path"]} split file does not exist'
                    )
            else:
                pars["training"]["split_path"] = self._default_split_file(split_info)
        pars["training"]["only_load_model"] = only_load_model
        pars["data"]["saved_data_path"] = None
        pars["data"]["feature_save_path"] = None
        pars_data_copy = self._get_data_pars(pars)
        saved_data_name = self._saved_datasets().find_name(pars_data_copy)
        if saved_data_name is not None:
            pars["data"]["saved_data_path"] = self._dataset_store_path(saved_data_name)
            pars["data"]["feature_save_path"] = self._dataset_store_path(
                saved_data_name
            ).split(".")[0]
        else:
            dataset_path = self._dataset_store_path(episode_name)
            if os.path.exists(dataset_path):
                name, ext = dataset_path.split(".")
                i = 0
                while os.path.exists(f"{name}_{i}.{ext}"):
                    i += 1
                dataset_path = f"{name}_{i}.{ext}"
            pars["data"]["saved_data_path"] = dataset_path
            pars["data"]["feature_save_path"] = dataset_path.split(".")[0]
        split_split = pars["training"]["partition_method"].split(":")
        random = True
        for partition_method in options.partition_methods["fixed"]:
            method_split = partition_method.split(":")
            if len(split_split) != len(method_split):
                continue
            equal = True
            for x, y in zip(split_split, method_split):
                if y.startswith("{"):
                    continue
                if x != y:
                    equal = False
                    break
            if equal:
                random = False
                break
        if random and os.path.exists(pars["training"]["split_path"]):
            pars["training"]["partition_method"] = "file"
        pars["general"]["save_dataset"] = True
        return pars

    def _get_data_pars(self, pars: Dict) -> Dict:
        """
        Get a complete description of the data from a general parameters dictionary
        """

        pars_data_copy = deepcopy(pars["data"])
        for par in [
            "only_load_annotated",
            "exclusive",
            "feature_extraction",
            "ignored_clips",
            "len_segment",
            "overlap",
        ]:
            pars_data_copy[par] = pars["general"].get(par, None)
        pars_data_copy.update(pars["features"])
        return pars_data_copy

    def count_classes(
        self,
        load_episode: str = None,
        parameters_update: Dict = None,
        remove_saved_features: bool = False,
        bouts: bool = True,
    ) -> Dict:
        """
        Get a dictionary of class counts in different modes

        Parameters
        ----------
        load_episode : str, optional
            the episode settings to load
        parameters_update : dict, optional
            a dictionary of parameter updates (only for "data" and "general" categories)
        remove_saved_features : bool, default False
            if `True`, the dataset that is used for computation is then deleted
        bouts : bool, default False
            if `True`, instead of frame counts segment counts are returned

        Returns
        -------
        class_counts : dict
            a dictionary where first-level keys are "train", "val" and "test", second-level keys are
            class names and values are class counts (in frames)
        """

        if load_episode is None:
            task, parameters = self._make_task_training(
                episode_name="_", parameters_update=parameters_update, throwaway=True
            )
        else:
            task, parameters, _ = self._make_task_prediction(
                "_",
                load_episode=load_episode,
                parameters_update=parameters_update,
            )
        class_counts = task.count_classes(bouts=bouts)
        behaviors = task.behaviors_dict()
        class_counts = {
            kk: {behaviors.get(k, "unknown"): v for k, v in vv.items()}
            for kk, vv in class_counts.items()
        }
        if remove_saved_features:
            self._remove_stores(parameters)
        return class_counts

    def plot_class_distribution(
        self,
        parameters_update: Dict = None,
        frame_cutoff: int = 1,
        bout_cutoff: int = 1,
        print_full: bool = False,
        remove_saved_features: bool = False,
    ) -> None:
        """
        Make a class distribution plot

        You can either specify the parameters, choose an existing dataset or do neither (in that case a dataset
        is created or laoded for the computation with the default parameters).

        Parameters
        ----------
        parameters_update : dict, optional
            a dictionary of parameter updates (only for "data" and "general" categories)
        remove_saved_features : bool, default False
            if `True`, the dataset that is used for computation is then deleted
        """

        task, parameters = self._make_task_training(
            episode_name="_", parameters_update=parameters_update, throwaway=True
        )
        cutoff = {True: bout_cutoff, False: frame_cutoff}
        for bouts in [True, False]:
            class_counts = task.count_classes(bouts=bouts)
            if print_full:
                print("Bouts:" if bouts else "Frames:")
                for k, v in class_counts.items():
                    if sum(v.values()) != 0:
                        print(f"  {k}:")
                        values, keys = zip(
                            *[
                                x
                                for x in sorted(zip(v.values(), v.keys()), reverse=True)
                                if x[-1] != -100
                            ]
                        )
                        for kk, vv in zip(keys, values):
                            print(f"    {task.behaviors_dict()[kk]}: {vv}")
            class_counts = {
                kk: {k: v for k, v in vv.items() if v >= cutoff[bouts]}
                for kk, vv in class_counts.items()
            }
            for key, d in class_counts.items():
                if sum(d.values()) != 0:
                    values, keys = zip(
                        *[x for x in sorted(zip(d.values(), d.keys())) if x[-1] != -100]
                    )
                    keys = [task.behaviors_dict()[x] for x in keys]
                    plt.bar(keys, values)
                    plt.title(key)
                    plt.xticks(rotation=45, ha="right")
                    if bouts:
                        plt.ylabel("bouts")
                    else:
                        plt.ylabel("frames")
                    plt.tight_layout()
                    plt.show()
        if remove_saved_features:
            self._remove_stores(parameters)

    def _generate_mask(
        self,
        mask_name: str,
        perc_annotated: float = 0.1,
        parameters_update: Dict = None,
        remove_saved_features: bool = False,
    ) -> None:
        """
        Generate a real_lens for active learning simulation

        Parameters
        ----------
        mask_name : str
            the name of the real_lens
        """

        print(f"GENERATING {mask_name}")
        task, parameters = self._make_task_training(
            f"_{mask_name}", parameters_update=parameters_update, throwaway=True
        )
        val_intervals, val_ids = task.dataset("val").get_intervals()  # 1
        unannotated_intervals = task.dataset("train").get_unannotated_intervals()  # 2
        unannotated_intervals = task.dataset("val").get_unannotated_intervals(
            first_intervals=unannotated_intervals
        )
        ids = task.dataset("train").get_ids()
        mask = {video_id: {} for video_id in ids}
        total_all = 0
        total_masked = 0
        for video_id, clip_ids in ids.items():
            for clip_id in clip_ids:
                frames = np.ones(task.dataset("train").get_len(video_id, clip_id))
                if clip_id in val_intervals[video_id]:
                    for start, end in val_intervals[video_id][clip_id]:
                        frames[start:end] = 0
                if clip_id in unannotated_intervals[video_id]:
                    for start, end in unannotated_intervals[video_id][clip_id]:
                        frames[start:end] = 0
                annotated = np.where(frames)[0]
                total_all += len(annotated)
                masked = annotated[-int(len(annotated) * (1 - perc_annotated)) :]
                total_masked += len(masked)
                mask[video_id][clip_id] = self._get_intervals(masked)
        file = {
            "masked": mask,
            "val_intervals": val_intervals,
            "val_ids": val_ids,
            "unannotated": unannotated_intervals,
        }
        self._save_mask(file, mask_name)
        if remove_saved_features:
            self._remove_stores(parameters)
        print("\n")
        # print(f'Unmasked: {sum([(vv == 0).sum() for v in real_lens.values() for vv in v.values()])} frames')

    def _get_intervals(self, frame_indices: np.ndarray):
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
        if len(frame_indices) > 0:
            breaks = np.where(np.diff(frame_indices) != 1)[0]
            start = frame_indices[0]
            for k in breaks:
                masked_intervals.append([start, frame_indices[k] + 1])
                start = frame_indices[k + 1]
            masked_intervals.append([start, frame_indices[-1] + 1])
        return masked_intervals

    def _update_mask_with_uncertainty(
        self,
        mask_name: str,
        episode_name: Union[str, None],
        classes: List,
        load_epoch: int = None,
        n_frames: int = 10000,
        method: str = "least_confidence",
        min_length: int = 30,
        augment_n: int = 0,
        parameters_update: Dict = None,
    ):
        """
        Update real_lens with frame-wise uncertainty scores for active learning

        Parameters
        ----------
        mask_name : str
            the name of the real_lens
        episode_name : str
            the name of the episode to load
        classes : list
            a list of class names or indices; their uncertainty scores will be computed separately and stacked
        n_frames : int, default 10000
            the number of frames to "annotate"
        method : {"least_confidence", "entropy"}
            the method used to calculate the scores from the probability predictions (`"least_confidence"`: `1 - p_i` if
            `p_i > 0.5` or `p_i` if `p_i < 0.5`; `"entropy"`: `- p_i * log(p_i) - (1 - p_i) * log(1 - p_i)`)
        min_length : int
            the minimum length (in frames) of the annotated intervals
        augment_n : int, default 0
            the number of augmentations to average over
        parameters_update : dict, optional
            the dictionary used to update the parameters from the config

        Returns
        -------
        score_dicts : dict
            a nested dictionary where first level keys are video ids, second level keys are clip ids and values
            are score tensors
        """

        print(f"UPDATING {mask_name}")
        task, parameters, _ = self._make_task_prediction(
            prediction_name=mask_name,
            load_episode=episode_name,
            parameters_update=parameters_update,
            load_epoch=load_epoch,
            mode="train",
        )
        score_tensors = task.generate_uncertainty_score(classes, augment_n, method)
        self._update_mask(task, mask_name, score_tensors, n_frames, min_length)
        print("\n")

    def _update_mask_with_BALD(
        self,
        mask_name: str,
        episode_name: str,
        classes: List,
        load_epoch: int = None,
        augment_n: int = 0,
        n_frames: int = 10000,
        num_models: int = 10,
        kernel_size: int = 11,
        min_length: int = 30,
        parameters_update: Dict = None,
    ):
        """
        Update real_lens with frame-wise Bayesian Active Learning by Disagreement scores for active learning

        Parameters
        ----------
        mask_name : str
            the name of the real_lens
        episode_name : str
            the name of the episode to load
        classes : list
            a list of class names or indices; their uncertainty scores will be computed separately and stacked
        augment_n : int, default 0
            the number of augmentations to average over
        n_frames : int, default 10000
            the number of frames to "annotate"
        num_models : int, default 10
            the number of dropout masks to apply
        kernel_size : int, default 11
            the size of the smoothing gaussian kernel
        min_length : int
            the minimum length (in frames) of the annotated intervals
        parameters_update : dict, optional
            the dictionary used to update the parameters from the config

        Returns
        -------
        score_dicts : dict
            a nested dictionary where first level keys are video ids, second level keys are clip ids and values
            are score tensors
        """

        print(f"UPDATING {mask_name}")
        task, parameters, mode = self._make_task_prediction(
            mask_name,
            load_episode=episode_name,
            parameters_update=parameters_update,
            load_epoch=load_epoch,
        )
        score_tensors = task.generate_bald_score(
            classes, augment_n, num_models, kernel_size
        )
        self._update_mask(task, mask_name, score_tensors, n_frames, min_length)
        print("\n")

    def _suggest_intervals(
        self,
        dataset: BehaviorDataset,
        score_tensors: Dict,
        n_frames: int,
        min_length: int,
    ) -> Dict:
        """
        Suggest intervals with highest score of total length `n_frames`

        Parameters
        ----------
        dataset : BehaviorDataset
            the dataset
        score_tensors : dict
            a dictionary where keys are clip ids and values are framewise score tensors
        n_frames : int
            the number of frames to "annotate"
        min_length : int

        Returns
        -------
        active_learning_intervals : Dict
            active learning dictionary with suggested intervals
        """

        video_intervals, _ = dataset.get_intervals()
        taken = {
            video_id: defaultdict(lambda: {}) for video_id in video_intervals.keys()
        }
        annotated = dataset.get_annotated_intervals()
        for video_id in video_intervals:
            for clip_id in video_intervals[video_id]:
                taken[video_id][clip_id] = torch.zeros(
                    dataset.get_len(video_id, clip_id)
                )
                if video_id in annotated and clip_id in annotated[video_id]:
                    for start, end in annotated[video_id][clip_id]:
                        score_tensors[video_id][clip_id][:, start:end] = -10
                        taken[video_id][clip_id][int(start) : int(end)] = 1
        n_frames = (
            sum([(vv == 1).sum() for v in taken.values() for vv in v.values()])
            + n_frames
        )
        factor = 1
        threshold_start = float(
            torch.mean(
                torch.tensor(
                    [
                        torch.mean(
                            torch.tensor([torch.mean(y[y > 0]) for y in x.values()])
                        )
                        for x in score_tensors.values()
                    ]
                )
            )
        )
        while (
            sum([(vv == 1).sum() for v in taken.values() for vv in v.values()])
            < n_frames
        ):
            threshold = threshold_start * factor
            intervals = []
            interval_scores = []
            key1 = list(score_tensors.keys())[0]
            key2 = list(score_tensors[key1].keys())[0]
            num_scores = score_tensors[key1][key2].shape[0]
            for i in range(num_scores):
                v_dict = dataset.find_valleys(
                    predicted=score_tensors,
                    threshold=threshold,
                    min_frames=min_length,
                    main_class=i,
                    low=False,
                )
                for v_id, interval_list in v_dict.items():
                    intervals += [x + [v_id] for x in interval_list]
                    interval_scores += [
                        float(torch.mean(score_tensors[v_id][clip_id][i, start:end]))
                        for start, end, clip_id in interval_list
                    ]
            intervals = np.array(intervals)[np.argsort(interval_scores)[::-1]]
            i = 0
            while sum(
                [(vv == 1).sum() for v in taken.values() for vv in v.values()]
            ) < n_frames and i < len(intervals):
                start, end, clip_id, video_id = intervals[i]
                i += 1
                taken[video_id][clip_id][int(start) : int(end)] = 1
            factor *= 0.9
            if factor < 0.05:
                warnings.warn(f"Could not find enough frames!")
                break
        active_learning_intervals = {video_id: [] for video_id in video_intervals}
        for video_id in taken:
            for clip_id in taken[video_id]:
                if video_id in annotated and clip_id in annotated[video_id]:
                    for start, end in annotated[video_id][clip_id]:
                        taken[video_id][clip_id][int(start) : int(end)] = 0
                if (taken[video_id][clip_id] == 1).sum() == 0:
                    continue
                indices = np.where(taken[video_id][clip_id].numpy())[0]
                boundaries = self._get_intervals(indices)
                active_learning_intervals[video_id] += [
                    [start, end, clip_id] for start, end in boundaries
                ]
        return active_learning_intervals

    def _update_mask(
        self,
        task: TaskDispatcher,
        mask_name: str,
        score_tensors: Dict,
        n_frames: int,
        min_length: int,
    ) -> None:
        """
        Update the real_lens with intervals with the highest score of total length `n_frames`

        Parameters
        ----------
        mask_name : str
            the name of the real_lens
        score_tensors : dict
            a dictionary where keys are clip ids and values are framewise score tensors
        n_frames : int
            the number of frames to "annotate"
        min_length : int
            the minimum length of the annotated intervals
        """

        mask = self._load_mask(mask_name)
        video_intervals, _ = task.dataset("train").get_intervals()
        masked = {
            video_id: defaultdict(lambda: {}) for video_id in video_intervals.keys()
        }
        total_masked = 0
        total_all = 0
        for video_id in video_intervals:
            for clip_id in video_intervals[video_id]:
                masked[video_id][clip_id] = torch.zeros(
                    task.dataset("train").get_len(video_id, clip_id)
                )
                if (
                    video_id in mask["unannotated"]
                    and clip_id in mask["unannotated"][video_id]
                ):
                    for start, end in mask["unannotated"][video_id][clip_id]:
                        score_tensors[video_id][clip_id][:, start:end] = -10
                        masked[video_id][clip_id][int(start) : int(end)] = 1
                if (
                    video_id in mask["val_intervals"]
                    and clip_id in mask["val_intervals"][video_id]
                ):
                    for start, end in mask["val_intervals"][video_id][clip_id]:
                        score_tensors[video_id][clip_id][:, start:end] = -10
                        masked[video_id][clip_id][int(start) : int(end)] = 1
                total_all += torch.sum(masked[video_id][clip_id] == 0)
                if video_id in mask["masked"] and clip_id in mask["masked"][video_id]:
                    # print(f'{real_lens["masked"][video_id][clip_id]=}')
                    for start, end in mask["masked"][video_id][clip_id]:
                        masked[video_id][clip_id][int(start) : int(end)] = 1
                        total_masked += end - start
        old_n_frames = sum(
            [(vv == 0).sum() for v in masked.values() for vv in v.values()]
        )
        n_frames = old_n_frames + n_frames
        factor = 1
        while (
            sum([(vv == 0).sum() for v in masked.values() for vv in v.values()])
            < n_frames
        ):
            threshold = float(
                torch.mean(
                    torch.tensor(
                        [
                            torch.mean(
                                torch.tensor([torch.mean(y[y > 0]) for y in x.values()])
                            )
                            for x in score_tensors.values()
                        ]
                    )
                )
            )
            threshold = threshold * factor
            intervals = []
            interval_scores = []
            key1 = list(score_tensors.keys())[0]
            key2 = list(score_tensors[key1].keys())[0]
            num_scores = score_tensors[key1][key2].shape[0]
            for i in range(num_scores):
                v_dict = task.dataset("train").find_valleys(
                    predicted=score_tensors,
                    threshold=threshold,
                    min_frames=min_length,
                    main_class=i,
                    low=False,
                )
                for v_id, interval_list in v_dict.items():
                    intervals += [x + [v_id] for x in interval_list]
                    interval_scores += [
                        float(torch.mean(score_tensors[v_id][clip_id][i, start:end]))
                        for start, end, clip_id in interval_list
                    ]
            intervals = np.array(intervals)[np.argsort(interval_scores)[::-1]]
            i = 0
            while sum(
                [(vv == 0).sum() for v in masked.values() for vv in v.values()]
            ) < n_frames and i < len(intervals):
                start, end, clip_id, video_id = intervals[i]
                i += 1
                masked[video_id][clip_id][int(start) : int(end)] = 0
            factor *= 0.9
            if factor < 0.05:
                warnings.warn(f"Could not find enough frames!")
                break
        mask["masked"] = {video_id: {} for video_id in video_intervals}
        total_masked_new = 0
        for video_id in masked:
            for clip_id in masked[video_id]:
                if (
                    video_id in mask["unannotated"]
                    and clip_id in mask["unannotated"][video_id]
                ):
                    for start, end in mask["unannotated"][video_id][clip_id]:
                        masked[video_id][clip_id][int(start) : int(end)] = 0
                if (
                    video_id in mask["val_intervals"]
                    and clip_id in mask["val_intervals"][video_id]
                ):
                    for start, end in mask["val_intervals"][video_id][clip_id]:
                        masked[video_id][clip_id][int(start) : int(end)] = 0
                indices = np.where(masked[video_id][clip_id].numpy())[0]
                mask["masked"][video_id][clip_id] = self._get_intervals(indices)
        for video_id in mask["masked"]:
            for clip_id in mask["masked"][video_id]:
                for start, end in mask["masked"][video_id][clip_id]:
                    total_masked_new += end - start
        self._save_mask(mask, mask_name)
        with open(
            os.path.join(self.project_path, "results", f"{mask_name}.txt"), "a"
        ) as f:
            f.write(f"from {total_masked} to {total_masked_new} / {total_all}" + "\n")
        print(f"Unmasked from {total_masked} to {total_masked_new} / {total_all}")

    def plot_confusion_matrix(
        self,
        episode_name: str,
        load_epoch: int = None,
        parameters_update: Dict = None,
        type: str = "recall",
        mode: str = "val",
        remove_saved_features: bool = False,
    ) -> Tuple[ndarray, Iterable]:
        """
        Make a confusion matrix plot and return the data

        If the annotation is non-exclusive, only false positive labels are considered.

        Parameters
        ----------
        episode_name : str
            the name of the episode to load
        load_epoch : int, optional
            the index of the epoch to load (by default the last one is loaded)
        parameters_update : dict, optional
            a dictionary of parameter updates (only for "data" and "general" categories)
        mode : {'val', 'all', 'test', 'train'}
            the subset of the data to make the prediction for (forced to 'all' if data_path is not None)
        type : {"recall", "precision"}
            for datasets with non-exclusive annotation, if `type` is `"recall"`, only false positives are taken
            into account, and if `type` is `"precision"`, only false negatives
        remove_saved_features : bool, default False
            if `True`, the dataset that is used for computation is then deleted

        Returns
        -------
        confusion_matrix : np.ndarray
            a confusion matrix of shape `(#classes, #classes)` where `A[i, j] = F_ij/N_i`, `F_ij` is the number of
            frames that have the i-th label in the ground truth and a false positive j-th label in the prediction,
            `N_i` is the number of frames that have the i-th label in the ground truth
        classes : list
            a list of labels
        """

        task, parameters, mode = self._make_task_prediction(
            "_",
            load_episode=episode_name,
            load_epoch=load_epoch,
            parameters_update=parameters_update,
            mode=mode,
        )
        dataset = task.dataset(mode)
        prediction = task.predict(dataset, raw_output=True)
        confusion_matrix, classes, type = dataset.get_confusion_matrix(prediction, type)
        if remove_saved_features:
            self._remove_stores(parameters)
        fig, ax = plt.subplots(figsize=(len(classes), len(classes)))
        ax.imshow(confusion_matrix)
        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticks(np.arange(len(classes)))
        ax.set_yticklabels(classes)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(
                    j,
                    i,
                    np.round(confusion_matrix[i, j], 2),
                    ha="center",
                    va="center",
                    color="w",
                )
        if type is not None:
            ax.set_title(f"{type} {episode_name}")
        else:
            ax.set_title(episode_name)
        fig.tight_layout()
        plt.show()
        return confusion_matrix, classes

    def plot_predictions(
        self,
        episode_name: str,
        load_epoch: int = None,
        parameters_update: Dict = None,
        add_legend: bool = True,
        ground_truth: bool = True,
        colormap: str = "viridis",
        hide_axes: bool = False,
        min_classes: int = 1,
        width: float = 10,
        whole_video: bool = False,
        transparent: bool = False,
        drop_classes: Set = None,
        search_classes: Set = None,
        num_plots: int = 1,
        remove_saved_features: bool = False,
        smooth_interval_prediction: int = 0,
        data_path: str = None,
        file_paths: Set = None,
        mode: str = "val",
        behavior_name: str = None,
    ) -> None:
        """
        Visualize random predictions

        Parameters
        ----------
        episode_name : str
            the name of the episode to load
        load_epoch : int, optional
            the epoch to load (by default last)
        parameters_update : dict, optional
            parameter update dictionary
        add_legend : bool, default True
            if True, legend will be added to the plot
        ground_truth : bool, default True
            if True, ground truth will be added to the plot
        colormap : str, default 'Accent'
            the `matplotlib` colormap to use
        hide_axes : bool, default True
            if `True`, the axes will be hidden on the plot
        min_classes : int, default 1
            the minimum number of classes in a displayed interval
        width : float, default 10
            the width of the plot
        whole_video : bool, default False
            if `True`, whole videos are plotted instead of segments
        transparent : bool, default False
            if `True`, the background on the plot is transparent
        drop_classes : set, optional
            a set of class names to not be displayed
        search_classes : set, optional
            if given, only intervals where at least one of the classes is in ground truth will be shown
        num_plots : int, default 1
            the number of plots to make
        remove_saved_features : bool, default False
            if `True`, the dataset will be deleted after computation
        smooth_interval_prediction : int, default 0
            if >0, predictions shorter than this number of frames are removed (filled with prediction for the previous frame)
        data_path : str, optional
            the data path to run the prediction for
        mode : {'all', 'test', 'val', 'train'}
            the subset of the data to make the prediction for (forced to 'all' if data_path is not None)
        file_paths : set, optional
            a set of string file paths (data with all prefixes + feature files, in any order) to run the prediction
            for
        behavior_name : str, optional
            for non-exclusive classificaton datasets, choose which behavior to visualize (by default first in list)
        """

        other_path = os.path.join(self.project_path, "results", "other")
        task, parameters, mode = self._make_task_prediction(
            "_",
            load_episode=episode_name,
            parameters_update=parameters_update,
            load_epoch=load_epoch,
            data_path=data_path,
            file_paths=file_paths,
            mode=mode,
        )
        if not os.path.exists(other_path):
            os.mkdir(other_path)
        for i in range(num_plots):
            task.visualize_results(
                save_path=os.path.join(
                    other_path, f"{episode_name}_prediction_{i}.jpg"
                ),
                add_legend=add_legend,
                ground_truth=ground_truth,
                colormap=colormap,
                hide_axes=hide_axes,
                min_classes=min_classes,
                whole_video=whole_video,
                transparent=transparent,
                dataset=mode,
                drop_classes=drop_classes,
                search_classes=search_classes,
                width=width,
                smooth_interval_prediction=smooth_interval_prediction,
                behavior_name=behavior_name,
            )
        if remove_saved_features:
            self._remove_stores(parameters)

    def create_metadata_backup(self) -> None:
        """
        Create a copy of the meta files
        """

        meta_copy_path = os.path.join(self.project_path, "meta", "backup")
        meta_path = os.path.join(self.project_path, "meta")
        if os.path.exists(meta_copy_path):
            shutil.rmtree(meta_copy_path)
        os.mkdir(meta_copy_path)
        for file in os.listdir(meta_path):
            if file == "backup":
                continue
            shutil.copy(
                os.path.join(meta_path, file), os.path.join(meta_copy_path, file)
            )

    def load_metadata_backup(self) -> None:
        """
        Load from previously created meta data backup (in case of corruption)
        """

        meta_copy_path = os.path.join(self.project_path, "meta", "backup")
        meta_path = os.path.join(self.project_path, "meta")
        for file in os.listdir(meta_copy_path):
            shutil.copy(
                os.path.join(meta_copy_path, file), os.path.join(meta_path, file)
            )

    def get_behavior_dictionary(self, episode_name: str) -> Dict:
        """
        Get the behavior dictionary for an episode

        Parameters
        ----------
        episode_name : str
            the name of the episode

        Returns
        -------
        behaviors_dictionary : dict
            a dictionary where keys are label indices and values are label names
        """

        run = self._episodes().get_runs(episode_name)[0]
        return self._episode(run).get_behaviors_dict()

    def import_episodes(
        self,
        episodes_directory: str,
        name_map: Dict = None,
        repeat_policy: str = "error",
    ) -> None:
        """
        Import episodes exported with `Project.export_episodes`

        Parameters
        ----------
        episodes_directory : str
            the path to the exported episodes directory
        name_map : dict
            a name change dictionary for the episodes: keys are old names, values are new names
        """

        if name_map is None:
            name_map = {}
        episodes = pd.read_pickle(os.path.join(episodes_directory, "episodes.pickle"))
        to_remove = []
        import_string = "Imported episodes: "
        for episode_name in episodes.index:
            if episode_name in name_map:
                import_string += f"{episode_name} "
                episode_name = name_map[episode_name]
                import_string += f"({episode_name}), "
            else:
                import_string += f"{episode_name}, "
            try:
                self._check_episode_validity(episode_name, allow_doublecolon=True)
            except ValueError as e:
                if str(e).endswith("is already taken!"):
                    if repeat_policy == "skip":
                        to_remove.append(episode_name)
                    elif repeat_policy == "force":
                        self.remove_episode(episode_name)
                    elif repeat_policy == "error":
                        raise ValueError(
                            f"The {episode_name} episode name is already taken; please use the name_map parameter to rename it"
                        )
                    else:
                        raise ValueError(
                            f"The {repeat_policy} repeat policy is not recognized; please choose from ['skip', 'force' ans 'error']"
                        )
        episodes = episodes.drop(index=to_remove)
        self._episodes().update(
            episodes,
            name_map=name_map,
            force=(repeat_policy == "force"),
            data_path=self.data_path,
            annotation_path=self.annotation_path,
        )
        for episode_name in episodes.index:
            if episode_name in name_map:
                new_episode_name = name_map[episode_name]
            else:
                new_episode_name = episode_name
            model_dir = os.path.join(
                self.project_path, "results", "model", new_episode_name
            )
            old_model_dir = os.path.join(episodes_directory, "model", episode_name)
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
            os.mkdir(model_dir)
            for file in os.listdir(old_model_dir):
                shutil.copyfile(
                    os.path.join(old_model_dir, file), os.path.join(model_dir, file)
                )
            log_file = os.path.join(
                self.project_path, "results", "logs", f"{new_episode_name}.txt"
            )
            old_log_file = os.path.join(
                episodes_directory, "logs", f"{episode_name}.txt"
            )
            shutil.copyfile(old_log_file, log_file)
        print(import_string)
        print("\n")

    def export_episodes(
        self, episode_names: List, output_directory: str, name: str = None
    ) -> None:
        """
        Save selected episodes as a file that can be imported into another project with `Project.import_episodes`

        Parameters
        ----------
        episode_names : list
            a list of string episode names
        output_directory : str
            the path to the directory where the episodes will be saved
        name : str, optional
            the name of the episodes directory (by default `exported_episodes`)
        """

        if name is None:
            name = "exported_episodes"
        if os.path.exists(
            os.path.join(output_directory, name + ".zip")
        ) or os.path.exists(os.path.join(output_directory, name)):
            i = 1
            while os.path.exists(
                os.path.join(output_directory, name + f"_{i}.zip")
            ) or os.path.exists(os.path.join(output_directory, name + f"_{i}")):
                i += 1
            name = name + f"_{i}"
        dest_dir = os.path.join(output_directory, name)
        os.mkdir(dest_dir)
        os.mkdir(os.path.join(dest_dir, "model"))
        os.mkdir(os.path.join(dest_dir, "logs"))
        runs = []
        for episode in episode_names:
            runs += self._episodes().get_runs(episode)
        for run in runs:
            shutil.copytree(
                os.path.join(self.project_path, "results", "model", run),
                os.path.join(dest_dir, "model", run),
            )
            shutil.copyfile(
                os.path.join(self.project_path, "results", "logs", f"{run}.txt"),
                os.path.join(dest_dir, "logs", f"{run}.txt"),
            )
        data = self._episodes().get_subset(runs)
        data.to_pickle(os.path.join(dest_dir, "episodes.pickle"))

    def get_results_table(
        self,
        episode_names: List,
        metrics: List = None,
        include_std: bool = False,
        classes: List = None,
    ):
        """
        Genererate a `pandas` dataframe with a summary of episode results

        Parameters
        ----------
        episode_names : list
            a list of names of episodes to include
        metrics : list, optional
            a list of metric names to include
        include_std : bool, default False
            if `True`, for episodes with multiple runs the mean and standard deviation will be displayed;
            otherwise only mean
        classes : list, optional
            a list of names of classes to include (by default all are included)

        Returns
        -------
        results : pd.DataFrame
            a table with the results
        """

        run_names = []
        for episode in episode_names:
            run_names += self._episodes().get_runs(episode)
        episodes = self.list_episodes(run_names, print_results=False)
        metric_columns = [x for x in episodes.columns if x[0] == "results"]
        results_df = pd.DataFrame()
        if metrics is not None:
            metric_columns = [
                x for x in metric_columns if x[1].split("_")[0] in metrics
            ]
        for episode in episode_names:
            results = []
            metric_set = set()
            for run in self._episodes().get_runs(episode):
                beh_dict = self.get_behavior_dictionary(run)
                res_dict = defaultdict(lambda: {})
                for column in metric_columns:
                    if np.isnan(episodes.loc[run, column]):
                        continue
                    split = column[1].split("_")
                    if split[-1].isnumeric():
                        beh_ind = int(split[-1])
                        metric_name = "_".join(split[:-1])
                        beh = beh_dict[beh_ind]
                    else:
                        beh = "average"
                        metric_name = column[1]
                    res_dict[beh][metric_name] = episodes.loc[run, column]
                    metric_set.add(metric_name)
                if "average" not in res_dict:
                    res_dict["average"] = {}
                for metric in metric_set:
                    if metric not in res_dict["average"]:
                        arr = [
                            res_dict[beh][metric]
                            for beh in res_dict
                            if metric in res_dict[beh]
                        ]
                        res_dict["average"][metric] = np.mean(arr)
                results.append(res_dict)
            episode_results = {}
            for metric in metric_set:
                for beh in results[0].keys():
                    if classes is not None and beh not in classes:
                        continue
                    arr = []
                    for res_dict in results:
                        if metric in res_dict[beh]:
                            arr.append(res_dict[beh][metric])
                    if len(arr) > 0:
                        if include_std:
                            episode_results[
                                (beh, f"{episode} {metric} mean")
                            ] = np.mean(arr)
                            episode_results[(beh, f"{episode} {metric} std")] = np.std(
                                arr
                            )
                        else:
                            episode_results[(beh, f"{episode} {metric}")] = np.mean(arr)
            for key, value in episode_results.items():
                results_df.loc[key[0], key[1]] = value
        print(f"RESULTS:")
        print(results_df)
        print("\n")
        return results_df

    def episode_exists(self, episode_name: str) -> bool:
        """
        Check if an episode already exists

        Parameters
        ----------
        episode_name : str
            the episode name

        Returns
        -------
        exists : bool
            `True` if the episode exists
        """

        return self._episodes().check_name_validity(episode_name)

    def search_exists(self, search_name: str) -> bool:
        """
        Check if a search already exists

        Parameters
        ----------
        search_name : str
            the search name

        Returns
        -------
        exists : bool
            `True` if the search exists
        """

        return self._searches().check_name_validity(search_name)

    def prediction_exists(self, prediction_name: str) -> bool:
        """
        Check if a prediction already exists

        Parameters
        ----------
        prediction_name : str
            the prediction name

        Returns
        -------
        exists : bool
            `True` if the prediction exists
        """

        return self._predictions().check_name_validity(prediction_name)

    @staticmethod
    def project_name_available(projects_path: str, project_name: str):
        if projects_path is None:
            projects_path = os.path.join(str(Path.home()), "DLC2Action")
        return not os.path.exists(os.path.join(projects_path, project_name))

    def _update_episode_metrics(self, episode_name: str, metrics: Dict):
        """
        Update meta data with evaluation results
        """

        self._episodes().update_episode_metrics(episode_name, metrics)

    def rename_episode(self, episode_name: str, new_episode_name: str):
        shutil.move(
            os.path.join(self.project_path, "results", "model", episode_name),
            os.path.join(self.project_path, "results", "model", new_episode_name),
        )
        shutil.move(
            os.path.join(self.project_path, "results", "logs", f"{episode_name}.txt"),
            os.path.join(
                self.project_path, "results", "logs", f"{new_episode_name}.txt"
            ),
        )
        self._episodes().rename_episode(episode_name, new_episode_name)


class _Runner:
    """
    A helper class for running hyperparameter searches
    """

    def __init__(
        self,
        search_name,
        search_space: Dict,
        load_episode: str,
        load_epoch: int,
        metric: str,
        average: int,
        task: Union[TaskDispatcher, None],
        remove_saved_features: bool,
        project: Project,
    ):
        """
        Parameters
        ----------
        search_space : dict
            a dictionary representing the search space; of this general structure:
            {'group/param_name': ('float/int/float_log/int_log', start, end),
            'group/param_name': ('categorical', [choices])}, e.g.
            {'data/overlap': ('int', 5, 100), 'training/lr': ('float_log', 1e-4, 1e-2),
            'data/feature_extraction': ('categorical', ['kinematic', 'bones'])}
        load_episode : str
            the name of the episode to load the model from
        load_epoch : int
            the epoch to load the model from (if not provided, the last checkpoint is used)
        metric : str
            the metric to maximize/minimize (see direction)
        average : int
            the number of epochs to average the metric; if 0, the last value is taken
        remove_saved_features : bool
            if `True`, the old datasets will be deleted when data parameters change
        project : Project
            the parent `Project` instance
        """

        self.search_space = search_space
        self.load_episode = load_episode
        self.load_epoch = load_epoch
        self.metric = metric
        self.average = average
        self.feature_save_path = None
        self.remove_saved_featuress = remove_saved_features
        self.save_stores = project._save_stores
        self.remove_datasets = project.remove_saved_features
        self.task = task
        self.search_name = search_name
        self.update = project._update
        self.remove_episode = project.remove_episode
        self.fill = project._fill

    def clean(self):
        """
        Remove datasets if needed
        """

        if self.remove_saved_featuress:
            self.remove_datasets([os.path.basename(self.feature_save_path)])

    def run(self, trial, parameters):
        """
        Make a trial run
        """

        params = deepcopy(parameters)
        param_update = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {})))
        )
        for full_name, space in self.search_space.items():
            group, param_name = full_name.split("/")[0], "/".join(
                full_name.split("/")[1:]
            )
            log = space[0][-3:] == "log"
            if space[0].startswith("int"):
                value = trial.suggest_int(full_name, space[1], space[2], log=log)
            elif space[0].startswith("float"):
                value = trial.suggest_float(full_name, space[1], space[2], log=log)
            elif space[0] == "categorical":
                value = trial.suggest_categorical(full_name, space[1])
            else:
                raise ValueError(
                    "The search space has to be formatted as either "
                    '("float"/"int"/"float_log"/"int_log", start, end) '
                    f'or ("categorical", [choices]); got {space} for {group}/{param_name}'
                )
            if len(param_name.split("/")) == 1:
                param_update[group][param_name] = value
            else:
                pars = param_name.split("/")
                pars = [int(x) if x.isnumeric() else x for x in pars]
                if len(pars) == 2:
                    param_update[group][pars[0]][pars[1]] = value
                elif len(pars) == 3:
                    param_update[group][pars[0]][pars[1]][pars[2]] = value
                elif len(pars) == 4:
                    param_update[group][pars[0]][pars[1]][pars[2]][pars[3]] = value
        params = self.update(params, param_update)
        self.remove_episode(f"_{self.search_name}")
        params = self.fill(
            params,
            f"_{self.search_name}",
            self.load_episode,
            load_epoch=self.load_epoch,
            only_load_model=True,
        )
        if self.feature_save_path != params["data"]["feature_save_path"]:
            if self.feature_save_path is not None:
                self.clean()
            self.feature_save_path = params["data"]["feature_save_path"]
        self.save_stores(params)
        if self.task is None:
            self.task = TaskDispatcher(deepcopy(params))
        else:
            self.task.update_task(params)

        _, metrics_log = self.task.train(trial, self.metric)
        metric_values = metrics_log["val"][self.metric]
        if self.average > 0:
            value = np.mean(sorted(metric_values)[-self.average :])
        else:
            value = metric_values[-1]
        return value
