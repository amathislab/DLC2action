#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
"""
Handling meta (history) files
"""

import os
from typing import Dict, List, Tuple, Set, Union
import pandas as pd
from time import localtime, strftime
from copy import deepcopy
import numpy as np
from collections import defaultdict
import warnings
from dlc2action.utils import correct_path


class Run:
    """
    A class that manages operations with a single episode record
    """

    def __init__(
        self,
        episode_name: str,
        project_path: str,
        meta_path: str = None,
        params: Dict = None,
    ):
        """
        Parameters
        ----------
        episode_name : str
            the name of the episode
        meta_path : str, optional
            the path to the pickled SavedRuns dataframe
        params : dict, optional
            alternative to meta_path: pre-loaded pandas Series of episode parameters
        """

        self.name = episode_name
        self.project_path = project_path
        if meta_path is not None:
            try:
                self.params = pd.read_pickle(meta_path).loc[episode_name]
            except:
                raise ValueError(f"The {episode_name} episode does not exist!")
        elif params is not None:
            self.params = params
        else:
            raise ValueError("Either meta_path or params has to be not None")

    def training_time(self) -> int:
        """
        Get the training time in seconds

        Returns
        -------
        training_time : int
            the training time in seconds
        """

        time_str = self.params["meta"].get("training_time")
        try:
            if time_str is None or np.isnan(time_str):
                return np.nan
        except TypeError:
            pass
        h, m, s = time_str.split(":")
        seconds = int(h) * 3600 + int(m) * 60 + int(s)
        return seconds

    def model_file(self, load_epoch: int = None) -> str:
        """
        Get a checkpoint file path

        Parameters
        ----------
        project_path : str
            the current project folder path
        load_epoch : int, optional
            the epoch to load (the closest checkpoint will be chosen; if not given will be set to last)

        Returns
        -------
        checkpoint_path : str
            the path to the checkpoint
        """

        model_path = correct_path(
            self.params["training"]["model_save_path"], self.project_path
        )
        if load_epoch is None:
            model_file = sorted(os.listdir(model_path))[-1]
        else:
            model_files = os.listdir(model_path)
            if len(model_files) == 0:
                model_file = None
            else:
                epochs = [int(file[5:].split(".")[0]) for file in model_files]
                diffs = [np.abs(epoch - load_epoch) for epoch in epochs]
                argmin = np.argmin(diffs)
                model_file = model_files[argmin]
        model_file = os.path.join(model_path, model_file)
        return model_file

    def dataset_name(self) -> str:
        """
        Get the dataset name

        Returns
        -------
        dataset_name : str
            the name of the dataset record
        """

        data_path = correct_path(
            self.params["data"]["feature_save_path"], self.project_path
        )
        dataset_name = os.path.basename(data_path)
        return dataset_name

    def split_file(self) -> str:
        """
        Get the split file

        Returns
        -------
        split_path : str
            the path to the split file
        """

        return correct_path(self.params["training"]["split_path"], self.project_path)

    def log_file(self) -> str:
        """
        Get the log file

        Returns
        -------
        log_path : str
            the path to the log file
        """

        return correct_path(self.params["training"]["log_file"], self.project_path)

    def split_info(self) -> Dict:
        """
        Get the train/test/val split information

        Returns
        -------
        split_info : dict
            a dictionary with [val_frac, test_frac, partition_method] keys and corresponding values
        """

        val_frac = self.params["training"]["val_frac"]
        test_frac = self.params["training"]["test_frac"]
        partition_method = self.params["training"]["partition_method"]
        return {
            "val_frac": val_frac,
            "test_frac": test_frac,
            "partition_method": partition_method,
        }

    def same_split_info(self, split_info: Dict) -> bool:
        """
        Check whether this episode has the same split information

        Parameters
        ----------
        split_info : dict
            a dictionary with [val_frac, test_frac, partition_method] keys and corresponding values from another episode

        Returns
        -------
        result : bool
            if True, this episode has the same split information
        """

        self_split_info = self.split_info()
        for k in ["val_frac", "test_frac", "partition_method"]:
            if self_split_info[k] != split_info[k]:
                return False
        return True

    def get_metrics(self) -> List:
        """
        Get a list of tracked metrics

        Returns
        -------
        metrics : list
            a list of tracked metric names
        """

        return self.params["general"]["metric_functions"]

    def get_metric_log(self, mode: str, metric_name: str) -> np.ndarray:
        """
        Get the metric log

        Parameters
        ----------
        mode : {'train', 'val'}
            the mode to get the log from
        metric_name : str
            the metric to get the log for (has to be one of the metric computed for this episode during training)

        Returns
        -------
        log : np.ndarray
            the log of metric values (empty if the metric was not computed during training)
        """

        metric_array = []
        with open(self.log_file()) as f:
            for line in f.readlines():
                if mode == "train" and line.startswith("[epoch"):
                    line = line.split("]: ")[1]
                elif mode == "val" and line.startswith("validation"):
                    line = line.split("validation: ")[1]
                else:
                    continue
                metrics = line.split(", ")
                for metric in metrics:
                    name, value = metric.split()
                    if name == metric_name:
                        metric_array.append(float(value))
        return np.array(metric_array)

    def get_epoch_list(self, mode) -> List:
        """
        Get a list of epoch indices

        Returns
        -------
        epoch_list : list
            a list of int epoch indices
        """

        epoch_list = []
        with open(self.log_file()) as f:
            for line in f.readlines():
                if line.startswith("[epoch"):
                    epoch = int(line[7:].split("]:")[0])
                    if mode == "train":
                        epoch_list.append(epoch)
                elif mode == "val":
                    epoch_list.append(epoch)
        return epoch_list

    def get_metrics(self) -> List:
        """
        Get a list of metric names in the episode log

        Returns
        -------
        metrics : List
            a list of string metric names
        """

        metrics = []
        with open(self.log_file()) as f:
            for line in f.readlines():
                if line.startswith("[epoch"):
                    line = line.split("]: ")[1]
                elif line.startswith("validation"):
                    line = line.split("validation: ")[1]
                else:
                    continue
                metric_logs = line.split(", ")
                for metric in metric_logs:
                    name, _ = metric.split()
                    metrics.append(name)
                break
        return metrics

    def unfinished(self) -> bool:
        """
        Check whether this episode was interrupted

        Returns
        -------
        result : bool
            True if the number of epochs in the log file is smaller than in the parameters
        """

        num_epoch_theor = self.params["training"]["num_epochs"]
        log_file = self.log_file()
        if not isinstance(log_file, str):
            return False
        if not os.path.exists(log_file):
            return True
        with open(self.log_file()) as f:
            num_epoch = 0
            val = False
            for line in f.readlines():
                num_epoch += 1
                if num_epoch == 2 and line.startswith("validation"):
                    val = True
            if val:
                num_epoch //= 2
        return num_epoch < num_epoch_theor

    def get_class_ind(self, class_name: str) -> int:
        """
        Get the integer label from a class name

        Parameters
        ----------
        class_name : str
            the name of the class

        Returns
        -------
        class_ind : int
            the integer label
        """

        behaviors_dict = self.params["meta"]["behaviors_dict"]
        for k, v in behaviors_dict.items():
            if v == class_name:
                return k
        raise ValueError(
            f"The {class_name} class is not in classes predicted by {self.name} ({behaviors_dict})"
        )

    def get_behaviors_dict(self) -> Dict:
        """
        Get behaviors dictionary in the episode

        Returns
        -------
        behaviors_dict : dict
            a dictionary with class indices as keys and labels as values
        """

        return self.params["meta"]["behaviors_dict"]

    def get_num_classes(self) -> int:
        """
        Get number of classes in episode

        Returns
        -------
        num_classes : int
            the number of classes
        """

        return len(self.params["meta"]["behaviors_dict"])


class DecisionThresholds:
    """
    A class that saves and looks up tuned decision thresholds
    """

    def __init__(self, path: str) -> None:
        """
        Parameters
        ----------
        path : str
            the path to the pickled SavedRuns dataframe
        """

        self.path = path
        self.data = pd.read_pickle(path)

    def save_thresholds(
        self,
        episode_names: List,
        epochs: List,
        metric_name: str,
        metric_parameters: Dict,
        thresholds: List,
    ) -> None:
        """
        Add a new record

        Parameters
        ----------
        episode_name : str
            the name of the episode
        epoch : int
            the epoch index
        metric_name : str
            the name of the metric the thresholds were tuned on
        metric_parameters : dict
            the metric parameter dictionary
        thresholds : list
            a list of float decision thresholds
        """

        episodes = set(zip(episode_names, epochs))
        for key in ["average", "threshold_value", "ignored_classes"]:
            if key in metric_parameters:
                metric_parameters.pop(key)
        parameters = {(metric_name, k): v for k, v in metric_parameters.items()}
        parameters["thresholds"] = thresholds
        parameters["episodes"] = episodes
        pars = {k: [v] for k, v in parameters.items()}
        self.data = pd.concat([self.data, pd.DataFrame.from_dict(pars)], axis=0)
        self._save()

    def find_thresholds(
        self,
        episode_names: List,
        epochs: List,
        metric_name: str,
        metric_parameters: Dict,
    ) -> Union[List, None]:
        """
        Find a record

        Parameters
        ----------
        episode_name : str
            the name of the episode
        epoch : int
            the epoch index
        metric_name : str
            the name of the metric the thresholds were tuned on
        metric_parameters : dict
            the metric parameter dictionary

        Returns
        -------
        thresholds : list
            a list of float decision thresholds
        """

        episodes = set(zip(episode_names, epochs))
        for key in ["average", "threshold_value", "ignored_classes"]:
            if key in metric_parameters:
                metric_parameters.pop(key)
        parameters = {(metric_name, k): v for k, v in metric_parameters.items()}
        parameters["episodes"] = episodes
        filter = deepcopy(parameters)
        for key, value in parameters.items():
            if value is None:
                filter.pop(key)
            elif key not in self.data.columns:
                return None
        data = self.data[(self.data[list(filter)] == pd.Series(filter)).all(axis=1)]
        if len(data) > 0:
            thresholds = data.iloc[0]["thresholds"]
            return thresholds
        else:
            return None

    def _save(self) -> None:
        """
        Save the records
        """

        self.data.copy().to_pickle(self.path)


class SavedRuns:
    """
    A class that manages operations with all episode (or prediction) records
    """

    def __init__(self, path: str, project_path: str) -> None:
        """
        Parameters
        ----------
        path : str
            the path to the pickled SavedRuns dataframe
        """

        self.path = path
        self.project_path = project_path
        self.data = pd.read_pickle(path)

    def update(
        self,
        data: pd.DataFrame,
        data_path: str,
        annotation_path: str,
        name_map: Dict = None,
        force: bool = False,
    ) -> None:
        """
        Update with new data

        Parameters
        ----------
        data : pd.DataFrame
            the new dataframe
        data_path : str
            the new data path
        annotation_path : str
            the new annotation path
        name_map : dict, optional
            the name change dictionary; keys are old episode names and values are new episode names
        force : bool, default False
            replace existing episodes if `True`
        """

        if name_map is None:
            name_map = {}
        data = data.rename(index=name_map)
        for episode in data.index:
            new_model = os.path.join(self.project_path, "results", "model", episode)
            data.loc[episode, ("training", "model_save_path")] = new_model
            new_log = os.path.join(
                self.project_path, "results", "logs", f"{episode}.txt"
            )
            data.loc[episode, ("training", "log_file")] = new_log
            old_split = data.loc[episode, ("training", "split_path")]
            if old_split is None:
                new_split = None
            else:
                new_split = os.path.join(
                    self.project_path, "results", "splits", os.path.basename(old_split)
                )
            data.loc[episode, ("training", "split_path")] = new_split
            data.loc[episode, ("data", "data_path")] = data_path
            data.loc[episode, ("data", "annotation_path")] = annotation_path
            if episode in self.data.index:
                if force:
                    self.data = self.data.drop(index=[episode])
                else:
                    raise RuntimeError(f"The {episode} episode name is already taken!")
        self.data = pd.concat([self.data, data])
        self._save()

    def get_subset(self, episode_names: List) -> pd.DataFrame:
        """
        Get a subset of the raw metadata

        Parameters
        ----------
        episode_names : list
            a list of the episodes to include
        """

        for episode in episode_names:
            if episode not in self.data.index:
                raise ValueError(
                    f"The {episode} episode is not in the records; please run `Project.list_episodes()` to explore the records"
                )
        return self.data.loc[episode_names]

    def get_saved_data_path(self, episode_name: str) -> str:
        """
        Get the `saved_data_path` parameter for the episode

        Parameters
        ----------
        episode_name : str
            the name of the episode

        Returns
        -------
        saved_data_path : str
            the saved data path
        """

        return self.data.loc[episode_name]["data"]["saved_data_path"]

    def check_name_validity(self, episode_name: str) -> bool:
        """
        Check if an episode name already exists

        Parameters
        ----------
        episode_name : str
            the name to check

        Returns
        -------
        result : bool
            True if the name can be used
        """

        if episode_name in self.data.index:
            return False
        else:
            return True

    def update_episode_metrics(self, episode_name: str, metrics: Dict) -> None:
        """
        Update meta data with evaluation results

        Parameters
        ----------
        episode_name : str
            the name of the episode to update
        metrics : dict
            a dictionary of the metrics
        """

        for key, value in metrics.items():
            self.data.loc[episode_name, ("results", key)] = value
        self._save()

    def save_episode(
        self,
        episode_name: str,
        parameters: Dict,
        behaviors_dict: Dict,
        suppress_validation: bool = False,
        training_time: str = None,
    ) -> None:
        """
        Save a new run record

        Parameters
        ----------
        episode_name : str
            the name of the episode
        parameters : dict
            the parameters to save
        behaviors_dict : dict
            the dictionary of behaviors (keys are indices, values are names)
        suppress_validation : bool, optional False
            if True, existing episode with the same name will be overwritten
        training_time : str, optional
            the training time in '%H:%M:%S' format
        """

        if not suppress_validation and episode_name in self.data.index:
            raise ValueError(f"Episode {episode_name} already exists!")
        pars = deepcopy(parameters)
        if "meta" not in pars:
            pars["meta"] = {
                "time": strftime("%Y-%m-%d %H:%M:%S", localtime()),
                "behaviors_dict": behaviors_dict,
            }
        else:
            pars["meta"]["time"] = strftime("%Y-%m-%d %H:%M:%S", localtime())
            pars["meta"]["behaviors_dict"] = behaviors_dict
        if training_time is not None:
            pars["meta"]["training_time"] = training_time
        if len(parameters.keys()) > 1:
            pars["losses"] = pars["losses"].get(pars["general"]["loss_function"], {})
            for metric_name in pars["general"]["metric_functions"]:
                pars[metric_name] = pars["metrics"].get(metric_name, {})
            if pars["general"].get("ssl", None) is not None:
                for ssl_name in pars["general"]["ssl"]:
                    pars[ssl_name] = pars["ssl"].get(ssl_name, {})
            for group_name in ["metrics", "ssl"]:
                if group_name in pars:
                    pars.pop(group_name)
        data = {
            (big_key, small_key): value
            for big_key, big_value in pars.items()
            for small_key, value in big_value.items()
        }
        list_keys = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")
            for k, v in data.items():
                if k not in self.data.columns:
                    self.data[k] = np.nan
                if isinstance(v, list) and not isinstance(v, str):
                    list_keys.append(k)
            for k in list_keys:
                self.data[k] = self.data[k].astype(object)
            self.data.loc[episode_name] = data
        self._save()

    def load_parameters(self, episode_name: str) -> Dict:
        """
        Load the task parameters from a record

        Parameters
        ----------
        episode_name : str
            the name of the episode to load

        Returns
        -------
        parameters : dict
            the loaded task parameters
        """

        parameters = defaultdict(lambda: defaultdict(lambda: {}))
        episode = self.data.loc[episode_name].dropna().to_dict()
        keys = ["data", "augmentations", "general", "training", "model", "features"]
        for key in episode:
            big_key, small_key = key
            if big_key in keys:
                parameters[big_key][small_key] = episode[key]
        # parameters = {k: dict(v) for k, v in parameters.items()}
        ssl_keys = parameters["general"].get("ssl", None)
        metric_keys = parameters["general"].get("metric_functions", None)
        loss_key = parameters["general"]["loss_function"]
        if ssl_keys is None:
            ssl_keys = []
        if metric_keys is None:
            metric_keys = []
        for key in episode:
            big_key, small_key = key
            if big_key in ssl_keys:
                parameters["ssl"][big_key][small_key] = episode[key]
            elif big_key in metric_keys:
                parameters["metrics"][big_key][small_key] = episode[key]
            elif big_key == "losses":
                parameters["losses"][loss_key][small_key] = episode[key]
        parameters = {k: dict(v) for k, v in parameters.items()}
        parameters["general"]["num_classes"] = Run(
            episode_name, self.project_path, params=self.data.loc[episode_name]
        ).get_num_classes()
        return parameters

    def get_active_datasets(self) -> List:
        """
        Get a list of names of datasets that are used by unfinished episodes

        Returns
        -------
        active_datasets : list
            a list of dataset names used by unfinished episodes
        """

        active_datasets = []
        for episode_name in self.unfinished_episodes():
            run = Run(
                episode_name, self.project_path, params=self.data.loc[episode_name]
            )
            active_datasets.append(run.dataset_name())
        return active_datasets

    def list_episodes(
        self,
        episode_names: List = None,
        value_filter: str = "",
        display_parameters: List = None,
    ) -> pd.DataFrame:
        """
        Get a filtered pandas dataframe with episode metadata

        Parameters
        ----------
        episode_names : List
            a list of strings of episode names
        value_filter : str
            a string of filters to apply of this general structure:
            'group_name1/par_name1::(<>=)value1,group_name2/par_name2::(<>=)value2', e.g.
            'data/overlap::=50,results/recall::>0.5,data/feature_extraction::=kinematic'
        display_parameters : List
            list of parameters to display (e.g. ['data/overlap', 'results/recall'])

        Returns
        -------
        pandas.DataFrame
            the filtered dataframe
        """

        if episode_names is not None:
            data = deepcopy(self.data.loc[episode_names])
        else:
            data = deepcopy(self.data)
        if len(data) == 0:
            return pd.DataFrame()
        try:
            filters = value_filter.split(",")
            if filters == [""]:
                filters = []
            for f in filters:
                par_name, condition = f.split("::")
                group_name, par_name = par_name.split("/")
                sign, value = condition[0], condition[1:]
                if value[0] == "=":
                    sign += "="
                    value = value[1:]
                try:
                    value = float(value)
                except:
                    if value == "True":
                        value = True
                    elif value == "False":
                        value = False
                    elif value == "None":
                        value = None
                if value is None:
                    if sign == "=":
                        data = data[data[group_name][par_name].isna()]
                    elif sign == "!=":
                        data = data[~data[group_name][par_name].isna()]
                elif sign == ">":
                    data = data[data[group_name][par_name] > value]
                elif sign == ">=":
                    data = data[data[group_name][par_name] >= value]
                elif sign == "<":
                    data = data[data[group_name][par_name] < value]
                elif sign == "<=":
                    data = data[data[group_name][par_name] <= value]
                elif sign == "=":
                    data = data[data[group_name][par_name] == value]
                elif sign == "!=":
                    data = data[data[group_name][par_name] != value]
                else:
                    raise ValueError(
                        "Please use one of the signs: [>, <, >=, <=, =, !=]"
                    )
        except ValueError:
            raise ValueError(
                f"The {value_filter} filter is not valid, please use the following format:"
                f" 'group1/parameter1::[sign][value],group2/parameter2::[sign][value]', "
                f"e.g. 'training/num_epochs::>=200,model/num_f_maps::=128,meta/time::>2022-06-01'"
            )
        if display_parameters is not None:
            if type(display_parameters[0]) is str:
                display_parameters = [
                    (x.split("/")[0], x.split("/")[1]) for x in display_parameters
                ]
            display_parameters = [x for x in display_parameters if x in data.columns]
            data = data[display_parameters]
        return data

    def rename_episode(self, episode_name, new_episode_name):
        if episode_name in self.data.index and new_episode_name not in self.data.index:
            self.data.loc[new_episode_name] = self.data.loc[episode_name]
            model_path = self.data.loc[new_episode_name, ("training", "model_path")]
            self.data.loc[new_episode_name, ("training", "model_path")] = os.path.join(
                os.path.dirname(model_path), new_episode_name
            )
            log_path = self.data.loc[new_episode_name, ("training", "log_file")]
            self.data.loc[new_episode_name, ("training", "log_file")] = os.path.join(
                os.path.dirname(log_path), f"{new_episode_name}.txt"
            )
            self.data = self.data.drop(index=episode_name)
            self._save()
        else:
            raise ValueError("The names are wrong")

    def remove_episode(self, episode_name: str) -> None:
        """
        Remove all model, logs and metafile records related to an episode

        Parameters
        ----------
        episode_name : str
            the name of the episode to remove
        """

        if episode_name in self.data.index:
            self.data = self.data.drop(index=episode_name)
            self._save()

    def unfinished_episodes(self) -> List:
        """
        Get a list of unfinished episodes (currently running or interrupted)

        Returns
        -------
        interrupted_episodes: List
            a list of string names of unfinished episodes in the records
        """

        unfinished = []
        for name, params in self.data.iterrows():
            if Run(name, project_path=self.project_path, params=params).unfinished():
                unfinished.append(name)
        return unfinished

    def update_episode_results(
        self,
        episode_name: str,
        logs: Tuple,
        training_time: str = None,
    ) -> None:
        """
        Add results to an episode record

        Parameters
        ----------
        episode_name : str
            the name of the episode to update
        logs : dict
            a log dictionary from task.train()
        training_time : str
            the training time
        """

        metrics_log = logs[1]
        results = {}
        for key, value in metrics_log["val"].items():
            results[("results", key)] = value[-1]
        if training_time is not None:
            results[("meta", "training_time")] = training_time
        for k, v in results.items():
            self.data.loc[episode_name, k] = v
        self._save()

    def get_runs(self, episode_name: str) -> List:
        """
        Get a list of runs with this episode name (episodes like `episode_name::0`)

        Parameters
        ----------
        episode_name : str
            the name of the episode

        Returns
        -------
        runs_list : List
            a list of string run names
        """

        if episode_name is None:
            return []
        index = self.data.index
        runs_list = []
        for name in index:
            if name.startswith(episode_name):
                split = name.split("::")
                if split[0] == episode_name:
                    if len(split) > 1 and split[-1].isnumeric() or len(split) == 1:
                        runs_list.append(name)
                elif name == episode_name:
                    runs_list.append(name)
        return runs_list

    def _save(self):
        """
        Save the dataframe
        """

        self.data.copy().to_pickle(self.path)


class Searches(SavedRuns):
    """
    A class that manages operations with search records
    """

    def save_search(
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
        Save a new search record

        Parameters
        ----------
        search_name : str
            the name of the search to save
        parameters : dict
            the task parameters to save
        n_trials : int
            the number of trials in the search
        best_params : dict
            the best parameters dictionary
        best_value : float
            the best valie
        metric : str
            the name of the objective metric
        search_space : dict
            a dictionary representing the search space; of this general structure:
            {'group/param_name': ('float/int/float_log/int_log', start, end),
            'group/param_name': ('categorical', [choices])}, e.g.
            {'data/overlap': ('int', 5, 100), 'training/lr': ('float_log', 1e-4, 1e-2),
            'data/feature_extraction': ('categorical', ['kinematic', 'bones'])}
        """

        pars = deepcopy(parameters)
        pars["results"] = {"best_value": best_value, "best_params": best_params}
        pars["meta"] = {
            "objective": metric,
            "n_trials": n_trials,
            "search_space": search_space,
        }
        self.save_episode(search_name, pars, {})

    def get_best_params_raw(self, search_name: str) -> Dict:
        """
        Get the raw dictionary of best parameters found by a search

        Parameters
        ----------
        search_name : str
            the name of the search

        Returns
        -------
        best_params : dict
            a dictionary of the best parameters where the keys are in '{group}/{name}' format
        """

        return self.data.loc[search_name]["results"]["best_params"]

    def get_best_params(
        self,
        search_name: str,
        load_parameters: List = None,
        round_to_binary: List = None,
    ) -> Dict:
        """
        Get the best parameters from a search

        Parameters
        ----------
        search_name : str
            the name of the search
        load_parameters : List, optional
            a list of string names of the parameters to load (if not provided all parameters are loaded)
        round_to_binary : List, optional
            a list of string names of the loaded parameters that should be rounded to the nearest power of two

        Returns
        -------
        best_params : dict
            a dictionary of the best parameters
        """

        if round_to_binary is None:
            round_to_binary = []
        params = self.data.loc[search_name]["results"]["best_params"]
        if load_parameters is not None:
            params = {k: v for k, v in params.items() if k in load_parameters}
        for par_name in round_to_binary:
            if par_name not in params:
                continue
            if not isinstance(params[par_name], float) and not isinstance(
                params[par_name], int
            ):
                raise TypeError(
                    f"Cannot round {par_name} parameter of type {type(par_name)} to a power of two"
                )
            i = 1
            while 2**i < params[par_name]:
                i += 1
            if params[par_name] - (2 ** (i - 1)) < (2**i) - params[par_name]:
                params[par_name] = 2 ** (i - 1)
            else:
                params[par_name] = 2**i
        res = defaultdict(lambda: defaultdict(lambda: {}))
        for k, v in params.items():
            big_key, small_key = k.split("/")[0], "/".join(k.split("/")[1:])
            if len(small_key.split("/")) == 1:
                res[big_key][small_key] = v
            else:
                group, key = small_key.split("/")
                res[big_key][group][key] = v
        model = self.data.loc[search_name]["general"]["model_name"]
        return res, model


class Suggestions(SavedRuns):
    def save_suggestion(self, episode_name: str, parameters: Dict, meta_parameters):
        pars = deepcopy(parameters)
        pars["meta"] = meta_parameters
        super().save_episode(episode_name, pars, behaviors_dict=None)


class SavedStores:
    """
    A class that manages operation with saved dataset records
    """

    def __init__(self, path):
        """
        Parameters
        ----------
        path : str
            the path to the pickled SavedRuns dataframe
        """

        self.path = path
        self.data = pd.read_pickle(path)
        self.skip_keys = [
            "feature_save_path",
            "saved_data_path",
            "real_lens",
            "recompute_annotation",
        ]

    def clear(self) -> None:
        """
        Remove all datasets
        """

        for dataset_name in self.data.index:
            self.remove_dataset(dataset_name)

    def dataset_names(self) -> List:
        """
        Get a list of dataset names

        Returns
        -------
        dataset_names : List
            a list of string dataset names
        """

        return list(self.data.index)

    def remove(self, names: List) -> None:
        """
        Remove some datasets

        Parameters
        ----------
        names : List
            a list of string names of the datasets to delete
        """

        for dataset_name in names:
            if dataset_name in self.data.index:
                self.remove_dataset(dataset_name)

    def remove_dataset(self, dataset_name: str) -> None:
        """
        Remove a dataset record

        Parameters
        ----------
        dataset_name : str
            the name of the dataset to remove
        """

        if dataset_name in self.data.index:
            self.data = self.data.drop(index=dataset_name)
            self._save()

    def find_name(self, parameters: Dict) -> str:
        """
        Find a record that satisfies the parameters (if it exists)

        Parameters
        ----------
        parameters : dict
            a dictionary of data parameters

        Returns
        -------
        name : str
            the name of a record that has the same parameters (None if it does not exist; the earliest if there are
            several)
        """

        filter = deepcopy(parameters)
        for key, value in parameters.items():
            if value is None or key in self.skip_keys:
                filter.pop(key)
            elif key not in self.data.columns:
                return None
        saved_annotation = self.data[
            (self.data[list(filter)] == pd.Series(filter)).all(axis=1)
        ]
        for i in range(len(saved_annotation)):
            ok = True
            for key in saved_annotation.columns:
                if key in self.skip_keys:
                    continue
                isnull = pd.isnull(saved_annotation.iloc[i][key])
                if not isinstance(isnull, bool):
                    isnull = False
                if key not in filter and not isnull:
                    ok = False
            if ok:
                name = saved_annotation.iloc[i].name
                return name
        return None

    def save_store(self, episode_name: str, parameters: Dict) -> None:
        """
        Save a new saved dataset record

        Parameters
        ----------
        episode_name : str
            the name of the dataset
        parameters : dict
            a dictionary of data parameters
        """

        pars = deepcopy(parameters)
        for k, v in parameters.items():
            if k not in self.data.columns:
                self.data[k] = np.nan
        if self.find_name(pars) is None:
            self.data.loc[episode_name] = pars
        self._save()

    def _save(self):
        """
        Save the dataframe
        """

        self.data.to_pickle(self.path)

    def check_name_validity(self, store_name: str) -> bool:
        """
        Check if a store name already exists

        Parameters
        ----------
        episode_name : str
            the name to check

        Returns
        -------
        result : bool
            True if the name can be used
        """

        if store_name in self.data.index:
            return False
        else:
            return True
