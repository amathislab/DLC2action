#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
"""
Class that provides an interface for `dlc2action.task.universal_task.Task`
"""

import inspect
from typing import Dict, Union, Tuple, List, Callable, Set
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from collections.abc import Iterable, Mapping
import torch
from copy import deepcopy
import numpy as np
from optuna.trial import Trial
import warnings

from dlc2action.data.dataset import BehaviorDataset
from dlc2action.task.universal_task import Task
from dlc2action.transformer.base_transformer import Transformer
from dlc2action.ssl.base_ssl import SSLConstructor
from dlc2action.ssl.base_ssl import EmptySSL
from dlc2action.model.base_model import LoadedModel, Model
from dlc2action.metric.base_metric import Metric
from dlc2action.utils import PostProcessor

from dlc2action import options


class TaskDispatcher:
    """
    A class that manages the interactions between config dictionaries and a Task
    """

    def __init__(self, parameters: Dict) -> None:
        """
        Parameters
        ----------
        parameters : dict
            a dictionary of task parameters
        """

        pars = deepcopy(parameters)
        self.class_weights = None
        self.general_parameters = pars.get("general", {})
        self.data_parameters = pars.get("data", {})
        self.model_parameters = pars.get("model", {})
        self.training_parameters = pars.get("training", {})
        self.loss_parameters = pars.get("losses", {})
        self.metric_parameters = pars.get("metrics", {})
        self.ssl_parameters = pars.get("ssl", {})
        self.aug_parameters = pars.get("augmentations", {})
        self.feature_parameters = pars.get("features", {})
        self.blanks = {blank: [] for blank in options.blanks}

        self.task = None
        self._initialize_task()
        self._print_behaviors()

    @staticmethod
    def complete_function_parameters(parameters, function, general_dicts: List) -> Dict:
        """
        Complete a parameter dictionary with values from other dictionaries if required by a function

        Parameters
        ----------
        parameters : dict
            the function parameters dictionary
        function : callable
            the function to be inspected
        general_dicts : list
            a list of dictionaries where the missing values will be pulled from
        """

        parameter_names = inspect.getfullargspec(function).args
        for param in parameter_names:
            for dic in general_dicts:
                if param not in parameters and param in dic:
                    parameters[param] = dic[param]
        return parameters

    @staticmethod
    def complete_dataset_parameters(
        parameters: dict,
        general_dict: dict,
        data_type: str,
        annotation_type: str,
    ) -> Dict:
        """
        Complete a parameter dictionary with values from other dictionaries if required by a dataset

        Parameters
        ----------
        parameters : dict
            the function parameters dictionary
        general_dict : dict
            the dictionary where the missing values will be pulled from
        data_type : str
            the input type of the dataset
        annotation_type : str
            the annotation type of the dataset

        Returns
        -------
        parameters : dict
            the updated parameter dictionary
        """

        params = deepcopy(parameters)
        parameter_names = BehaviorDataset.get_parameters(data_type, annotation_type)
        for param in parameter_names:
            if param not in params and param in general_dict:
                params[param] = general_dict[param]
        return params

    @staticmethod
    def check(parameters: Dict, name: str) -> bool:
        """
        Check whether there is a non-`None` value under the name key in the parameters dictionary

        Parameters
        ----------
        parameters : dict
            the dictionary to check
        name : str
            the key to check

        Returns
        -------
        result : bool
            True if a non-`None` value exists
        """

        if name in parameters and parameters[name] is not None:
            return True
        else:
            return False

    @staticmethod
    def get(parameters: Dict, name: str, default):
        """
        Get the value under the name key or the default if it is `None` or does not exist

        Parameters
        ----------
        parameters : dict
            the dictionary to check
        name : str
            the key to check
        default
            the default value to return

        Returns
        -------
        value
            the resulting value
        """

        if TaskDispatcher.check(parameters, name):
            return parameters[name]
        else:
            return default

    @staticmethod
    def make_dataloader(
        dataset: BehaviorDataset, batch_size: int = 32, shuffle: bool = False
    ) -> DataLoader:
        """
        Make a torch dataloader from a dataset

        Parameters
        ----------
        dataset : dlc2action.data.dataset.BehaviorDataset
            the dataset
        batch_size : int
            the batch size

        Returns
        -------
        dataloader : DataLoader
            the dataloader (or `None` if the length of the dataset is 0)
        """

        if dataset is None or len(dataset) == 0:
            return None
        else:
            return DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle)

    def _construct_ssl(self) -> List:
        """
        Generate SSL constructors
        """

        ssl_list = deepcopy(self.general_parameters.get("ssl", None))
        if not isinstance(ssl_list, Iterable):
            ssl_list = [ssl_list]
        for i, ssl in enumerate(ssl_list):
            if type(ssl) is str:
                if ssl in options.ssl_constructors:
                    pars = self.get(self.ssl_parameters, ssl, default={})
                    pars = self.complete_function_parameters(
                        parameters=pars,
                        function=options.ssl_constructors[ssl],
                        general_dicts=[
                            self.model_parameters,
                            self.data_parameters,
                            self.general_parameters,
                        ],
                    )
                    ssl_list[i] = options.ssl_constructors[ssl](**pars)
                else:
                    raise ValueError(
                        f"The {ssl} SSL is not available, please choose from {list(options.ssl_constructors.keys())}"
                    )
            elif ssl is None:
                ssl_list[i] = EmptySSL()
            elif not isinstance(ssl, SSLConstructor):
                raise TypeError(
                    f"The ssl parameter has to be a list of either strings, SSLConstructor instances or None, got {type(ssl)}"
                )
        return ssl_list

    def _construct_model(self) -> Model:
        """
        Generate a model
        """

        if self.check(self.general_parameters, "model"):
            pars = self.complete_function_parameters(
                function=LoadedModel,
                parameters=self.model_parameters,
                general_dicts=[self.general_parameters],
            )
            model = LoadedModel(**pars)
        elif self.check(self.general_parameters, "model_name"):
            name = self.general_parameters["model_name"]
            if name in options.models:
                pars = self.complete_function_parameters(
                    function=options.models[name],
                    parameters=self.model_parameters,
                    general_dicts=[self.general_parameters],
                )
                model = options.models[name](**pars)
            else:
                raise ValueError(
                    f"The {name} model is not available, please choose from {list(options.models.keys())}"
                )
        else:
            raise ValueError(
                "You need to provide either a model or its name in the model_parameters!"
            )

        if self.get(self.training_parameters, "freeze_features", False):
            model.freeze_feature_extractor()
        return model

    def _construct_dataset(self) -> BehaviorDataset:
        """
        Generate a dataset
        """

        data_type = self.general_parameters.get("data_type", None)
        if data_type is None:
            raise ValueError(
                "You need to provide the data_type parameter in the data parameters!"
            )
        annotation_type = self.get(self.general_parameters, "annotation_type", "none")
        feature_extraction = self.general_parameters.get("feature_extraction", "none")
        if feature_extraction is None:
            raise ValueError(
                "You need to provide the feature_extraction parameter in the data parameters!"
            )
        feature_extraction_pars = self.complete_function_parameters(
            self.feature_parameters,
            options.feature_extractors[feature_extraction],
            [self.general_parameters, self.data_parameters],
        )

        pars = self.complete_dataset_parameters(
            self.data_parameters,
            self.general_parameters,
            data_type=data_type,
            annotation_type=annotation_type,
        )
        pars["feature_extraction_pars"] = feature_extraction_pars
        dataset = BehaviorDataset(**pars)

        if self.get(self.general_parameters, "save_dataset", default=False):
            save_data_path = self.data_parameters.get("saved_data_path", None)
            dataset.save(save_path=save_data_path)

        return dataset

    def _construct_transformer(self) -> Transformer:
        """
        Generate a transformer
        """

        features = self.general_parameters["feature_extraction"]
        name = options.extractor_to_transformer[features]
        if name in options.transformers:
            transformer_class = options.transformers[name]
            pars = self.complete_function_parameters(
                function=transformer_class,
                parameters=self.aug_parameters,
                general_dicts=[self.general_parameters],
            )
            transformer = transformer_class(**pars)
        else:
            raise ValueError(f"The {name} transformer is not available")
        return transformer

    def _construct_loss(self) -> torch.nn.Module:
        """
        Generate a loss function
        """

        if "loss_function" not in self.general_parameters:
            raise ValueError(
                'Please add a "loss_function" key to the parameters["general"] dictionary (either a name '
                f"from {list(options.losses.keys())} or a function)"
            )
        else:
            loss_function = self.general_parameters["loss_function"]
        if type(loss_function) is str:
            if loss_function in options.losses:
                pars = self.get(self.loss_parameters, loss_function, default={})
                pars = self._set_loss_weights(pars)
                pars = self.complete_function_parameters(
                    function=options.losses[loss_function],
                    parameters=pars,
                    general_dicts=[self.general_parameters],
                )
                loss = options.losses[loss_function](**pars)
            else:
                raise ValueError(
                    f"The {loss_function} loss is not available, please choose from {list(options.losses.keys())}"
                )
        else:
            loss = loss_function
        return loss

    def _construct_metrics(self) -> List:
        """
        Generate the metric
        """

        metric_functions = self.get(
            self.general_parameters, "metric_functions", default={}
        )
        if isinstance(metric_functions, Iterable):
            metrics = {}
            for func in metric_functions:
                if isinstance(func, str):
                    if func in options.metrics:
                        pars = self.get(self.metric_parameters, func, default={})
                        pars = self.complete_function_parameters(
                            function=options.metrics[func],
                            parameters=pars,
                            general_dicts=[self.general_parameters],
                        )
                        metrics[func] = options.metrics[func](**pars)
                    else:
                        raise ValueError(
                            f"The {func} metric is not available, please choose from {list(options.metrics.keys())}"
                        )
                elif isinstance(func, Metric):
                    name = "function_1"
                    i = 1
                    while name in metrics:
                        i += 1
                        name = f"function_{i}"
                    metrics[name] = func
                else:
                    raise TypeError(
                        'The elements of parameters["general"]["metric_functions"] have to be either strings '
                        f"from {list(options.metrics.keys())} or Metric instances; got {type(func)} instead"
                    )
        elif isinstance(metric_functions, dict):
            metrics = metric_functions
        else:
            raise TypeError(
                'The value at parameters["general"]["metric_functions"] can be either list, dictionary or None;'
                f"got {type(metric_functions)} instead"
            )
        return metrics

    def _construct_optimizer(self) -> Optimizer:
        """
        Generate an optimizer
        """

        if "optimizer" in self.training_parameters:
            name = self.training_parameters["optimizer"]
            if name in options.optimizers:
                optimizer = options.optimizers[name]
            else:
                raise ValueError(
                    f"The {name} optimizer is not available, please choose from {list(options.optimizers.keys())}"
                )
        else:
            optimizer = None
        return optimizer

    def _construct_predict_functions(self) -> Tuple[Callable, Callable]:
        """
        Construct predict functions
        """

        predict_function = self.training_parameters.get("predict_function", None)
        primary_predict_function = self.training_parameters.get(
            "primary_predict_function", None
        )
        model_name = self.general_parameters.get("model_name", "")
        threshold = self.training_parameters.get("hard_threshold", 0.5)
        if not isinstance(predict_function, Callable):
            if model_name in ["c2f_tcn", "c2f_transformer", "c2f_tcn_p"]:
                if self.general_parameters["exclusive"]:
                    func = lambda x: torch.softmax(x, dim=1)
                else:
                    func = lambda x: torch.sigmoid(x)

                def primary_predict_function(x):
                    if len(x.shape) != 4:
                        x = x.reshape((4, -1, x.shape[-2], x.shape[-1]))
                    weights = [1, 1, 1, 1]
                    ensemble_prob = func(x[0]) * weights[0] / sum(weights)
                    for i, outp_ele in enumerate(x[1:]):
                        ensemble_prob = ensemble_prob + func(outp_ele) * weights[
                            i + 1
                        ] / sum(weights)
                    return ensemble_prob

            else:
                if model_name.startswith("ms_tcn") or model_name in [
                    "asformer",
                    "transformer",
                    "c3d_ms",
                    "transformer_ms",
                ]:
                    f = lambda x: x[-1] if len(x.shape) == 4 else x
                elif model_name == "asrf":

                    def f(x):
                        x = x[-1]
                        # bounds = x[:, 0, :].unsqueeze(1)
                        cls = x[:, 1:, :]
                        # device = x.device
                        # x = PostProcessor("refinement_with_boundary")._refinement_with_boundary(cls.detach().cpu().numpy(), bounds.detach().cpu().numpy())
                        # x = torch.tensor(x).to(device)
                        return cls

                elif model_name == "actionclip":

                    def f(x):
                        video_embedding, text_embedding, logit_scale = (
                            x["video"],
                            x["text"],
                            x["logit_scale"],
                        )
                        B, Ff, T = video_embedding.shape
                        video_embedding = video_embedding.permute(0, 2, 1).reshape(
                            (B * T, -1)
                        )
                        video_embedding /= video_embedding.norm(dim=-1, keepdim=True)
                        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
                        similarity = logit_scale * video_embedding @ text_embedding.T
                        similarity = similarity.reshape((B, T, -1)).permute(0, 2, 1)
                        return similarity

                else:
                    f = lambda x: x
                if self.general_parameters["exclusive"]:
                    primary_predict_function = lambda x: torch.softmax(f(x), dim=1)
                else:
                    primary_predict_function = lambda x: torch.sigmoid(f(x))
            if self.general_parameters["exclusive"]:
                predict_function = lambda x: torch.max(x.data, dim=1)[1]
            else:
                predict_function = lambda x: (x > threshold).int()
        return primary_predict_function, predict_function

    def _get_parameters_from_training(self) -> Dict:
        """
        Get the training parameters that need to be passed to the Task
        """

        task_training_par_names = [
            "lr",
            "parallel",
            "device",
            "verbose",
            "log_file",
            "augment_train",
            "augment_val",
            "hard_threshold",
            "ssl_losses",
            "model_save_path",
            "model_save_epochs",
            "pseudolabel",
            "pseudolabel_start",
            "correction_interval",
            "pseudolabel_alpha_f",
            "alpha_growth_stop",
            "num_epochs",
            "validation_interval",
            "ignore_tags",
            "skip_metrics",
        ]
        task_training_pars = {
            name: self.training_parameters[name]
            for name in task_training_par_names
            if self.check(self.training_parameters, name)
        }
        if self.check(self.general_parameters, "ssl"):
            ssl_weights = [
                self.training_parameters["ssl_weights"][x]
                for x in self.general_parameters["ssl"]
            ]
            task_training_pars["ssl_weights"] = ssl_weights
        return task_training_pars

    def _update_parameters_from_ssl(self, ssl_list: list) -> None:
        """
        Update the necessary parameters given the list of SSL constructors
        """

        if self.task is not None:
            self.task.set_ssl_transformations([ssl.transformation for ssl in ssl_list])
            self.task.set_ssl_losses([ssl.loss for ssl in ssl_list])
            self.task.set_keep_target_none(
                [ssl.type in ["contrastive"] for ssl in ssl_list]
            )
            self.task.set_generate_ssl_input(
                [ssl.type == "contrastive" for ssl in ssl_list]
            )
        self.data_parameters["ssl_transformations"] = [
            ssl.transformation for ssl in ssl_list
        ]
        self.training_parameters["ssl_losses"] = [ssl.loss for ssl in ssl_list]
        self.model_parameters["ssl_types"] = [ssl.type for ssl in ssl_list]
        self.model_parameters["ssl_modules"] = [
            ssl.construct_module() for ssl in ssl_list
        ]
        self.aug_parameters["generate_ssl_input"] = [
            x.type == "contrastive" for x in ssl_list
        ]
        self.aug_parameters["keep_target_none"] = [
            x.type == "contrastive" for x in ssl_list
        ]

    def _set_loss_weights(self, parameters):
        """
        Replace the `"dataset_inverse_weights"` blank in loss parameters with class weight values
        """

        for k in list(parameters.keys()):
            if parameters[k] in [
                "dataset_inverse_weights",
                "dataset_proportional_weights",
            ]:
                if parameters[k] == "dataset_inverse_weights":
                    parameters[k] = self.class_weights
                else:
                    parameters[k] = self.proportional_class_weights
                print("Initializing class weights:")
                string = "    "
                if isinstance(parameters[k], Mapping):
                    for key, val in parameters[k].items():
                        string += ": ".join(
                            (
                                " " + str(key),
                                ", ".join((map(lambda x: str(np.round(x, 3)), val))),
                            )
                        )
                else:
                    string += ", ".join(
                        (map(lambda x: str(np.round(x, 3)), parameters[k]))
                    )
                print(string)
        return parameters

    def _partition_dataset(
        self, dataset: BehaviorDataset
    ) -> Tuple[BehaviorDataset, BehaviorDataset, BehaviorDataset]:
        """
        Partition the dataset into train, validation and test subsamples
        """

        use_test = self.get(self.training_parameters, "use_test", 0)
        split_path = self.training_parameters.get("split_path", None)
        partition_method = self.training_parameters.get("partition_method", "random")
        val_frac = self.get(self.training_parameters, "val_frac", 0)
        test_frac = self.get(self.training_parameters, "test_frac", 0)
        save_split = self.get(self.training_parameters, "save_split", True)
        normalize = self.get(self.training_parameters, "normalize", False)
        skip_normalization_keys = self.training_parameters.get(
            "skip_normalization_keys"
        )
        stats = self.training_parameters.get("stats")
        train_dataset, test_dataset, val_dataset = dataset.partition_train_test_val(
            use_test,
            split_path,
            partition_method,
            val_frac,
            test_frac,
            save_split,
            normalize,
            skip_normalization_keys,
            stats,
        )
        bs = int(self.training_parameters.get("batch_size", 32))
        train_dataloader, test_dataloader, val_dataloader = (
            self.make_dataloader(train_dataset, batch_size=bs, shuffle=True),
            self.make_dataloader(test_dataset, batch_size=bs, shuffle=False),
            self.make_dataloader(val_dataset, batch_size=bs, shuffle=False),
        )
        return train_dataloader, test_dataloader, val_dataloader

    def _initialize_task(self):
        """
        Create a `dlc2action.task.universal_task.Task` instance
        """

        dataset = self._construct_dataset()
        self._update_data_blanks(dataset)
        model = self._construct_model()
        self._update_model_blanks(model)
        ssl_list = self._construct_ssl()
        self._update_parameters_from_ssl(ssl_list)
        model.set_ssl(ssl_constructors=ssl_list)
        dataset.set_ssl_transformations([ssl.transformation for ssl in ssl_list])
        transformer = self._construct_transformer()
        metrics = self._construct_metrics()
        optimizer = self._construct_optimizer()
        primary_predict_function, predict_function = self._construct_predict_functions()

        task_training_pars = self._get_parameters_from_training()
        train_dataloader, test_dataloader, val_dataloader = self._partition_dataset(
            dataset
        )
        self.class_weights = train_dataloader.dataset.class_weights()
        self.proportional_class_weights = train_dataloader.dataset.class_weights(True)
        loss = self._construct_loss()
        exclusive = self.general_parameters["exclusive"]

        task_pars = {
            "train_dataloader": train_dataloader,
            "model": model,
            "loss": loss,
            "transformer": transformer,
            "metrics": metrics,
            "val_dataloader": val_dataloader,
            "test_dataloader": test_dataloader,
            "exclusive": exclusive,
            "optimizer": optimizer,
            "predict_function": predict_function,
            "primary_predict_function": primary_predict_function,
        }
        task_pars.update(task_training_pars)

        self.task = Task(**task_pars)
        checkpoint_path = self.training_parameters.get("checkpoint_path", None)
        if checkpoint_path is not None:
            only_model = self.get(self.training_parameters, "only_load_model", False)
            load_strict = self.get(self.training_parameters, "load_strict", True)
            self.task.load_from_checkpoint(checkpoint_path, only_model, load_strict)
        if (
            self.general_parameters["only_load_annotated"]
            and self.general_parameters.get("ssl") is not None
        ):
            warnings.warn(
                "Note that you are using SSL modules but only loading annotated files! Set "
                "general/only_load_annotated to False to change that"
            )

    def _update_data_blanks(
        self, dataset: BehaviorDataset = None, remember: bool = False
    ) -> None:
        """
        Update all blanks from a dataset
        """

        if dataset is None:
            dataset = self.dataset()
        self._update_dim_parameter(dataset, remember)
        self._update_bodyparts_parameter(dataset, remember)
        self._update_num_classes_parameter(dataset, remember)
        self._update_len_segment_parameter(dataset, remember)
        self._update_boundary_parameter(dataset, remember)

    def _update_model_blanks(self, model: Model, remember: bool = False) -> None:
        self._update_features_parameter(model, remember)

    def _update_parameter(self, blank_name: str, value, remember: bool = False):
        parameters = [
            self.model_parameters,
            self.ssl_parameters,
            self.general_parameters,
            self.feature_parameters,
            self.data_parameters,
            self.training_parameters,
            self.metric_parameters,
            self.loss_parameters,
            self.aug_parameters,
        ]
        par_names = [
            "model",
            "ssl",
            "general",
            "feature",
            "data",
            "training",
            "metrics",
            "losses",
            "augmentations",
        ]
        for names in self.blanks[blank_name]:
            group = names[0]
            key = names[1]
            ind = par_names.index(group)
            if len(names) == 3:
                if names[2] in parameters[ind][key]:
                    parameters[ind][key][names[2]] = value
            else:
                if key in parameters[ind]:
                    parameters[ind][key] = value
        for name, dic in zip(par_names, parameters):
            for k, v in dic.items():
                if v == blank_name:
                    dic[k] = value
                    if [name, k] not in self.blanks[blank_name]:
                        self.blanks[blank_name].append([name, k])
                elif isinstance(v, Mapping):
                    for kk, vv in v.items():
                        if vv == blank_name:
                            dic[k][kk] = value
                            if [name, k, kk] not in self.blanks[blank_name]:
                                self.blanks[blank_name].append([name, k, kk])

    def _update_features_parameter(self, model: Model, remember: bool = False) -> None:
        """
        Fill the `"model_features"` blank
        """

        value = model.features_shape()
        self._update_parameter("model_features", value, remember)

    def _update_bodyparts_parameter(
        self, dataset: BehaviorDataset, remember: bool = False
    ) -> None:
        """
        Fill the `"dataset_bodyparts"` blank
        """

        value = dataset.bodyparts_order()
        self._update_parameter("dataset_bodyparts", value, remember)

    def _update_dim_parameter(
        self, dataset: BehaviorDataset, remember: bool = False
    ) -> None:
        """
        Fill the `"dataset_features"` blank
        """

        value = dataset.features_shape()
        self._update_parameter("dataset_features", value, remember)

    def _update_boundary_parameter(
        self, dataset: BehaviorDataset, remember: bool = False
    ) -> None:
        """
        Fill the `"dataset_features"` blank
        """

        value = dataset.boundary_class_weight()
        self._update_parameter("dataset_boundary_weight", value, remember)

    def _update_num_classes_parameter(
        self, dataset: BehaviorDataset, remember: bool = False
    ) -> None:
        """
        Fill in the `"dataset_classes"` blank
        """

        value = dataset.num_classes()
        self._update_parameter("dataset_classes", value, remember)

    def _update_len_segment_parameter(
        self, dataset: BehaviorDataset, remember: bool = False
    ) -> None:
        """
        Fill in the `"dataset_len_segment"` blank
        """

        value = dataset.len_segment()
        self._update_parameter("dataset_len_segment", value, remember)

    def _print_behaviors(self):
        behavior_set = self.behaviors_dict()
        print(f"Behavior indices:")
        for key, value in sorted(behavior_set.items()):
            print(f"    {key}: {value}")

    def update_task(self, parameters: Dict) -> None:
        """
        Update the `dlc2action.task.universal_task.Task` instance given the parameter updates

        Parameters
        ----------
        parameters : dict
            the dictionary of parameter updates
        """

        pars = deepcopy(parameters)
        # for blank_name in self.blanks:
        #     for names in self.blanks[blank_name]:
        #         group = names[0]
        #         key = names[1]
        #         if len(names) == 3:
        #             if (
        #                 group in pars
        #                 and key in pars[group]
        #                 and names[2] in pars[group][key]
        #             ):
        #                 pars[group][key].pop(names[2])
        #         else:
        #             if group in pars and key in pars[group]:
        #                 pars[group].pop(key)
        stay = False
        if "ssl" in pars:
            for key in pars["ssl"]:
                if key in self.ssl_parameters:
                    self.ssl_parameters[key].update(pars["ssl"][key])
                else:
                    self.ssl_parameters[key] = pars["ssl"][key]

        if "general" in pars:
            if stay:
                stay = False
            if (
                "model_name" in pars["general"]
                and pars["general"]["model_name"]
                != self.general_parameters["model_name"]
            ):
                if "model" not in pars:
                    raise ValueError(
                        "When updating a task with a new model name you need to pass the parameters for the "
                        "new model"
                    )
                self.model_parameters = {}
            self.general_parameters.update(pars["general"])
            data_related = [
                "num_classes",
                "exclusive",
                "data_type",
                "annotation_type",
            ]
            ssl_related = ["ssl", "exclusive", "num_classes"]
            loss_related = ["num_classes", "loss_function", "exclusive"]
            augmentation_related = ["augmentation_type"]
            metric_related = ["metric_functions"]
            related_lists = [
                data_related,
                ssl_related,
                loss_related,
                augmentation_related,
                metric_related,
            ]
            names = ["data", "ssl", "losses", "augmentations", "metrics"]
            for related_list, name in zip(related_lists, names):
                if (
                    any([x in pars["general"] for x in related_list])
                    and name not in pars
                ):
                    pars[name] = {}

        if "training" in pars:
            if "data" not in pars or not stay:
                for x in [
                    "to_ram",
                    "use_test",
                    "partition_method",
                    "val_frac",
                    "test_frac",
                    "save_split",
                    "batch_size",
                    "save_split",
                ]:
                    if (
                        x in pars["training"]
                        and pars["training"][x] != self.training_parameters[x]
                    ):
                        if "data" not in pars:
                            pars["data"] = {}
                        stay = True
            self.training_parameters.update(pars["training"])
            self.task.update_parameters(self._get_parameters_from_training())

        if "data" in pars or "features" in pars:
            for k, v in pars["data"].items():
                if k not in self.data_parameters or v != self.data_parameters[k]:
                    stay = True
            for k, v in pars["features"].items():
                if k not in self.feature_parameters or v != self.feature_parameters[k]:
                    stay = True
            if stay:
                self.data_parameters.update(pars["data"])
                self.feature_parameters.update(pars["features"])
                dataset = self._construct_dataset()
                (
                    train_dataloader,
                    test_dataloader,
                    val_dataloader,
                ) = self._partition_dataset(dataset)
                self.task.set_dataloaders(
                    train_dataloader, val_dataloader, test_dataloader
                )
                self.class_weights = train_dataloader.dataset.class_weights()
                self.proportional_class_weights = (
                    train_dataloader.dataset.class_weights(True)
                )
                if "losses" not in pars:
                    pars["losses"] = {}

        if "model" in pars:
            self.model_parameters.update(pars["model"])

        self._update_data_blanks()

        if "augmentations" in pars:
            self.aug_parameters.update(pars["augmentations"])
            transformer = self._construct_transformer()
            self.task.set_transformer(transformer)

        if "losses" in pars:
            for key in pars["losses"]:
                if key in self.loss_parameters:
                    self.loss_parameters[key].update(pars["losses"][key])
                else:
                    self.loss_parameters[key] = pars["losses"][key]
            self.loss_parameters.update(pars["losses"])
            loss = self._construct_loss()
            self.task.set_loss(loss)

        if "metrics" in pars:
            for key in pars["metrics"]:
                if key in self.metric_parameters:
                    self.metric_parameters[key].update(pars["metrics"][key])
                else:
                    self.metric_parameters[key] = pars["metrics"][key]
            metrics = self._construct_metrics()
            self.task.set_metrics(metrics)

        self.task.set_ssl_transformations(self.data_parameters["ssl_transformations"])
        self._set_loss_weights(
            pars.get("losses", {}).get(self.general_parameters["loss_function"], {})
        )
        model = self._construct_model()
        predict_functions = self._construct_predict_functions()
        self.task.set_predict_functions(*predict_functions)
        self._update_model_blanks(model)
        ssl_list = self._construct_ssl()
        self._update_parameters_from_ssl(ssl_list)
        model.set_ssl(ssl_constructors=ssl_list)
        self.task.set_ssl_transformations([ssl.transformation for ssl in ssl_list])
        self.task.set_model(model)
        if "training" in pars and "checkpoint_path" in pars["training"]:
            checkpoint_path = pars["training"]["checkpoint_path"]
            only_model = pars["training"].get("only_load_model", False)
            load_strict = pars["training"].get("load_strict", True)
            self.task.load_from_checkpoint(checkpoint_path, only_model, load_strict)
        if (
            self.general_parameters["only_load_annotated"]
            and self.general_parameters.get("ssl") is not None
        ):
            warnings.warn(
                "Note that you are using SSL modules but only loading annotated files! Set "
                "general/only_load_annotated to False to change that"
            )
        if self.task.dataset("train").annotation_class() != "none":
            self._print_behaviors()

    def train(
        self,
        trial: Trial = None,
        optimized_metric: str = None,
        autostop_metric: str = None,
        autostop_interval: int = 10,
        autostop_threshold: float = 0.001,
        loading_bar: bool = False,
    ) -> Tuple:
        """
        Train the task and return a log of epoch-average loss and metric

        You can use the autostop parameters to finish training when the parameters are not improving. It will be
        stopped if the average value of `autostop_metric` over the last `autostop_interval` epochs is smaller than
        the average over the previous `autostop_interval` epochs + `autostop_threshold`. For example, if the
        current epoch is 120 and `autostop_interval` is 50, the averages over epochs 70-120 and 20-70 will be compared.

        Parameters
        ----------
        trial : Trial
            an `optuna` trial (for hyperparameter searches)
        optimized_metric : str
            the name of the metric being optimized (for hyperparameter searches)
        to_ram : bool, default False
            if `True`, the dataset will be loaded in RAM (this speeds up the calculations but can lead to crashes
            if the dataset is too large)
        autostop_interval : int, default 50
            the number of epochs to average the autostop metric over
        autostop_threshold : float, default 0.001
            the autostop difference threshold
        autostop_metric : str, optional
            the autostop metric (can be any one of the tracked metrics of `'loss'`)
        main_task_on : bool, default True
            if `False`, the main task (action segmentation) will not be used in training
        ssl_on : bool, default True
            if `False`, the SSL task will not be used in training

        Returns
        -------
        loss_log: list
            a list of float loss function values for each epoch
        metrics_log: dict
            a dictionary of metric value logs (first-level keys are 'train' and 'val', second-level keys are metric
            names, values are lists of function values)
        """

        to_ram = self.training_parameters.get("to_ram", False)
        logs = self.task.train(
            trial,
            optimized_metric,
            to_ram,
            autostop_metric=autostop_metric,
            autostop_interval=autostop_interval,
            autostop_threshold=autostop_threshold,
            main_task_on=self.training_parameters.get("main_task_on", True),
            ssl_on=self.training_parameters.get("ssl_on", True),
            temporal_subsampling_size=self.training_parameters.get(
                "temporal_subsampling_size"
            ),
            loading_bar=loading_bar,
        )
        return logs

    def save_model(self, save_path: str) -> None:
        """
        Save the model of the `dlc2action.task.universal_task.Task` instance

        Parameters
        ----------
        save_path : str
            the path to the saved file
        """

        self.task.save_model(save_path)

    def evaluate(
        self,
        data: Union[DataLoader, BehaviorDataset, str] = None,
        augment_n: int = 0,
        verbose: bool = True,
    ) -> Tuple:
        """
        Evaluate the Task model

        Parameters
        ----------
        data : torch.utils.data.DataLoader | dlc2action.data.dataset.BehaviorDataset, optional
            the data to evaluate on (if not provided, evaluate on the Task validation dataset)
        augment_n : int, default 0
            the number of augmentations to average results over
        verbose : bool, default True
            if True, the process is reported to standard output

        Returns
        -------
        loss : float
            the average value of the loss function
        ssl_loss : float
            the average value of the SSL loss function
        metric : dict
            a dictionary of average values of metric functions
        """

        res = self.task.evaluate(
            data,
            augment_n,
            int(self.training_parameters.get("batch_size", 32)),
            verbose,
        )
        return res

    def evaluate_prediction(
        self,
        prediction: torch.Tensor,
        data: Union[DataLoader, BehaviorDataset, str] = None,
    ) -> Tuple:
        """
        Compute metrics for a prediction

        Parameters
        ----------
        prediction : torch.Tensor
            the prediction
        data : torch.utils.data.DataLoader | dlc2action.data.dataset.BehaviorDataset, optional
            the data the prediction was made for (if not provided, take the validation dataset)

        Returns
        -------
        loss : float
            the average value of the loss function
        metric : dict
            a dictionary of average values of metric functions
        """

        return self.task.evaluate_prediction(
            prediction, data, int(self.training_parameters.get("batch_size", 32))
        )

    def predict(
        self,
        data: Union[DataLoader, BehaviorDataset, str],
        raw_output: bool = False,
        apply_primary_function: bool = True,
        augment_n: int = 0,
        embedding: bool = False,
    ) -> torch.Tensor:
        """
        Make a prediction with the Task model

        Parameters
        ----------
        data : torch.utils.data.DataLoader | dlc2action.data.dataset.BehaviorDataset, optional
            the data to evaluate on (if not provided, evaluate on the Task validation dataset)
        raw_output : bool, default False
            if `True`, the raw predicted probabilities are returned
        apply_primary_function : bool, default True
            if `True`, the primary predict function is applied (to map the model output into a shape corresponding to
            the input)
        augment_n : int, default 0
            the number of augmentations to average results over

        Returns
        -------
        prediction : torch.Tensor
            a prediction for the input data
        """

        to_ram = self.training_parameters.get("to_ram", False)
        return self.task.predict(
            data,
            raw_output,
            apply_primary_function,
            augment_n,
            int(self.training_parameters.get("batch_size", 32)),
            to_ram,
            embedding=embedding,
        )

    def dataset(self, mode: str = "train") -> BehaviorDataset:
        """
        Get a dataset

        Parameters
        ----------
        mode : {'train', 'val', 'test'}
            the dataset to get

        Returns
        -------
        dataset : dlc2action.data.dataset.BehaviorDataset
            the dataset
        """

        return self.task.dataset(mode)

    def generate_full_length_prediction(
        self,
        dataset: Union[BehaviorDataset, str] = None,
        augment_n: int = 10,
    ) -> Dict:
        """
        Compile a prediction for the original input sequences

        Parameters
        ----------
        dataset : dlc2action.data.dataset.BehaviorDataset | str, optional
            the dataset to generate a prediction for (if `None`, generate for the `dlc2action.task.universal_task.Task`
            instance validation dataset)
        augment_n : int, default 10
            the number of augmentations to average results over

        Returns
        -------
        prediction : dict
            a nested dictionary where first level keys are video ids, second level keys are clip ids and values
            are prediction tensors
        """

        return self.task.generate_full_length_prediction(
            dataset, int(self.training_parameters.get("batch_size", 32)), augment_n
        )

    def generate_submission(
        self,
        frame_number_map_file: str,
        dataset: Union[BehaviorDataset, str] = None,
        augment_n: int = 10,
    ) -> Dict:
        """
        Generate a MABe-22 style submission dictionary

        Parameters
        ----------
        frame_number_map_file : str
            path to the frame number map file
        dataset : BehaviorDataset, optional
            the dataset to generate a prediction for (if `None`, generate for the validation dataset)
        augment_n : int, default 10
            the number of augmentations to average results over

        Returns
        -------
        submission : dict
            a dictionary with frame number mapping and embeddings
        """

        return self.task.generate_submission(
            frame_number_map_file,
            dataset,
            int(self.training_parameters.get("batch_size", 32)),
            augment_n,
        )

    def behaviors_dict(self):
        """
        Get a behavior dictionary

        Keys are label indices and values are label names.

        Returns
        -------
        behaviors_dict : dict
            behavior dictionary
        """

        return self.task.behaviors_dict()

    def count_classes(self, bouts: bool = False) -> Dict:
        """
        Get a dictionary of class counts in different modes

        Parameters
        ----------
        bouts : bool, default False
            if `True`, instead of frame counts segment counts are returned

        Returns
        -------
        class_counts : dict
            a dictionary where first-level keys are "train", "val" and "test", second-level keys are
            class names and values are class counts (in frames)
        """

        return self.task.count_classes(bouts)

    def _visualize_results_label(
        self,
        label: str,
        save_path: str = None,
        add_legend: bool = True,
        ground_truth: bool = True,
        hide_axes: bool = False,
        width: int = 10,
        whole_video: bool = False,
        transparent: bool = False,
        dataset: BehaviorDataset = None,
        smooth_interval: int = 0,
        title: str = None,
    ):
        return self.task._visualize_results_label(
            label,
            save_path,
            add_legend,
            ground_truth,
            hide_axes,
            width,
            whole_video,
            transparent,
            dataset,
            smooth_interval=smooth_interval,
            title=title,
        )

    def visualize_results(
        self,
        save_path: str = None,
        add_legend: bool = True,
        ground_truth: bool = True,
        colormap: str = "viridis",
        hide_axes: bool = False,
        min_classes: int = 1,
        width: float = 10,
        whole_video: bool = False,
        transparent: bool = False,
        dataset: Union[BehaviorDataset, DataLoader, str, None] = None,
        drop_classes: Set = None,
        search_classes: Set = None,
        smooth_interval_prediction: int = None,
    ) -> None:
        """
        Visualize random predictions

        Parameters
        ----------
        save_path : str, optional
            the path where the plot will be saved
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
        dataset : BehaviorDataset | DataLoader | str | None, optional
            the dataset to make the prediction for (if not provided, the validation dataset is used)
        drop_classes : set, optional
            a set of class names to not be displayed
        search_classes : set, optional
            if given, only intervals where at least one of the classes is in ground truth will be shown
        """

        return self.task.visualize_results(
            save_path,
            add_legend,
            ground_truth,
            colormap,
            hide_axes,
            min_classes,
            width,
            whole_video,
            transparent,
            dataset,
            drop_classes,
            search_classes,
            smooth_interval_prediction=smooth_interval_prediction,
        )

    def generate_uncertainty_score(
        self,
        classes: List,
        augment_n: int = 0,
        method: str = "least_confidence",
        predicted: torch.Tensor = None,
        behaviors_dict: Dict = None,
    ) -> Dict:
        """
        Generate frame-wise scores for active learning

        Parameters
        ----------
        classes : list
            a list of class names or indices; their confidence scores will be computed separately and stacked
        augment_n : int, default 0
            the number of augmentations to average over
        method : {"least_confidence", "entropy"}
            the method used to calculate the scores from the probability predictions (`"least_confidence"`: `1 - p_i` if
            `p_i > 0.5` or `p_i` if `p_i < 0.5`; `"entropy"`: `- p_i * log(p_i) - (1 - p_i) * log(1 - p_i)`)

        Returns
        -------
        score_dicts : dict
            a nested dictionary where first level keys are video ids, second level keys are clip ids and values
            are score tensors
        """

        return self.task.generate_uncertainty_score(
            classes,
            augment_n,
            int(self.training_parameters.get("batch_size", 32)),
            method,
            predicted,
            behaviors_dict,
        )

    def generate_bald_score(
        self,
        classes: List,
        augment_n: int = 0,
        num_models: int = 10,
        kernel_size: int = 11,
    ) -> Dict:
        """
        Generate frame-wise Bayesian Active Learning by Disagreement scores for active learning

        Parameters
        ----------
        classes : list
            a list of class names or indices; their confidence scores will be computed separately and stacked
        augment_n : int, default 0
            the number of augmentations to average over
        num_models : int, default 10
            the number of dropout masks to apply
        kernel_size : int, default 11
            the size of the smoothing gaussian kernel

        Returns
        -------
        score_dicts : dict
            a nested dictionary where first level keys are video ids, second level keys are clip ids and values
            are score tensors
        """

        return self.task.generate_bald_score(
            classes,
            augment_n,
            int(self.training_parameters.get("batch_size", 32)),
            num_models,
            kernel_size,
        )

    def get_normalization_stats(self) -> Dict:
        """
        Get the pre-computed normalization stats

        Returns
        -------
        normalization_stats : dict
            a dictionary of means and stds
        """

        return self.task.get_normalization_stats()

    def exists(self, mode) -> bool:
        """
        Check whether the task has a train/test/validation subset

        Parameters
        ----------
        mode : {"train", "val", "test"}
            the name of the subset to check for

        Returns
        -------
        exists : bool
            `True` if the subset exists
        """

        dl = self.task.dataloader(mode)
        if dl is None:
            return False
        else:
            return True
