#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
"""
Training and inference
"""

from typing import Callable, Dict, Union, Tuple, List, Set, Any, Optional

import numpy as np
from torch.optim import Optimizer, Adam
from torch import nn
from torch.utils.data import DataLoader
import torch
from collections.abc import Iterable
from tqdm import tqdm
import warnings
from random import randint
from matplotlib import pyplot as plt
from matplotlib import cm
from collections import defaultdict
from dlc2action.transformer.base_transformer import Transformer, EmptyTransformer
from dlc2action.data.dataset import BehaviorDataset
from dlc2action.model.base_model import Model, LoadedModel
from dlc2action.metric.base_metric import Metric
import os
import random
from copy import deepcopy, copy
from optuna.trial import Trial
from optuna import TrialPruned
from math import pi, sqrt, exp, floor, ceil


class MyDataParallel(nn.DataParallel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.process_labels = self.module.process_labels

    def freeze_feature_extractor(self):
        self.module.freeze_feature_extractor()

    def unfreeze_feature_extractor(self):
        self.module.unfreeze_feature_extractor()

    def transform_labels(self, device):
        return self.module.transform_labels(device)

    def logit_scale(self):
        return self.module.logit_scale()

    def main_task_off(self):
        self.module.main_task_off()

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def ssl_on(self):
        self.module.ssl_on()

    def ssl_off(self):
        self.module.ssl_off()

    def extract_features(self, x, start=0):
        return self.module.extract_features(x, start)


class Task:
    """
    A universal trainer class that performs training, evaluation and prediction for all types of tasks and data
    """

    def __init__(
        self,
        train_dataloader: DataLoader,
        model: Union[nn.Module, Model],
        loss: Callable[[torch.Tensor, torch.Tensor], float],
        num_epochs: int = 0,
        transformer: Transformer = None,
        ssl_losses: List = None,
        ssl_weights: List = None,
        lr: float = 1e-3,
        metrics: Dict = None,
        val_dataloader: DataLoader = None,
        test_dataloader: DataLoader = None,
        optimizer: Optimizer = None,
        device: str = "cuda",
        verbose: bool = True,
        log_file: Union[str, None] = None,
        augment_train: int = 1,
        augment_val: int = 0,
        validation_interval: int = 1,
        predict_function: Union[Callable[[torch.Tensor], torch.Tensor], None] = None,
        primary_predict_function: Callable = None,
        exclusive: bool = True,
        ignore_tags: bool = True,
        threshold: float = 0.5,
        model_save_path: str = None,
        model_save_epochs: int = 5,
        pseudolabel: bool = False,
        pseudolabel_start: int = 100,
        correction_interval: int = 2,
        pseudolabel_alpha_f: float = 3,
        alpha_growth_stop: int = 600,
        parallel: bool = False,
        skip_metrics: List = None,
    ) -> None:
        """
        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader
            a training dataloader
        model : dlc2action.model.base_model.Model
            a model
        loss : callable
            a loss function
        num_epochs : int, default 0
            the number of epochs
        transformer : dlc2action.transformer.base_transformer.Transformer, optional
            a transformer
        ssl_losses : list, optional
            a list of SSL losses
        ssl_weights : list, optional
            a list of SSL weights (if not provided initializes to 1)
        lr : float, default 1e-3
            learning rate
        metrics : dict, optional
            a list of metric functions
        val_dataloader : torch.utils.data.DataLoader, optional
            a validation dataloader
        optimizer : torch.optim.Optimizer, optional
            an optimizer (`Adam` by default)
        device : str, default 'cuda'
            the device to train the model on
        verbose : bool, default True
            if `True`, the process is described in standard output
        log_file : str, optional
            the path to a text file where the process will be logged
        augment_train : {1, 0}
            number of augmentations to apply at training
        augment_val : int, default 0
            number of augmentations to apply at validation
        validation_interval : int, default 1
            every time this number of epochs passes, validation metrics are computed
        predict_function : callable, optional
            a function that maps probabilities to class predictions (if not provided, a default is generated)
        primary_predict_function : callable, optional
            a function that maps model output to probabilities (if not provided, initialized as identity)
        exclusive : bool, default True
            set to False for multi-label classification
        ignore_tags : bool, default False
            if `True`, samples with different meta tags will be mixed in batches
        threshold : float, default 0.5
            the threshold used for multi-label classification default prediction function
        model_save_path : str, optional
            the path to the folder where model checkpoints will be saved (checkpoints will not be saved if the path
            is not provided)
        model_save_epochs : int, default 5
            the interval for saving the model checkpoints (the last epoch is always saved)
        pseudolabel : bool, default False
            if True, the pseudolabeling procedure will be applied
        pseudolabel_start : int, default 100
            pseudolabeling starts after this epoch
        correction_interval : int, default 1
            after this number of epochs, if the pseudolabeling is on, the model is trained on the labeled data and
            new pseudolabels are generated
        pseudolabel_alpha_f : float, default 3
            the maximum value of pseudolabeling alpha
        alpha_growth_stop : int, default 600
            pseudolabeling alpha stops growing after this epoch
        """

        # pseudolabeling might be buggy right now -- not using it!
        if skip_metrics is None:
            skip_metrics = []
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.transformer = transformer
        self.num_epochs = num_epochs
        self.skip_metrics = skip_metrics
        self.verbose = verbose
        self.augment_train = int(augment_train)
        self.augment_val = int(augment_val)
        self.ignore_tags = ignore_tags
        self.validation_interval = int(validation_interval)
        self.log_file = log_file
        self.loss = loss
        self.model_save_path = model_save_path
        self.model_save_epochs = model_save_epochs
        self.epoch = 0

        if metrics is None:
            metrics = {}
        self.metrics = metrics

        if optimizer is None:
            optimizer = Adam

        if ssl_weights is None:
            ssl_weights = [1 for _ in ssl_losses]
        if not isinstance(ssl_weights, Iterable):
            ssl_weights = [ssl_weights for _ in ssl_losses]
        self.ssl_weights = ssl_weights

        self.optimizer_class = optimizer
        self.lr = lr
        if not isinstance(model, Model):
            self.model = LoadedModel(model=model)
        else:
            self.set_model(model)
        self.parallel = parallel

        if self.transformer is None:
            self.augment_val = 0
            self.augment_train = 0
            self.transformer = EmptyTransformer()

        if self.augment_train > 1:
            warnings.warn(
                'The "augment_train" parameter is too large -> setting it to 1.'
            )
            self.augment_train = 1

        try:
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)
        except:
            raise ("The format of the device is incorrect")

        if ssl_losses is None:
            self.ssl_losses = [lambda x, y: 0]
        else:
            self.ssl_losses = ssl_losses

        if primary_predict_function is None:
            if exclusive:
                primary_predict_function = lambda x: nn.Softmax(x, dim=1)
            else:
                primary_predict_function = lambda x: torch.sigmoid(x)
        self.primary_predict_function = primary_predict_function

        if predict_function is None:
            if exclusive:
                self.predict_function = lambda x: torch.max(x.data, 1)[1]
            else:
                self.predict_function = lambda x: (x > threshold).int()
        else:
            self.predict_function = predict_function

        self.pseudolabel = pseudolabel
        self.alpha_f = pseudolabel_alpha_f
        self.T2 = alpha_growth_stop
        self.T1 = pseudolabel_start
        self.t = correction_interval
        if self.T2 <= self.T1:
            raise ValueError(
                f"The pseudolabel_start parameter has to be smaller than alpha_growth_stop; got "
                f"{pseudolabel_start=} and {alpha_growth_stop=}"
            )
        self.decision_thresholds = [0.5 for x in self.behaviors_dict()]

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Save a general checkpoint

        Parameters
        ----------
        checkpoint_path : str
            the path where the checkpoint will be saved
        """

        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )

    def load_from_checkpoint(
        self, checkpoint_path, only_model: bool = False, load_strict: bool = True
    ) -> None:
        """
        Load from a checkpoint

        Parameters
        ----------
        checkpoint_path : str
            the path to the checkpoint
        only_model : bool, default False
            if `True`, only the model state dictionary will be loaded (and not the epoch and the optimizer state
            dictionary)
        load_strict : bool, default True
            if `True`, any inconsistencies in state dictionaries are regarded as errors
        """

        if checkpoint_path is None:
            return
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=load_strict)
        if not only_model:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epoch = checkpoint["epoch"]

    def save_model(self, save_path: str) -> None:
        """
        Save the model state dictionary

        Parameters
        ----------
        save_path : str
            the path where the state will be saved
        """

        torch.save(self.model.state_dict(), save_path)
        print("saved the model successfully")

    def _apply_predict_functions(self, predicted):
        """
        Map from model output to prediction
        """

        predicted = self.primary_predict_function(predicted)
        predicted = self.predict_function(predicted)
        return predicted

    def _get_prediction(
        self,
        main_input: Dict,
        tags: torch.Tensor,
        ssl_inputs: List = None,
        ssl_targets: List = None,
        augment_n: int = 0,
        embedding: bool = False,
        subsample: List = None,
    ) -> Tuple:
        """
        Get the prediction of `self.model` for input averaged over `augment_n` augmentations
        """

        if augment_n == 0:
            augment_n = 1
            augment = False
        else:
            augment = True
        model_input, ssl_inputs, ssl_targets = self.transformer.transform(
            deepcopy(main_input),
            ssl_inputs=ssl_inputs,
            ssl_targets=ssl_targets,
            augment=augment,
            subsample=subsample,
        )
        if not embedding:
            predicted, ssl_predicted = self.model(model_input, ssl_inputs, tag=tags)
        else:
            predicted = self.model.extract_features(model_input)
            ssl_predicted = None
        if self.parallel and predicted is not None and len(predicted.shape) == 4:
            predicted = predicted.reshape(
                (-1, model_input.shape[0], *predicted.shape[2:])
            )
        if predicted is not None and augment_n > 1:
            self.model.ssl_off()
            for i in range(augment_n - 1):
                model_input, *_ = self.transformer.transform(
                    deepcopy(main_input), augment=augment
                )
                if not embedding:
                    pred, _ = self.model(model_input, None, tag=tags)
                else:
                    pred = self.model.extract_features(model_input)
                predicted += pred.detach()
            self.model.ssl_on()
            predicted /= augment_n
        if self.model.process_labels:
            class_embedding = self.model.transform_labels(predicted.device)
            beh_dict = self.behaviors_dict()
            class_embedding = torch.cat(
                [class_embedding[beh_dict[k]] for k in sorted(beh_dict.keys())], 0
            )
            predicted = {
                "video": predicted,
                "text": class_embedding,
                "logit_scale": self.model.logit_scale(),
                "device": predicted.device,
            }
        return predicted, ssl_predicted, ssl_targets

    def _ssl_loss(self, ssl_predicted, ssl_targets):
        """
        Apply SSL losses
        """

        ssl_loss = []
        for loss, predicted, target in zip(self.ssl_losses, ssl_predicted, ssl_targets):
            ssl_loss.append(loss(predicted, target))
        return ssl_loss

    def _loss_function(
        self,
        batch: Dict,
        augment_n: int,
        temporal_subsampling_size: int = None,
        skip_metrics: List = None,
    ) -> Tuple[float, float]:
        """
        Calculate the loss function and the metric for a dataloader batch

        Averaging the predictions over augment_n augmentations.
        """

        if "target" not in batch or torch.isnan(batch["target"]).all():
            raise ValueError("Cannot compute loss function with nan targets!")
        main_input = {k: v.to(self.device) for k, v in batch["input"].items()}
        main_target, ssl_targets, ssl_inputs = None, None, None
        main_target = batch["target"].to(self.device)
        if "ssl_targets" in batch:
            ssl_targets = [
                {k: v.to(self.device) for k, v in x.items()}
                if isinstance(x, dict)
                else None
                for x in batch["ssl_targets"]
            ]
        if "ssl_inputs" in batch:
            ssl_inputs = [
                {k: v.to(self.device) for k, v in x.items()}
                if isinstance(x, dict)
                else None
                for x in batch["ssl_inputs"]
            ]
        if temporal_subsampling_size is not None:
            subsample = sorted(
                random.sample(
                    range(main_target.shape[-1]),
                    int(temporal_subsampling_size * main_target.shape[-1]),
                )
            )
            main_target = main_target[..., subsample]
        else:
            subsample = None

        if self.ignore_tags:
            tag = None
        else:
            tag = batch.get("tag")
        predicted, ssl_predicted, ssl_targets = self._get_prediction(
            main_input, tag, ssl_inputs, ssl_targets, augment_n, subsample=subsample
        )
        del main_input, ssl_inputs
        return self._compute(
            ssl_predicted,
            ssl_targets,
            predicted,
            main_target,
            tag=batch.get("tag"),
            skip_metrics=skip_metrics,
        )

    def _compute(
        self,
        ssl_predicted: List,
        ssl_targets: List,
        predicted: torch.Tensor,
        main_target: torch.Tensor,
        tag: Any = None,
        skip_loss: bool = False,
        apply_primary_function: bool = True,
        skip_metrics: List = None,
    ) -> Tuple[float, float]:
        """
        Compute the losses and metrics from predictions
        """

        if skip_metrics is None:
            skip_metrics = []
        if not skip_loss:
            ssl_losses = self._ssl_loss(ssl_predicted, ssl_targets)
            if predicted is not None:
                loss = self.loss(predicted, main_target)
            else:
                loss = 0
        else:
            ssl_losses, loss = [], 0

        if predicted is not None:
            if isinstance(predicted, dict):
                predicted = {
                    k: v.detach()
                    for k, v in predicted.items()
                    if isinstance(v, torch.Tensor)
                }
            else:
                predicted = predicted.detach()
            if apply_primary_function:
                predicted = self.primary_predict_function(predicted)
            predicted_transformed = self.predict_function(predicted)
            for name, metric_function in self.metrics.items():
                if name not in skip_metrics:
                    if metric_function.needs_raw_data:
                        metric_function.update(predicted, main_target, tag)
                    else:
                        metric_function.update(
                            predicted_transformed,
                            main_target,
                            tag,
                        )
        return loss, ssl_losses

    def _calculate_metrics(self) -> Dict:
        """
        Calculate the final values of epoch metrics
        """

        epoch_metrics = {}
        for metric_name, metric in self.metrics.items():
            m = metric.calculate()
            if type(m) is dict:
                for k, v in m.items():
                    if type(v) is torch.Tensor:
                        v = v.item()
                    epoch_metrics[f"{metric_name}_{k}"] = v
            else:
                if type(m) is torch.Tensor:
                    m = m.item()
                epoch_metrics[metric_name] = m
            metric.reset()
        return epoch_metrics

    def _run_epoch(
        self,
        dataloader: DataLoader,
        mode: str,
        augment_n: int,
        verbose: bool = False,
        unlabeled: bool = None,
        alpha: float = 1,
        temporal_subsampling_size: int = None,
    ) -> Tuple:
        """
        Run one epoch on dataloader

        Averaging the predictions over augment_n augmentations.
        Use "train" mode for training and "val" mode for evaluation.
        """

        if mode == "train":
            self.model.train()
        elif mode == "val":
            self.model.eval()
            pass
        else:
            raise ValueError(
                f'Mode {mode} is not recognized, please choose either "train" for training or "val" for validation'
            )
        if self.ignore_tags:
            tags = [None]
        else:
            tags = dataloader.dataset.get_tags()
        epoch_loss = 0
        epoch_ssl_loss = defaultdict(lambda: 0)
        data_len = 0
        set_pars = dataloader.dataset.set_indexing_parameters
        skip_metrics = self.skip_metrics if mode == "train" else None
        for tag in tags:
            set_pars(unlabeled=unlabeled, tag=tag)
            data_len += len(dataloader)
            if verbose:
                dataloader = tqdm(dataloader)
            for batch in dataloader:
                loss, ssl_losses = self._loss_function(
                    batch,
                    augment_n,
                    temporal_subsampling_size=temporal_subsampling_size,
                    skip_metrics=skip_metrics,
                )
                if loss != 0:
                    loss = loss * alpha
                    epoch_loss += loss.item()
                for i, (ssl_loss, weight) in enumerate(
                    zip(ssl_losses, self.ssl_weights)
                ):
                    if ssl_loss != 0:
                        epoch_ssl_loss[i] += ssl_loss.item()
                        loss = loss + weight * ssl_loss
                if mode == "train":
                    self.optimizer.zero_grad()
                    if loss.requires_grad:
                        loss.backward()
                    self.optimizer.step()

        epoch_loss = epoch_loss / data_len
        epoch_ssl_loss = {k: v / data_len for k, v in epoch_ssl_loss.items()}
        epoch_metrics = self._calculate_metrics()

        return epoch_loss, epoch_ssl_loss, epoch_metrics

    def train(
        self,
        trial: Trial = None,
        optimized_metric: str = None,
        to_ram: bool = False,
        autostop_interval: int = 30,
        autostop_threshold: float = 0.001,
        autostop_metric: str = None,
        main_task_on: bool = True,
        ssl_on: bool = True,
        temporal_subsampling_size: int = None,
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

        if self.parallel and not isinstance(self.model, nn.DataParallel):
            self.model = MyDataParallel(self.model)
        self.model.to(self.device)
        assert autostop_metric in [None, "loss"] + list(self.metrics)
        autostop_interval //= self.validation_interval
        if trial is not None and optimized_metric is None:
            raise ValueError(
                "You need to provide the optimized metric name (optimized_metric parameter) "
                "for optuna pruning to work!"
            )
        if to_ram:
            print("transferring datasets to RAM...")
            self.train_dataloader.dataset.to_ram()
            if self.val_dataloader is not None and len(self.val_dataloader) > 0:
                self.val_dataloader.dataset.to_ram()
        loss_log = {"train": [], "val": []}
        metrics_log = {"train": defaultdict(lambda: []), "val": defaultdict(lambda: [])}
        if not main_task_on:
            self.model.main_task_off()
        if not ssl_on:
            self.model.ssl_off()
        while self.epoch < self.num_epochs:
            self.epoch += 1
            unlabeled = None
            alpha = 1
            if self.pseudolabel:
                if self.epoch >= self.T1:
                    unlabeled = (self.epoch - self.T1) % self.t != 0
                    if unlabeled:
                        alpha = self._alpha(self.epoch)
                else:
                    unlabeled = False
            epoch_loss, epoch_ssl_loss, epoch_metrics = self._run_epoch(
                dataloader=self.train_dataloader,
                mode="train",
                augment_n=self.augment_train,
                unlabeled=unlabeled,
                alpha=alpha,
                temporal_subsampling_size=temporal_subsampling_size,
                verbose=loading_bar,
            )
            loss_log["train"].append(epoch_loss)
            epoch_string = f"[epoch {self.epoch}]"
            if self.pseudolabel:
                if unlabeled:
                    epoch_string += " (unlabeled)"
                else:
                    epoch_string += " (labeled)"
            epoch_string += f": loss {epoch_loss:.4f}"

            if len(epoch_ssl_loss) != 0:
                for key, value in sorted(epoch_ssl_loss.items()):
                    metrics_log["train"][f"ssl_loss_{key}"].append(value)
                    epoch_string += f", ssl_loss_{key} {value:.4f}"

            for metric_name, metric_value in sorted(epoch_metrics.items()):
                if metric_name not in self.skip_metrics:
                    metrics_log["train"][metric_name].append(metric_value)
                    epoch_string += f", {metric_name} {metric_value:.3f}"

            if (
                self.val_dataloader is not None
                and self.epoch % self.validation_interval == 0
            ):
                with torch.no_grad():
                    epoch_string += "\n"
                    (
                        val_epoch_loss,
                        val_epoch_ssl_loss,
                        val_epoch_metrics,
                    ) = self._run_epoch(
                        dataloader=self.val_dataloader,
                        mode="val",
                        augment_n=self.augment_val,
                    )
                    loss_log["val"].append(val_epoch_loss)
                    epoch_string += f"validation: loss {val_epoch_loss:.4f}"

                    if len(val_epoch_ssl_loss) != 0:
                        for key, value in sorted(val_epoch_ssl_loss.items()):
                            metrics_log["val"][f"ssl_loss_{key}"].append(value)
                            epoch_string += f", ssl_loss_{key} {value:.4f}"

                    for metric_name, metric_value in sorted(val_epoch_metrics.items()):
                        metrics_log["val"][metric_name].append(metric_value)
                        epoch_string += f", {metric_name} {metric_value:.3f}"

                if trial is not None:
                    if optimized_metric not in metrics_log["val"]:
                        raise ValueError(
                            f"The {optimized_metric} metric set for optimization is not being logged!"
                        )
                    trial.report(metrics_log["val"][optimized_metric][-1], self.epoch)
                    if trial.should_prune():
                        raise TrialPruned()

            if self.verbose:
                print(epoch_string)

            if self.log_file is not None:
                with open(self.log_file, "a") as f:
                    f.write(epoch_string + "\n")

            save_condition = (
                (self.model_save_epochs != 0)
                and (self.epoch % self.model_save_epochs == 0)
            ) or (self.epoch == self.num_epochs)

            if self.epoch > 0 and save_condition and self.model_save_path is not None:
                epoch_s = str(self.epoch).zfill(len(str(self.num_epochs)))
                self.save_checkpoint(
                    os.path.join(self.model_save_path, f"epoch{epoch_s}.pt")
                )

            if self.pseudolabel and self.epoch >= self.T1 and not unlabeled:
                self._set_pseudolabels()

            if autostop_metric == "loss":
                if len(loss_log["val"]) > autostop_interval * 2:
                    if (
                        np.mean(loss_log["val"][-autostop_interval:])
                        < np.mean(
                            loss_log["val"][-2 * autostop_interval : -autostop_interval]
                        )
                        + autostop_threshold
                    ):
                        break
            elif autostop_metric in metrics_log["val"]:
                if len(metrics_log["val"][autostop_metric]) > autostop_interval * 2:
                    if (
                        np.mean(
                            metrics_log["val"][autostop_metric][-autostop_interval:]
                        )
                        < np.mean(
                            metrics_log["val"][autostop_metric][
                                -2 * autostop_interval : -autostop_interval
                            ]
                        )
                        + autostop_threshold
                    ):
                        break

        metrics_log = {k: dict(v) for k, v in metrics_log.items()}

        return loss_log, metrics_log

    def evaluate_prediction(
        self,
        prediction: Union[torch.Tensor, Dict],
        data: Union[DataLoader, BehaviorDataset, str] = None,
        batch_size: int = 32,
    ) -> Tuple:
        """
        Compute metrics for a prediction

        Parameters
        ----------
        prediction : torch.Tensor
            the prediction
        data : torch.utils.data.DataLoader | dlc2action.data.dataset.BehaviorDataset, optional
            the data the prediction was made for (if not provided, take the validation dataset)
        batch_size : int, default 32
            the batch size

        Returns
        -------
        loss : float
            the average value of the loss function
        metric : dict
            a dictionary of average values of metric functions
        """

        if type(data) is not DataLoader:
            dataset = self._get_dataset(data)
            data = DataLoader(dataset, shuffle=False, batch_size=batch_size)
        epoch_loss = 0
        if isinstance(prediction, dict):
            num_classes = len(self.behaviors_dict())
            length = dataset.len_segment()
            coords = dataset.annotation_store.get_original_coordinates()
            for batch in data:
                main_target = batch["target"]
                pr_coords = coords[batch["index"]]
                predicted = torch.zeros((len(pr_coords), num_classes, length))
                for i, c in enumerate(pr_coords):
                    video_id = dataset.input_store.get_video_id(c)
                    clip_id = dataset.input_store.get_clip_id(c)
                    start, end = dataset.input_store.get_clip_start_end(c)
                    predicted[i, :, : end - start] = prediction[video_id][clip_id][
                        :, start:end
                    ]
                self._compute(
                    [],
                    [],
                    predicted,
                    main_target,
                    skip_loss=True,
                    tag=batch.get("tag"),
                    apply_primary_function=False,
                )
        else:
            for batch in data:
                main_target = batch["target"]
                predicted = prediction[batch["index"]]
                self._compute(
                    [],
                    [],
                    predicted,
                    main_target,
                    skip_loss=True,
                    tag=batch.get("tag"),
                    apply_primary_function=False,
                )
        epoch_metrics = self._calculate_metrics()
        strings = [
            f"{metric_name} {metric_value:.3f}"
            for metric_name, metric_value in epoch_metrics.items()
        ]
        val_string = ", ".join(sorted(strings))
        print(val_string)
        return epoch_loss, epoch_metrics

    def evaluate(
        self,
        data: Union[DataLoader, BehaviorDataset, str] = None,
        augment_n: int = 0,
        batch_size: int = 32,
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
        batch_size : int, default 32
            the batch size
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

        if self.parallel and not isinstance(self.model, nn.DataParallel):
            self.model = MyDataParallel(self.model)
        self.model.to(self.device)
        if type(data) is not DataLoader:
            data = self._get_dataset(data)
            data = DataLoader(data, shuffle=False, batch_size=batch_size)
        with torch.no_grad():
            epoch_loss, epoch_ssl_loss, epoch_metrics = self._run_epoch(
                dataloader=data, mode="val", augment_n=augment_n, verbose=verbose
            )
        val_string = f"loss {epoch_loss:.4f}"
        for metric_name, metric_value in sorted(epoch_metrics.items()):
            val_string += f", {metric_name} {metric_value:.3f}"
        print(val_string)
        return epoch_loss, epoch_ssl_loss, epoch_metrics

    def predict(
        self,
        data: Union[DataLoader, BehaviorDataset, str] = None,
        raw_output: bool = False,
        apply_primary_function: bool = True,
        augment_n: int = 0,
        batch_size: int = 32,
        train_mode: bool = False,
        to_ram: bool = False,
        embedding: bool = False,
    ) -> torch.Tensor:
        """
        Make a prediction with the Task model

        Parameters
        ----------
        data : torch.utils.data.DataLoader | dlc2action.data.dataset.BehaviorDataset | str, optional
            the data to evaluate on (if not provided, evaluate on the Task validation dataset)
        raw_output : bool, default False
            if `True`, the raw predicted probabilities are returned
        apply_primary_function : bool, default True
            if `True`, the primary predict function is applied (to map the model output into a shape corresponding to
            the input)
        augment_n : int, default 0
            the number of augmentations to average results over
        batch_size : int, default 32
            the batch size
        train_mode : bool, default False
            if `True`, the model is used in training mode (affects dropout and batch normalization layers)
        to_ram : bool, default False
            if `True`, the dataset will be loaded in RAM (this speeds up the calculations but can lead to crashes
            if the dataset is too large)
        embedding : bool, default False
            if `True`, the output of feature extractor is returned, ignoring the prediction module of the model

        Returns
        -------
        prediction : torch.Tensor
            a prediction for the input data
        """

        if self.parallel and not isinstance(self.model, nn.DataParallel):
            self.model = MyDataParallel(self.model)
        self.model.to(self.device)
        if train_mode:
            self.model.train()
        else:
            self.model.eval()
        output = []
        if embedding:
            raw_output = True
            apply_primary_function = True
        if type(data) is not DataLoader:
            data = self._get_dataset(data)
            if to_ram:
                print("transferring dataset to RAM...")
                data.to_ram()
            data = DataLoader(data, shuffle=False, batch_size=batch_size)
        self.model.ssl_off()
        with torch.no_grad():
            for batch in tqdm(data):
                input = {k: v.to(self.device) for k, v in batch["input"].items()}
                predicted, _, _ = self._get_prediction(
                    input,
                    batch.get("tag"),
                    augment_n=augment_n,
                    embedding=embedding,
                )
                if apply_primary_function:
                    predicted = self.primary_predict_function(predicted)
                if not raw_output:
                    predicted = self.predict_function(predicted)
                output.append(predicted.detach().cpu())
        self.model.ssl_on()
        output = torch.cat(output).detach()
        return output

    def dataset(self, mode="train") -> BehaviorDataset:
        """
        Get a dataset

        Parameters
        ----------
        mode : {'train', 'val', 'test}
            the dataset to get

        Returns
        -------
        dataset : dlc2action.data.dataset.BehaviorDataset
            the dataset
        """

        dataloader = self.dataloader(mode)
        if dataloader is None:
            raise ValueError("The length of the dataloader is 0!")
        return dataloader.dataset

    def dataloader(self, mode: str = "train") -> DataLoader:
        """
        Get a dataloader

        Parameters
        ----------
        mode : {'train', 'val', 'test}
            the dataset to get

        Returns
        -------
        dataloader : torch.utils.data.DataLoader
            the dataloader
        """

        if mode == "train":
            return self.train_dataloader
        elif mode == "val":
            return self.val_dataloader
        elif mode == "test":
            return self.test_dataloader
        else:
            raise ValueError(
                f'The {mode} mode is not recognized, please choose from "train", "val" or "test"'
            )

    def _get_dataset(self, dataset):
        """
        Get a dataset from a dataloader, a string ('train', 'test' or 'val') or `None` (default)
        """

        if dataset is None:
            dataset = self.dataset("val")
        elif dataset in ["train", "val", "test"]:
            dataset = self.dataset(dataset)
        elif type(dataset) is DataLoader:
            dataset = dataset.dataset
        if type(dataset) is BehaviorDataset:
            return dataset
        else:
            raise TypeError(f"The {type(dataset)} type of dataset is not recognized!")

    def _get_dataloader(self, dataset):
        """
        Get a dataloader from a dataset, a string ('train', 'test' or 'val') or `None` (default)
        """

        if dataset is None:
            dataset = self.dataloader("val")
        elif dataset in ["train", "val", "test"]:
            dataset = self.dataloader(dataset)
            if dataset is None:
                raise ValueError(f"The length of the dataloader is 0!")
        elif type(dataset) is BehaviorDataset:
            dataset = DataLoader(dataset)
        if type(dataset) is DataLoader:
            return dataset
        else:
            raise TypeError(f"The {type(dataset)} type of dataset is not recognized!")

    def generate_full_length_prediction(
        self, dataset=None, batch_size=32, augment_n=10
    ):
        """
        Compile a prediction for the original input sequences

        Parameters
        ----------
        dataset : BehaviorDataset, optional
            the dataset to generate a prediction for (if `None`, generate for the validation dataset)
        batch_size : int, default 32
            the batch size
        augment_n : int, default 10
            the number of augmentations to average results over

        Returns
        -------
        prediction : dict
            a nested dictionary where first level keys are video ids, second level keys are clip ids and values
            are prediction tensors
        """

        dataset = self._get_dataset(dataset)
        if not isinstance(dataset, BehaviorDataset):
            raise TypeError(
                f"The dataset parameter has to be either None, string, "
                f"BehaviorDataset or Dataloader, got {type(dataset)}"
            )
        predicted = self.predict(
            dataset,
            raw_output=True,
            apply_primary_function=True,
            augment_n=augment_n,
            batch_size=batch_size,
        )
        predicted = dataset.generate_full_length_prediction(predicted)
        predicted = {
            v_id: {
                clip_id: self._apply_predict_functions(v.unsqueeze(0)).squeeze()
                for clip_id, v in video_dict.items()
            }
            for v_id, video_dict in predicted.items()
        }
        return predicted

    def generate_submission(
        self, frame_number_map_file, dataset=None, batch_size=32, augment_n=10
    ):
        """
        Generate a MABe-22 style submission dictionary

        Parameters
        ----------
        dataset : BehaviorDataset, optional
            the dataset to generate a prediction for (if `None`, generate for the validation dataset)
        batch_size : int, default 32
            the batch size
        augment_n : int, default 10
            the number of augmentations to average results over

        Returns
        -------
        submission : dict
            a dictionary with frame number mapping and embeddings
        """

        dataset = self._get_dataset(dataset)
        if not isinstance(dataset, BehaviorDataset):
            raise TypeError(
                f"The dataset parameter has to be either None, string, "
                f"BehaviorDataset or Dataloader, got {type(dataset)}"
            )
        predicted = self.predict(
            dataset,
            raw_output=True,
            apply_primary_function=True,
            augment_n=augment_n,
            batch_size=batch_size,
            embedding=True,
        )
        predicted = dataset.generate_full_length_prediction(predicted)
        frame_map = np.load(frame_number_map_file, allow_pickle=True).item()
        length = frame_map[list(frame_map.keys())[-1]][1]
        embeddings = None
        for video_id in list(predicted.keys()):
            split = video_id.split("--")
            if len(split) != 2 or len(predicted[video_id]) > 1:
                raise RuntimeError(
                    "Generating submissions is only implemented for the mabe22 dataset!"
                )
            if split[1] not in frame_map:
                raise RuntimeError(f"The {split[1]} video is not in the frame map file")
            v_id = split[1]
            clip_id = list(predicted[video_id].keys())[0]
            if embeddings is None:
                embeddings = np.zeros((length, predicted[video_id][clip_id].shape[0]))
            start, end = frame_map[v_id]
            embeddings[start:end, :] = predicted[video_id][clip_id].T
            predicted.pop(video_id)
        predicted = {
            "frame_number_map": frame_map,
            "embeddings": embeddings.astype(np.float32),
        }
        return predicted

    def _get_intervals(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Get a list of True group beginning and end indices from a boolean tensor
        """

        output, indices = torch.unique_consecutive(tensor, return_inverse=True)
        true_indices = torch.where(output)[0]
        starts = torch.tensor(
            [(indices == i).nonzero(as_tuple=True)[0][0] for i in true_indices]
        )
        ends = torch.tensor(
            [(indices == i).nonzero(as_tuple=True)[0][-1] + 1 for i in true_indices]
        )
        return torch.stack([starts, ends]).T

    def _smooth(self, tensor: torch.Tensor, smooth_interval: int = 1) -> torch.Tensor:
        """
        Get rid of jittering in a non-exclusive classification tensor

        First, remove intervals of 0 shorter than `smooth_interval`. Then, remove intervals of 1 shorter than
        `smooth_interval`.
        """

        if len(tensor.shape) > 1:
            for c in tensor.shape[1]:
                intervals = self._get_intervals(tensor[:, c] == 0)
                interval_lengths = torch.tensor(
                    [interval[1] - interval[0] for interval in intervals]
                )
                short_intervals = intervals[interval_lengths <= smooth_interval]
                for start, end in short_intervals:
                    tensor[start:end, c] = 1
                intervals = self._get_intervals(tensor[:, c] == 1)
                interval_lengths = torch.tensor(
                    [interval[1] - interval[0] for interval in intervals]
                )
                short_intervals = intervals[interval_lengths <= smooth_interval]
                for start, end in short_intervals:
                    tensor[start:end, c] = 0
        else:
            for c in tensor.unique():
                intervals = self._get_intervals(tensor == c)
                interval_lengths = torch.tensor(
                    [interval[1] - interval[0] for interval in intervals]
                )
                short_intervals = intervals[interval_lengths <= smooth_interval]
                for start, end in short_intervals:
                    if start == 0:
                        tensor[start:end] = tensor[end + 1]
                    else:
                        tensor[start:end] = tensor[start - 1]
        return tensor

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
        """
        Visualize random predictions

        Parameters
        ----------
        save_path : str, optional
            the path where the plot will be saved
        add_legend : bool, default True
            if `True`, legend will be added to the plot
        ground_truth : bool, default True
            if `True`, ground truth will be added to the plot
        colormap : str, default 'viridis'
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
        dataset : BehaviorDataset, optional
            the dataset to make the prediction for (if not provided, the validation dataset is used)
        drop_classes : set, optional
            a set of class names to not be displayed
        """

        if title is None:
            title = ""
        dataset = self._get_dataset(dataset)
        inverse_dict = {v: k for k, v in dataset.behaviors_dict().items()}
        label_ind = inverse_dict[label]
        labels = {1: label, -100: "unknown"}
        label_keys = [1, -100]
        color_list = ["blue", "gray"]
        if whole_video:
            predicted = self.generate_full_length_prediction(dataset)
            keys = list(predicted.keys())
        counter = 0
        if whole_video:
            max_iter = len(keys) * 5
        else:
            max_iter = len(dataset) * 5
        ok = False
        while not ok:
            counter += 1
            if counter > max_iter:
                raise RuntimeError(
                    "Plotting is taking too many iterations; you should probably make some of the parameters less restrictive"
                )
            if whole_video:
                i = randint(0, len(keys) - 1)
                prediction = predicted[keys[i]]
                keys_i = list(prediction.keys())
                j = randint(0, len(keys_i) - 1)
                full_p = prediction[keys_i[j]]
                prediction = prediction[keys_i[j]][label_ind]
            else:
                dataloader = DataLoader(dataset)
                i = randint(0, len(dataloader) - 1)
                for num, batch in enumerate(dataloader):
                    if num == i:
                        break
                input_data = {k: v.to(self.device) for k, v in batch["input"].items()}
                prediction, *_ = self._get_prediction(
                    input_data, batch.get("tag"), augment_n=5
                )
                prediction = self._apply_predict_functions(prediction)
                j = randint(0, len(prediction) - 1)
                full_p = prediction[j]
                prediction = prediction[j][label_ind]
            classes = [x for x in torch.unique(prediction) if int(x) in label_keys]
            ok = 1 in classes
        fig, ax = plt.subplots(figsize=(width, 2))
        for c in classes:
            c_i = label_keys.index(int(c))
            output, indices, counts = torch.unique_consecutive(
                prediction == c, return_inverse=True, return_counts=True
            )
            long_indices = torch.where(output)[0]
            res_indices_start = [
                (indices == i).nonzero(as_tuple=True)[0][0].item() for i in long_indices
            ]
            res_indices_end = [
                (indices == i).nonzero(as_tuple=True)[0][-1].item() + 1
                for i in long_indices
            ]
            res_indices_len = [
                end - start for start, end in zip(res_indices_start, res_indices_end)
            ]
            ax.broken_barh(
                list(zip(res_indices_start, res_indices_len)),
                (0, 1),
                label=labels[int(c)],
                facecolors=color_list[c_i],
            )
        if ground_truth:
            gt = batch["target"][j][label_ind].to(self.device)
            classes_gt = [x for x in torch.unique(gt) if int(x) in label_keys]
            for c in classes_gt:
                c_i = label_keys.index(int(c))
                if c in classes:
                    label = None
                else:
                    label = labels[int(c)]
                output, indices, counts = torch.unique_consecutive(
                    gt == c, return_inverse=True, return_counts=True
                )
                long_indices = torch.where(output * (counts > 5))[0]
                res_indices_start = [
                    (indices == i).nonzero(as_tuple=True)[0][0].item()
                    for i in long_indices
                ]
                res_indices_end = [
                    (indices == i).nonzero(as_tuple=True)[0][-1].item() + 1
                    for i in long_indices
                ]
                res_indices_len = [
                    end - start
                    for start, end in zip(res_indices_start, res_indices_end)
                ]
                ax.broken_barh(
                    list(zip(res_indices_start, res_indices_len)),
                    (1.5, 1),
                    facecolors=color_list[c_i],
                    label=label,
                )
        self._compute(
            main_target=batch["target"][j].unsqueeze(0).to(self.device),
            predicted=full_p.unsqueeze(0).to(self.device),
            ssl_targets=[],
            ssl_predicted=[],
            skip_loss=True,
        )
        metrics = self._calculate_metrics()
        if smooth_interval > 0:
            smoothed = self._smooth(full_p, smooth_interval=smooth_interval)[
                label_ind, :
            ]
            for c in classes:
                c_i = label_keys.index(int(c))
                output, indices, counts = torch.unique_consecutive(
                    smoothed == c, return_inverse=True, return_counts=True
                )
                long_indices = torch.where(output)[0]
                res_indices_start = [
                    (indices == i).nonzero(as_tuple=True)[0][0].item()
                    for i in long_indices
                ]
                res_indices_end = [
                    (indices == i).nonzero(as_tuple=True)[0][-1].item() + 1
                    for i in long_indices
                ]
                res_indices_len = [
                    end - start
                    for start, end in zip(res_indices_start, res_indices_end)
                ]
                ax.broken_barh(
                    list(zip(res_indices_start, res_indices_len)),
                    (3, 1),
                    label=labels[int(c)],
                    facecolors=color_list[c_i],
                )
        keys = list(metrics.keys())
        for key in keys:
            if key.split("_")[-1] != (str(label_ind)):
                metrics.pop(key)
        title = [title]
        for key, value in metrics.items():
            title.append(f"{'_'.join(key.split('_')[: -1])}: {value:.2f}")
        title = ", ".join(title)
        if not ground_truth:
            ax.axes.yaxis.set_visible(False)
        else:
            ax.set_yticks([0.5, 2])
            ax.set_yticklabels(["prediction", "ground truth"])
        if add_legend:
            ax.legend()
        if hide_axes:
            plt.axis("off")
        plt.title(title)
        plt.xlim((0, len(prediction)))
        if save_path is not None:
            plt.savefig(save_path, transparent=transparent)
        plt.show()

    def visualize_results(
        self,
        save_path: str = None,
        add_legend: bool = True,
        ground_truth: bool = True,
        colormap: str = "viridis",
        hide_axes: bool = False,
        min_classes: int = 1,
        width: int = 10,
        whole_video: bool = False,
        transparent: bool = False,
        dataset: Union[BehaviorDataset, DataLoader, str, None] = None,
        drop_classes: Set = None,
        search_classes: Set = None,
        num_samples: int = 1,
        smooth_interval_prediction: int = None,
        behavior_name: str = None,
    ):
        """
        Visualize random predictions

        Parameters
        ----------
        save_path : str, optional
            the path where the plot will be saved
        add_legend : bool, default True
            if `True`, legend will be added to the plot
        ground_truth : bool, default True
            if `True`, ground truth will be added to the plot
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

        if drop_classes is None:
            drop_classes = []
        dataset = self._get_dataset(dataset)
        if dataset.annotation_class() == "exclusive_classification":
            exclusive = True
        elif dataset.annotation_class() == "nonexclusive_classification":
            exclusive = False
        else:
            raise NotImplementedError(
                f"Results visualisation is not implemented for {dataset.annotation_class() } datasets!"
            )
        if exclusive:
            labels = {
                k: v
                for k, v in dataset.behaviors_dict().items()
                if v not in drop_classes
            }
            labels.update({-100: "unknown"})
        else:
            inverse_dict = {v: k for k, v in dataset.behaviors_dict().items()}
            if behavior_name is None:
                behavior_name = list(inverse_dict.keys())[0]
            label_ind = inverse_dict[behavior_name]
            labels = {1: label, -100: "unknown"}
        label_keys = sorted([int(x) for x in labels.keys()])
        if search_classes is None:
            ok = True
        else:
            ok = False
        classes = []
        if whole_video:
            predicted = self.generate_full_length_prediction(dataset)
            keys = list(predicted.keys())
        counter = 0
        if whole_video:
            max_iter = len(keys) * 2
        else:
            max_iter = len(dataset) * 2
        while len(classes) < min_classes or not ok:
            counter += 1
            if counter > max_iter:
                raise RuntimeError(
                    "Plotting is taking too many iterations; you should probably make some of the parameters less restrictive"
                )
            if whole_video:
                i = randint(0, len(keys) - 1)
                prediction = predicted[keys[i]]
                keys_i = list(prediction.keys())
                j = randint(0, len(keys_i) - 1)
                prediction = prediction[keys_i[j]]
                key1 = keys[i]
                key2 = keys_i[j]
            else:
                dataloader = DataLoader(dataset)
                i = randint(0, len(dataloader) - 1)
                for num, batch in enumerate(dataloader):
                    if num == i:
                        break
                input = {k: v.to(self.device) for k, v in batch["input"].items()}
                prediction, *_ = self._get_prediction(
                    input, batch.get("tag"), augment_n=5
                )
                prediction = self._apply_predict_functions(prediction)
                j = randint(0, len(prediction) - 1)
                prediction = prediction[j]
            if not exclusive:
                prediction = prediction[label_ind]
            if smooth_interval_prediction > 0:
                unsmoothed_prediction = deepcopy(prediction)
                prediction = self._smooth(prediction, smooth_interval_prediction)
                height = 3
            else:
                height = 2
            classes = [
                labels[int(x)] for x in torch.unique(prediction) if x in label_keys
            ]
            if search_classes is not None:
                ok = any([x in classes for x in search_classes])
        fig, ax = plt.subplots(figsize=(width, height))
        color_list = cm.get_cmap(colormap, lut=len(labels)).colors

        def _plot_prediction(prediction, height, set_labels=True):
            for c in label_keys:
                c_i = label_keys.index(int(c))
                output, indices, counts = torch.unique_consecutive(
                    prediction == c, return_inverse=True, return_counts=True
                )
                long_indices = torch.where(output)[0]
                if len(long_indices) == 0:
                    continue
                res_indices_start = [
                    (indices == i).nonzero(as_tuple=True)[0][0].item()
                    for i in long_indices
                ]
                res_indices_end = [
                    (indices == i).nonzero(as_tuple=True)[0][-1].item() + 1
                    for i in long_indices
                ]
                res_indices_len = [
                    end - start
                    for start, end in zip(res_indices_start, res_indices_end)
                ]
                if set_labels:
                    label = labels[int(c)]
                else:
                    label = None
                ax.broken_barh(
                    list(zip(res_indices_start, res_indices_len)),
                    (height, 1),
                    label=label,
                    facecolors=color_list[c_i],
                )

        if smooth_interval_prediction > 0:
            _plot_prediction(unsmoothed_prediction, 0)
            _plot_prediction(prediction, 1.5, set_labels=False)
            gt_height = 3
        else:
            _plot_prediction(prediction, 0)
            gt_height = 1.5
        if ground_truth:
            if not whole_video:
                gt = batch["target"][j].to(self.device)
            else:
                gt = dataset.generate_full_length_gt()[key1][key2]
            for c in label_keys:
                c_i = label_keys.index(int(c))
                if labels[int(c)] in classes:
                    label = None
                else:
                    label = labels[int(c)]
                output, indices, counts = torch.unique_consecutive(
                    gt == c, return_inverse=True, return_counts=True
                )
                long_indices = torch.where(output)[0]
                if len(long_indices) == 0:
                    continue
                res_indices_start = [
                    (indices == i).nonzero(as_tuple=True)[0][0].item()
                    for i in long_indices
                ]
                res_indices_end = [
                    (indices == i).nonzero(as_tuple=True)[0][-1].item() + 1
                    for i in long_indices
                ]
                res_indices_len = [
                    end - start
                    for start, end in zip(res_indices_start, res_indices_end)
                ]
                ax.broken_barh(
                    list(zip(res_indices_start, res_indices_len)),
                    (gt_height, 1),
                    facecolors=color_list[c_i] if c != "unknown" else "gray",
                    label=label,
                )
        if not ground_truth:
            ax.axes.yaxis.set_visible(False)
        else:
            if smooth_interval_prediction > 0:
                ax.set_yticks([0.5, 2, 3.5])
                ax.set_yticklabels(["prediction", "smoothed", "ground truth"])
            else:
                ax.set_yticks([0.5, 2])
                ax.set_yticklabels(["prediction", "ground truth"])
        if add_legend:
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        if hide_axes:
            plt.axis("off")
        if save_path is not None:
            plt.savefig(save_path, transparent=transparent)
        plt.show()

    def set_ssl_transformations(self, ssl_transformations):
        self.train_dataloader.dataset.set_ssl_transformations(ssl_transformations)
        if self.val_dataloader is not None:
            self.val_dataloader.dataset.set_ssl_transformations(ssl_transformations)

    def set_ssl_losses(self, ssl_losses: list) -> None:
        """
        Set SSL losses

        Parameters
        ----------
        ssl_losses : list
            a list of callable SSL losses
        """

        self.ssl_losses = ssl_losses

    def set_log(self, log: str) -> None:
        """
        Set the log file

        Parameters
        ----------
        log: str
            the mew log file path
        """

        self.log_file = log

    def set_keep_target_none(self, keep_target_none: List) -> None:
        """
        Set the keep_target_none parameter of the transformer

        Parameters
        ----------
        keep_target_none : list
            a list of bool values
        """

        self.transformer.keep_target_none = keep_target_none

    def set_generate_ssl_input(self, generate_ssl_input: list) -> None:
        """
        Set the generate_ssl_input parameter of the transformer

        Parameters
        ----------
        generate_ssl_input : list
            a list of bool values
        """

        self.transformer.generate_ssl_input = generate_ssl_input

    def set_model(self, model: Model) -> None:
        """
        Set a new model

        Parameters
        ----------
        model: Model
            the new model
        """

        self.epoch = 0
        self.model = model
        self.optimizer = self.optimizer_class(model.parameters(), lr=self.lr)
        if self.model.process_labels:
            self.model.set_behaviors(list(self.behaviors_dict().values()))

    def set_dataloaders(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader = None,
        test_dataloader: DataLoader = None,
    ) -> None:
        """
        Set new dataloaders

        Parameters
        ----------
        train_dataloader: torch.utils.data.DataLoader
            the new train dataloader
        val_dataloader : torch.utils.data.DataLoader
            the new validation dataloader
        test_dataloader : torch.utils.data.DataLoader
            the new test dataloader
        """

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def set_loss(self, loss: Callable) -> None:
        """
        Set new loss function

        Parameters
        ----------
        loss: callable
            the new loss function
        """

        self.loss = loss

    def set_metrics(self, metrics: dict) -> None:
        """
        Set new metric

        Parameters
        ----------
        metrics : dict
            the new metric dictionary
        """

        self.metrics = metrics

    def set_transformer(self, transformer: Transformer) -> None:
        """
        Set a new transformer

        Parameters
        ----------
        transformer: Transformer
            the new transformer
        """

        self.transformer = transformer

    def set_predict_functions(
        self, primary_predict_function: Callable, predict_function: Callable
    ) -> None:
        """
        Set new predict functions

        Parameters
        ----------
        primary_predict_function : callable
            the new primary predict function
        predict_function : callable
            the new predict function
        """

        self.primary_predict_function = primary_predict_function
        self.predict_function = predict_function

    def _set_pseudolabels(self):
        """
        Set pseudolabels
        """

        self.train_dataloader.dataset.set_unlabeled(True)
        predicted = self.predict(
            data=self.dataset("train"),
            raw_output=False,
            augment_n=self.augment_val,
            ssl_off=True,
        )
        self.train_dataloader.dataset.set_annotation(predicted.detach())

    def _alpha(self, epoch):
        """
        Get the current pseudolabeling alpha parameter
        """

        if epoch <= self.T1:
            return 0
        elif epoch < self.T2:
            return self.alpha_f * (epoch - self.T1) / (self.T2 - self.T1)
        else:
            return self.alpha_f

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

        class_counts = {}
        for x in ["train", "val", "test"]:
            try:
                class_counts[x] = self.dataset(x).count_classes(bouts)
            except ValueError:
                class_counts[x] = {k: 0 for k in self.behaviors_dict().keys()}
        return class_counts

    def behaviors_dict(self) -> Dict:
        """
        Get a behavior dictionary

        Keys are label indices and values are label names.

        Returns
        -------
        behaviors_dict : dict
            behavior dictionary
        """

        return self.dataset().behaviors_dict()

    def update_parameters(self, parameters: Dict) -> None:
        """
        Update training parameters from a dictionary

        Parameters
        ----------
        parameters : dict
            the update dictionary
        """

        self.lr = parameters.get("lr", self.lr)
        self.parallel = parameters.get("parallel", self.parallel)
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.lr)
        self.verbose = parameters.get("verbose", self.verbose)
        self.device = parameters.get("device", self.device)
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.augment_train = int(parameters.get("augment_train", self.augment_train))
        self.augment_val = int(parameters.get("augment_val", self.augment_val))
        ssl_weights = parameters.get("ssl_weights", self.ssl_weights)
        if ssl_weights is None:
            ssl_weights = [1 for _ in self.ssl_losses]
        if not isinstance(ssl_weights, Iterable):
            ssl_weights = [ssl_weights for _ in self.ssl_losses]
        self.ssl_weights = ssl_weights
        self.num_epochs = parameters.get("num_epochs", self.num_epochs)
        self.model_save_epochs = parameters.get(
            "model_save_epochs", self.model_save_epochs
        )
        self.model_save_path = parameters.get("model_save_path", self.model_save_path)
        self.pseudolabel = parameters.get("pseudolabel", self.pseudolabel)
        self.T1 = int(parameters.get("pseudolabel_start", self.T1))
        self.T2 = int(parameters.get("alpha_growth_stop", self.T2))
        self.t = int(parameters.get("correction_interval", self.t))
        self.alpha_f = parameters.get("pseudolabel_alpha_f", self.alpha_f)
        self.log_file = parameters.get("log_file", self.log_file)

    def generate_uncertainty_score(
        self,
        classes: List,
        augment_n: int = 0,
        batch_size: int = 32,
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
        batch_size : int, default 32
            the batch size
        method : {"least_confidence", "entropy"}
            the method used to calculate the scores from the probability predictions (`"least_confidence"`: `1 - p_i` if
            `p_i > 0.5` or `p_i` if `p_i < 0.5`; `"entropy"`: `- p_i * log(p_i) - (1 - p_i) * log(1 - p_i)`)

        Returns
        -------
        score_dicts : dict
            a nested dictionary where first level keys are video ids, second level keys are clip ids and values
            are score tensors
        """

        dataset = self.dataset("train")
        if behaviors_dict is None:
            behaviors_dict = self.behaviors_dict()
        if not isinstance(dataset, BehaviorDataset):
            raise TypeError(
                f"The dataset parameter has to be either None, string, "
                f"BehaviorDataset or Dataloader, got {type(dataset)}"
            )
        if predicted is None:
            predicted = self.predict(
                dataset,
                raw_output=True,
                apply_primary_function=True,
                augment_n=augment_n,
                batch_size=batch_size,
            )
        predicted = dataset.generate_full_length_prediction(predicted)
        if isinstance(classes[0], str):
            behaviors_dict_inverse = {v: k for k, v in behaviors_dict.items()}
            classes = [behaviors_dict_inverse[c] for c in classes]
        for v_id, v in predicted.items():
            for clip_id, vv in v.items():
                if method == "least_confidence":
                    predicted[v_id][clip_id][vv > 0.5] = 1 - vv[vv > 0.5]
                elif method == "entropy":
                    predicted[v_id][clip_id][vv != -100] = (
                        -vv * torch.log(vv) - (1 - vv) * torch.log(1 - vv)
                    )[vv != -100]
                elif method == "random":
                    predicted[v_id][clip_id] = torch.rand(vv.shape)
                else:
                    raise ValueError(
                        f"The {method} method is not recognized; please choose from ['least_confidence', 'entropy']"
                    )
                predicted[v_id][clip_id][vv == -100] = 0

        predicted = {
            v_id: {clip_id: v[classes, :] for clip_id, v in video_dict.items()}
            for v_id, video_dict in predicted.items()
        }
        return predicted

    def generate_bald_score(
        self,
        classes: List,
        augment_n: int = 0,
        batch_size: int = 32,
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
        batch_size : int, default 32
            the batch size
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

        dataset = self.dataset("train")
        dataset = self._get_dataset(dataset)
        if not isinstance(dataset, BehaviorDataset):
            raise TypeError(
                f"The dataset parameter has to be either None, string, "
                f"BehaviorDataset or Dataloader, got {type(dataset)}"
            )
        predictions = []
        for _ in range(num_models):
            predicted = self.predict(
                dataset,
                raw_output=True,
                apply_primary_function=True,
                augment_n=augment_n,
                batch_size=batch_size,
                train_mode=True,
            )
            predicted = dataset.generate_full_length_prediction(predicted)
            if isinstance(classes[0], str):
                behaviors_dict_inverse = {
                    v: k for k, v in self.behaviors_dict().items()
                }
                classes = [behaviors_dict_inverse[c] for c in classes]
            for v_id, v in predicted.items():
                for clip_id, vv in v.items():
                    vv[vv != -100] = (vv[vv != -100] > 0.5).int().float()
                    predicted[v_id][clip_id] = vv
            predicted = {
                v_id: {clip_id: v[classes, :] for clip_id, v in video_dict.items()}
                for v_id, video_dict in predicted.items()
            }
            predictions.append(predicted)
        result = {v_id: {} for v_id in predictions[0]}
        r = range(-int(kernel_size / 2), int(kernel_size / 2) + 1)
        gauss = [1 / (1 * sqrt(2 * pi)) * exp(-float(x) ** 2 / (2 * 1**2)) for x in r]
        gauss = [x / sum(gauss) for x in gauss]
        kernel = torch.FloatTensor([[gauss]])
        for v_id in predictions[0]:
            for clip_id in predictions[0][v_id]:
                consensus = (
                    (
                        torch.mean(
                            torch.stack([x[v_id][clip_id] for x in predictions]), dim=0
                        )
                        > 0.5
                    )
                    .int()
                    .float()
                )
                consensus[predictions[0][v_id][clip_id] == -100] = -100
                result[v_id][clip_id] = torch.zeros(consensus.shape)
                for x in predictions:
                    result[v_id][clip_id] += (x[v_id][clip_id] != consensus).int()
                result[v_id][clip_id] = result[v_id][clip_id] * 2 / num_models
                res = torch.zeros(result[v_id][clip_id].shape)
                for i in range(len(classes)):
                    res[
                        i, floor(kernel_size // 2) : -floor(kernel_size // 2)
                    ] = torch.nn.functional.conv1d(
                        result[v_id][clip_id][i, :].unsqueeze(0).unsqueeze(0), kernel
                    )[
                        0, ...
                    ]
                result[v_id][clip_id] = res
        return result

    def get_normalization_stats(self) -> Optional[Dict]:
        return self.train_dataloader.dataset.stats
