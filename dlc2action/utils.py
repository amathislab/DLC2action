#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
"""
## Utility functions

- `TensorDict` is a convenient data structure for keeping (and indexing) lists of feature dictionaries,
- `apply_threshold`, `apply_threshold_hysteresis` and `apply_threshold_max` are utility functions for
`dlc2action.data.dataset.BehaviorDataset.find_valleys`,
- `strip_suffix` is used to get rid of suffices if a string (usually filename) ends with one of them,
- `strip_prefix` is used to get rid of prefixes if a string (usually filename) starts with one of them,
- `rotation_matrix_2d` and `rotation_matrix_3d` are used to generate rotation matrices by
`dlc2action.transformer.base_transformer.Transformer` instances
"""

import torch
from typing import List, Dict, Union
from collections.abc import Iterable
import warnings
import os
import numpy as np
from torch import nn
from torch.nn import functional as F
import math


class TensorDict:
    """
    A class that handles indexing in a dictionary of tensors of the same length
    """

    def __init__(self, obj: Union[Dict, Iterable] = None) -> None:
        """
        Parameters
        ----------
        obj : dict | iterable, optional
            either a dictionary of torch.Tensor instances of the same length or an iterable of dictionaries with
            the same keys (if not passed, a blank TensorDict is initialized)
        """

        if obj is None:
            obj = {}
        if isinstance(obj, list):
            self.keys = list(obj[0].keys())
            self.dict = {key: [] for key in self.keys}
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                for element in obj:
                    for key in self.keys:
                        self.dict[key].append(torch.tensor(element[key]))
            # self.dict = {k: torch.stack(v) for k, v in self.dict.items()}
            old_dict = self.dict
            self.dict = {}
            for k, v in old_dict.items():
                self.dict[k] = torch.stack(v)
        elif isinstance(obj, dict):
            if not all([isinstance(obj[key], torch.Tensor) for key in obj]):
                raise TypeError(
                    f"The values in the dictionary passed to TensorDict need to be torch.Tensor instances;"
                    f"got {[type(obj[key]) for key in obj]}"
                )
            lengths = [len(obj[key]) for key in obj]
            if not all([x == lengths[0] for x in lengths]):
                raise ValueError(
                    f"The value tensors in the dictionary passed to TensorDict need to have the same length;"
                    f"got {lengths}"
                )
            self.dict = obj
            self.keys = list(self.dict.keys())
        else:
            raise TypeError(
                f"TensorDict can only be constructed from an iterable of dictionaries of from a dictionary "
                f"of tensors (got {type(obj)})"
            )
        if len(self.keys) > 0:
            self.type = type(self.dict[self.keys[0]])
        else:
            self.type = None

    def __len__(self) -> int:
        return len(self.dict[self.keys[0]])

    def __getitem__(self, ind: Union[int, List]):
        """
        Index the TensorDict

        Parameters
        ----------
        ind : int | list
            the index/indices

        Returns
        -------
        dict | TensorDict
            the indexed elements of all value lists combined in a dictionary (if length of ind is 1) or a TensorDict
        """

        x = {key: self.dict[key][ind] for key in self.dict}
        if type(ind) is not int:
            d = TensorDict(x)
            return d
        else:
            return x

    def append(self, element: Dict) -> None:
        """
        Append an element

        Parameters
        ----------
        element : dict
            a dictionary
        """

        type_element = type(element[list(element.keys())[0]])
        if self.type is None:
            self.dict = {k: v.unsqueeze(0) for k, v in element.items()}
            self.keys = list(element.keys())
            self.type = type_element
        else:
            for key in self.keys:
                if key not in element:
                    raise ValueError(
                        f"The dictionary appended to TensorDict needs to have the same keys as the "
                        f"TensorDict; got {element.keys()} and {self.keys}"
                    )
                self.dict[key] = torch.cat([self.dict[key], element[key].unsqueeze(0)])

    def remove(self, indices: List) -> None:
        """
        Remove indexed elements

        Parameters
        ----------
        indices : list
            the indices to remove
        """

        mask = torch.ones(len(self))
        mask[indices] = 0
        mask = mask.bool()
        for key, value in self.dict.items():
            self.dict[key] = value[mask]


def apply_threshold(
    tensor: torch.Tensor,
    threshold: float,
    low: bool = True,
    error_mask: torch.Tensor = None,
    min_frames: int = 0,
    smooth_interval: int = 0,
    masked_intervals: List = None,
):
    """
    Apply a hard threshold to a tensor and return indices of the intervals that passed

    If `error_mask` is not `None`, the elements marked `False` are treated as if they did not pass the threshold.
    If `min_frames` is not 0, the intervals are additionally filtered by length.

    Parameters
    ----------
    tensor : torch.Tensor
        the tensor to apply the threshold to
    threshold : float
        the threshold
    error_mask : torch.Tensor, optional
        a boolean real_lens to apply to the results
    min_frames : int, default 0
        the minimum number of frames in the resulting intervals (shorter intervals are discarded)

    Returns
    -------
    indices_start : list
        a list of indices of the first frames of the chosen intervals
    indices_end : list
        a list of indices of the last frames of the chosen intervals
    """

    if masked_intervals is None:
        masked_intervals = []
    if low:
        p = tensor <= threshold
    else:
        p = tensor >= threshold
    p = smooth(p, smooth_interval)
    if error_mask is not None:
        p = p * error_mask
    for start, end in masked_intervals:
        p[start:end] = False
    output, indices, counts = torch.unique_consecutive(
        p, return_inverse=True, return_counts=True
    )
    long_indices = torch.where(output * (counts > min_frames))[0]
    res_indices_start = [
        (indices == i).nonzero(as_tuple=True)[0][0].item() for i in long_indices
    ]
    res_indices_end = [
        (indices == i).nonzero(as_tuple=True)[0][-1].item() + 1 for i in long_indices
    ]
    return res_indices_start, res_indices_end


def apply_threshold_hysteresis(
    tensor: torch.Tensor,
    soft_threshold: float,
    hard_threshold: float,
    low: bool = True,
    error_mask: torch.Tensor = None,
    min_frames: int = 0,
    smooth_interval: int = 0,
    masked_intervals: List = None,
):
    """
    Apply a hysteresis threshold to a tensor and return indices of the intervals that passed

    In the chosen intervals all values pass the soft threshold and at least one value passes the hard threshold.
    If `error_mask` is not `None`, the elements marked `False` are treated as if they did not pass the threshold.
    If `min_frames` is not 0, the intervals are additionally filtered by length.

    Parameters
    ----------
    tensor : torch.Tensor
        the tensor to apply the threshold to
    soft_threshold : float
        the soft threshold
    hard_threshold : float
        the hard threshold
    error_mask : torch.Tensor, optional
        a boolean real_lens to apply to the results
    min_frames : int, default 0
        the minimum number of frames in the resulting intervals (shorter intervals are discarded)

    Returns
    -------
    indices_start : list
        a list of indices of the first frames of the chosen intervals
    indices_end : list
        a list of indices of the last frames of the chosen intervals
    """

    if masked_intervals is None:
        masked_intervals = []
    if low:
        p = tensor <= soft_threshold
        hard = tensor <= hard_threshold
    else:
        p = tensor >= soft_threshold
        hard = tensor >= hard_threshold
    p = smooth(p, smooth_interval)
    if error_mask is not None:
        p = p * error_mask
    for start, end in masked_intervals:
        p[start:end] = False
    output, indices, counts = torch.unique_consecutive(
        p, return_inverse=True, return_counts=True
    )
    long_indices = torch.where(output * (counts > min_frames))[0]
    indices_start = [
        (indices == i).nonzero(as_tuple=True)[0][0].item() for i in long_indices
    ]
    indices_end = [
        (indices == i).nonzero(as_tuple=True)[0][-1].item() + 1 for i in long_indices
    ]
    res_indices_start = []
    res_indices_end = []
    for start, end in zip(indices_start, indices_end):
        if torch.sum(hard[start:end]) > 0:
            res_indices_start.append(start)
            res_indices_end.append(end)
    return res_indices_start, res_indices_end


def apply_threshold_max(
    tensor: torch.Tensor,
    threshold: float,
    main_class: int,
    error_mask: torch.Tensor = None,
    min_frames: int = 0,
    smooth_interval: int = 0,
    masked_intervals: List = None,
):
    """
    Apply a max hysteresis threshold to a tensor and return indices of the intervals that passed

    In the chosen intervals the values at the `main_class` index are larger than the others everywhere
    and at least one value at the `main_class` index passes the threshold.
    If `error_mask` is not `None`, the elements marked `False`are treated as if they did not pass the threshold.
    If min_frames is not 0, the intervals are additionally filtered by length.

    Parameters
    ----------
    tensor : torch.Tensor
        the tensor to apply the threshold to (of shape `(#classes, #frames)`)
    threshold : float
        the threshold
    main_class : int
        the class that conditions the soft threshold
    error_mask : torch.Tensor, optional
        a boolean real_lens to apply to the results
    min_frames : int, default 0
        the minimum number of frames in the resulting intervals (shorter intervals are discarded)

    Returns
    -------
    indices_start : list
        a list of indices of the first frames of the chosen intervals (along dimension 1 of input tensor)
    indices_end : list
        a list of indices of the last frames of the chosen intervals (along dimension 1 of input tensor)
    """

    if masked_intervals is None:
        masked_intervals = []
    _, indices = torch.max(tensor, dim=0)
    p = indices == main_class
    p = smooth(p, smooth_interval)
    if error_mask is not None:
        p = p * error_mask
    for start, end in masked_intervals:
        p[start:end] = False
    output, indices, counts = torch.unique_consecutive(
        p, return_inverse=True, return_counts=True
    )
    long_indices = torch.where(output * (counts > min_frames))[0]
    indices_start = [
        (indices == i).nonzero(as_tuple=True)[0][0].item() for i in long_indices
    ]
    indices_end = [
        (indices == i).nonzero(as_tuple=True)[0][-1].item() + 1 for i in long_indices
    ]
    res_indices_start = []
    res_indices_end = []
    if threshold is not None:
        hard = tensor[main_class, :] > threshold
        for start, end in zip(indices_start, indices_end):
            if torch.sum(hard[start:end]) > 0:
                res_indices_start.append(start)
                res_indices_end.append(end)
        return res_indices_start, res_indices_end
    else:
        return indices_start, indices_end


def strip_suffix(text: str, suffix: Iterable):
    """
    Strip a suffix from a string if it is contained in a list

    Parameters
    ----------
    text : str
        the main string
    suffix : iterable
        the list of suffices to be stripped

    Returns
    -------
    result : str
        the stripped string
    """

    for s in suffix:
        if text.endswith(s):
            return text[: -len(s)]
    return text


def strip_prefix(text: str, prefix: Iterable):
    """
    Strip a prefix from a string if it is contained in a list

    Parameters
    ----------
    text : str
        the main string
    prefix : iterable
        the list of prefixes to be stripped

    Returns
    -------
    result : str
        the stripped string
    """

    if prefix is None:
        prefix = []
    for s in prefix:
        if text.startswith(s):
            return text[len(s) :]
    return text


def rotation_matrix_2d(angles: torch.Tensor) -> torch.Tensor:
    """
    Create a tensor of 2D rotation matrices from a tensor of angles

    Parameters
    ----------
    angles : torch.Tensor
        a tensor of angles of arbitrary shape `(...)`

    Returns
    -------
    rotation_matrices : torch.Tensor
        a tensor of 2D rotation matrices of shape `(..., 2, 2)`
    """

    cos = torch.cos(angles)
    sin = torch.sin(angles)
    R = torch.stack([cos, -sin, sin, cos], dim=-1).reshape(*angles.shape, 2, 2)
    return R


def rotation_matrix_3d(alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor):
    """
    Create a tensor of 3D rotation matrices from a tensor of angles

    Parameters
    ----------
    alpha : torch.Tensor
        a tensor of rotation angles around the x axis of arbitrary shape `(...)`
    beta : torch.Tensor
        a tensor of rotation angles around the y axis of arbitrary shape `(...)`
    gamma : torch.Tensor
        a tensor of rotation angles around the z axis of arbitrary shape `(...)`

    Returns
    -------
    rotation_matrices : torch.Tensor
        a tensor of 3D rotation matrices of shape `(..., 3, 3)`
    """

    cos = torch.cos(alpha)
    sin = torch.sin(alpha)
    Rx = torch.stack(
        [
            torch.ones(cos.shape),
            torch.zeros(cos.shape),
            torch.zeros(cos.shape),
            torch.zeros(cos.shape),
            cos,
            -sin,
            torch.zeros(cos.shape),
            sin,
            cos,
        ],
        dim=-1,
    ).reshape(*alpha.shape, 3, 3)
    cos = torch.cos(beta)
    sin = torch.sin(beta)
    Ry = torch.stack(
        [
            cos,
            torch.zeros(cos.shape),
            sin,
            torch.zeros(cos.shape),
            torch.ones(cos.shape),
            torch.zeros(cos.shape),
            -sin,
            torch.zeros(cos.shape),
            cos,
        ],
        dim=-1,
    ).reshape(*beta.shape, 3, 3)
    cos = torch.cos(gamma)
    sin = torch.sin(gamma)
    Rz = torch.stack(
        [
            cos,
            -sin,
            torch.zeros(cos.shape),
            sin,
            cos,
            torch.zeros(cos.shape),
            torch.zeros(cos.shape),
            torch.zeros(cos.shape),
            torch.ones(cos.shape),
        ],
        dim=-1,
    ).reshape(*gamma.shape, 3, 3)
    R = torch.einsum("...ij,...jk,...kl->...il", Rx, Ry, Rz)
    return R


def correct_path(path, project_path):
    if not isinstance(path, str):
        return path
    path = os.path.normpath(path).split(os.path.sep)
    if "results" in path:
        name = "results"
    else:
        name = "saved_datasets"
    ind = path.index(name) + 1
    return os.path.join(project_path, "results", *path[ind:])


class TensorList(list):
    """
    A list of tensors that can send each element to a `torch` device
    """

    def to_device(self, device: torch.device):
        for i, x in enumerate(self):
            self[i] = x.to_device(device)


def get_intervals(tensor: torch.Tensor) -> torch.Tensor:
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


def smooth(tensor: torch.Tensor, smooth_interval: int = 0) -> torch.Tensor:
    """
    Get rid of jittering in a non-exclusive classification tensor

    First, remove intervals of 0 shorter than `smooth_interval`. Then, remove intervals of 1 shorter than
    `smooth_interval`.
    """

    if smooth_interval == 0:
        return tensor
    intervals = get_intervals(tensor == 0)
    interval_lengths = torch.tensor(
        [interval[1] - interval[0] for interval in intervals]
    )
    short_intervals = intervals[interval_lengths <= smooth_interval]
    for start, end in short_intervals:
        tensor[start:end] = 1
    intervals = get_intervals(tensor == 1)
    interval_lengths = torch.tensor(
        [interval[1] - interval[0] for interval in intervals]
    )
    short_intervals = intervals[interval_lengths <= smooth_interval]
    for start, end in short_intervals:
        tensor[start:end] = 0
    return tensor


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a 1d tensor.
    Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
    """

    def __init__(self, kernel_size: int = 15, sigma: float = 1.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrid = torch.meshgrid(torch.arange(kernel_size))[0].float()

        mean = (kernel_size - 1) / 2
        kernel = kernel / (sigma * math.sqrt(2 * math.pi))
        kernel = kernel * torch.exp(-(((meshgrid - mean) / sigma) ** 2) / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        # kernel = kernel / torch.max(kernel)

        self.kernel = kernel.view(1, 1, *kernel.size())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        _, c, _ = inputs.shape
        inputs = F.pad(
            inputs,
            pad=((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2),
            mode="reflect",
        )
        kernel = self.kernel.repeat(c, *[1] * (self.kernel.dim() - 1)).to(inputs.device)
        return F.conv1d(inputs, weight=kernel, groups=c)


def argrelmax(prob: np.ndarray, threshold: float = 0.7) -> List[int]:
    """
    Calculate arguments of relative maxima.
    prob: np.array. boundary probability maps distributerd in [0, 1]
    prob shape is (T)
    ignore the peak whose value is under threshold
    Return:
        Index of peaks for each batch
    """
    # ignore the values under threshold
    prob[prob < threshold] = 0.0

    # calculate the relative maxima of boundary maps
    # treat the first frame as boundary
    peak = np.concatenate(
        [
            np.ones((1), dtype=np.bool),
            (prob[:-2] < prob[1:-1]) & (prob[2:] < prob[1:-1]),
            np.zeros((1), dtype=np.bool),
        ],
        axis=0,
    )

    peak_idx = np.where(peak)[0].tolist()

    return peak_idx


def decide_boundary_prob_with_similarity(x: torch.Tensor) -> torch.Tensor:
    """
    Decide action boundary probabilities based on adjacent frame similarities.
    Args:
        x: frame-wise video features (N, C, T)
    Return:
        boundary: action boundary probability (N, 1, T)
    """
    device = x.device

    # gaussian kernel.
    diff = x[0, :, 1:] - x[0, :, :-1]
    similarity = torch.exp(-torch.norm(diff, dim=0) / (2 * 1.0))

    # define action starting point as action boundary.
    start = torch.ones(1).float().to(device)
    boundary = torch.cat([start, similarity])
    boundary = boundary.view(1, 1, -1)
    return boundary


class PostProcessor(object):
    def __init__(
        self,
        name: str,
        boundary_th: int = 0.7,
        theta_t: int = 15,
        kernel_size: int = 15,
    ) -> None:
        self.func = {
            "refinement_with_boundary": self._refinement_with_boundary,
            "relabeling": self._relabeling,
            "smoothing": self._smoothing,
        }
        assert name in self.func

        self.name = name
        self.boundary_th = boundary_th
        self.theta_t = theta_t
        self.kernel_size = kernel_size

        if name == "smoothing":
            self.filter = GaussianSmoothing(self.kernel_size)

    def _is_probability(self, x: np.ndarray) -> bool:
        assert x.ndim == 3

        if x.shape[1] == 1:
            # sigmoid
            if x.min() >= 0 and x.max() <= 1:
                return True
            else:
                return False
        else:
            # softmax
            _sum = np.sum(x, axis=1).astype(np.float32)
            _ones = np.ones_like(_sum, dtype=np.float32)
            return np.allclose(_sum, _ones)

    def _convert2probability(self, x: np.ndarray) -> np.ndarray:
        """
        Args: x (N, C, T)
        """
        assert x.ndim == 3

        if self._is_probability(x):
            return x
        else:
            if x.shape[1] == 1:
                # sigmoid
                prob = 1 / (1 + np.exp(-x))
            else:
                # softmax
                prob = np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=1), 1)
            return prob.astype(np.float32)

    def _convert2label(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2 or x.ndim == 3

        if x.ndim == 2:
            return x.astype(np.int64)
        else:
            if not self._is_probability(x):
                x = self._convert2probability(x)

            label = np.argmax(x, axis=1)
            return label.astype(np.int64)

    def _refinement_with_boundary(
        self,
        outputs: np.array,
        boundaries: np.ndarray,
    ) -> np.ndarray:
        """
        Get segments which is defined as the span b/w two boundaries,
        and decide their classes by majority vote.
        Args:
            outputs: numpy array. shape (N, C, T)
                the model output for frame-level class prediction.
            boundaries: numpy array.  shape (N, 1, T)
                boundary prediction.
            masks: np.array. np.bool. shape (N, 1, T)
                valid length for each video
        Return:
            preds: np.array. shape (N, T)
                final class prediction considering boundaries.
        """

        preds = self._convert2label(outputs)
        boundaries = self._convert2probability(boundaries)

        for i, (output, pred, boundary) in enumerate(zip(outputs, preds, boundaries)):
            idx = argrelmax(boundary.squeeze(), threshold=self.boundary_th)

            # add the index of the last action ending
            T = pred.shape[0]
            idx.append(T)

            # majority vote
            for j in range(len(idx) - 1):
                count = np.bincount(pred[idx[j] : idx[j + 1]])
                modes = np.where(count == count.max())[0]
                if len(modes) == 1:
                    mode = modes
                else:
                    if outputs.ndim == 3:
                        # if more than one majority class exist
                        prob_sum_max = 0
                        for m in modes:
                            prob_sum = output[m, idx[j] : idx[j + 1]].sum()
                            if prob_sum_max < prob_sum:
                                mode = m
                                prob_sum_max = prob_sum
                    else:
                        # decide first mode when more than one majority class
                        # have the same number during oracle experiment
                        mode = modes[0]

                preds[i, idx[j] : idx[j + 1]] = mode

        return preds

    def _relabeling(self, outputs: np.ndarray, **kwargs: np.ndarray) -> np.ndarray:
        """
        Relabeling small action segments with their previous action segment
        Args:
            output: the results of action segmentation. (N, T) or (N, C, T)
            theta_t: the threshold of the size of action segments.
        Return:
            relabeled output. (N, T)
        """

        preds = self._convert2label(outputs)

        for i in range(preds.shape[0]):
            # shape (T,)
            last = preds[i][0]
            cnt = 1
            for j in range(1, preds.shape[1]):
                if last == preds[i][j]:
                    cnt += 1
                else:
                    if cnt > self.theta_t:
                        cnt = 1
                        last = preds[i][j]
                    else:
                        preds[i][j - cnt : j] = preds[i][j - cnt - 1]
                        cnt = 1
                        last = preds[i][j]

            if cnt <= self.theta_t:
                preds[i][j - cnt : j] = preds[i][j - cnt - 1]

        return preds

    def _smoothing(self, outputs: np.ndarray, **kwargs: np.ndarray) -> np.ndarray:
        """
        Smoothing action probabilities with gaussian filter.
        Args:
            outputs: frame-wise action probabilities. (N, C, T)
        Return:
            predictions: final prediction. (N, T)
        """

        outputs = self._convert2probability(outputs)
        outputs = self.filter(torch.Tensor(outputs)).numpy()

        preds = self._convert2label(outputs)
        return preds

    def __call__(self, outputs, **kwargs: np.ndarray) -> np.ndarray:

        preds = self.func[self.name](outputs, **kwargs)
        return preds
