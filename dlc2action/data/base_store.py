#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
"""
Abstract parent classes for the store objects
"""

import os.path
from typing import Dict, Union, List, Tuple, Set, Optional
from abc import ABC, abstractmethod
import numpy as np
import inspect
import torch


class Store(ABC):  # +
    """
    A general parent class for `AnnotationStore` and `InputStore`

    Processes input video information and generates ordered arrays of data samples and corresponding unique
    original coordinates, as well as some meta objects.
    It is assumed that the input videos are separated into clips (e.g. corresponding to different individuals).
    Each video and each clip inside the video has a unique id (video_id and clip_id, correspondingly).
    The original coordinates object contains information about the video_id, clip_id and start time of the
    samples in the original input data.
    A Store has to be fully defined with a tuple of key objects.
    The data array can be accessed with integer indices.
    The samples can be stored as a tensor or TensorDict in RAM or as an array of file paths to be loaded on runtime.
    """

    @abstractmethod
    def __len__(self) -> int:
        """
        Get the number of available samples

        Returns
        -------
        length : int
            the number of available samples
        """

    @abstractmethod
    def remove(self, indices: List) -> None:
        """
        Remove the samples corresponding to indices

        Parameters
        ----------
        indices : int
            a list of integer indices to remove
        """

    @abstractmethod
    def key_objects(self) -> Tuple:
        """
        Return a tuple of the key objects necessary to re-create the Store

        Returns
        -------
        key_objects : tuple
            a tuple of key objects
        """

    @abstractmethod
    def load_from_key_objects(self, key_objects: Tuple) -> None:
        """
        Load the information from a tuple of key objects

        Parameters
        ----------
        key_objects : tuple
            a tuple of key objects
        """

    @abstractmethod
    def to_ram(self) -> None:
        """
        Transfer the data samples to RAM if they were previously stored as file paths
        """

    @abstractmethod
    def get_original_coordinates(self) -> np.ndarray:
        """
        Return the original coordinates array

        Returns
        -------
        np.ndarray
            an array that contains the coordinates of the data samples in original input data (video id, clip id,
            start frame)
        """

    @abstractmethod
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

    @classmethod
    @abstractmethod
    def get_file_ids(cls, *args, **kwargs) -> List:
        """
        Process data parameters and return a list of ids  of the videos that should
        be processed by the __init__ function

        Returns
        -------
        video_ids : list
            a list of video file ids
        """

    @classmethod
    def get_parameters(cls) -> List:
        """
        Generate a list of parameter names for the __init__ function

        Returns
        -------
        parameter_names: list
            a list of necessary parameter names
        """

        return inspect.getfullargspec(cls.__init__).args

    @classmethod
    def new(cls):
        """
        Create a new instance of the same class

        Returns
        -------
        new_instance : Store
            a new instance of the same class
        """

        return cls()


class InputStore(Store):  # +
    """
    A class that generates model input data from video information and stores it

    Processes input video information and generates ordered arrays of data samples and corresponding unique
    original coordinates, as well as some meta objects.
    It is assumed that the input videos are separated into clips (e.g. corresponding to different individuals).
    Each video and each clip inside the video has a unique id (video_id and clip_id, correspondingly).
    The original coordinates object contains information about the video_id, clip_id and start time of the
    samples in the original input data.
    An InputStore has to be fully defined with a tuple of key objects.
    The data array can be accessed with integer indices.
    The samples can be stored as a TensorDict in RAM or as an array of file paths to be loaded on runtime.
    When no arguments are passed a blank class instance should be created that can later be filled with
    information from key objects
    """

    @abstractmethod
    def __init__(
        self,
        video_order: List = None,
        key_objects: Tuple = None,
        data_path: Union[str, List] = None,
        file_paths: List = None,
        feature_save_path: str = None,
        feature_extraction_pars: Dict = None,
        *args,
        **kwargs
    ):
        """
        Parameters
        ----------
        video_order : list, optional
            a list of video ids that should be processed in the same order (not passed if creating from key objects)
        key_objects : tuple, optional
            a tuple of key objects
        data_path : str | set, optional
            the path to the folder where the pose and feature files are stored or a set of such paths
            (not passed if creating from key objects or from `file_paths`)
        file_paths : set, optional
            a set of string paths to the pose and feature files
            (not passed if creating from key objects or from `data_path`)
        feature_save_path : str, optional
            the path to the folder where pre-processed files are stored (not passed if creating from key objects)
        feature_extraction_pars : dict, optional
            a dictionary of feature extraction parameters (not passed if creating from key objects)
        """

        if key_objects is not None:
            self.load_from_key_objects(key_objects)

    @abstractmethod
    def __getitem__(self, ind: int) -> Dict:
        """
        Return the sample corresponding to an index

        Parameters
        ----------
        ind : int
            index of the sample

        Returns
        -------
        sample : dict
            the corresponding sample (a dictionary of features)
        """

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def get_clip_start_end(self, coords: Tuple) -> Tuple[int, int]:
        """
        Get the clip start and end frames from an element of original coordinates

        Parameters
        ----------
        coords : tuple
            an element of original coordinates array

        Returns
        -------
        start: int
            the start frame of the clip that the coordinates point to
        end : int
            the end frame of the clip that the coordinates point to
        """

    @abstractmethod
    def get_clip_start(self, video_id: str, clip_id: str) -> int:
        """
        Get the clip start frame from the video id and the clip id

        Parameters
        ----------
        video_id : str
            the video id
        clip_id : str
            the clip id

        Returns
        -------
        clip_start : int
            the start frame of the clip
        """

    @abstractmethod
    def get_visibility(
        self, video_id: str, clip_id: str, start: int, end: int, score: float
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

    @abstractmethod
    def get_annotation_objects(self) -> Dict:
        """
        Get a dictionary of objects necessary to create an AnnotationStore

        Returns
        -------
        annotation_objects : dict
            a dictionary of objects to be passed to the AnnotationStore constructor where the keys are the names of
            the objects
        """

    @abstractmethod
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

    def get_clip_length_from_coords(self, coords: Tuple) -> int:
        """
        Get the length of a clip from an element of the original coordinates array

        Parameters
        ----------
        coords : tuple
            an element of the original coordinates array

        Returns
        -------
        clip_length : int
            the length of the clip
        """

        v_id = self.get_video_id(coords)
        clip_id = self.get_clip_id(coords)
        l = self.get_clip_length(v_id, clip_id)
        return l

    def get_folder_order(self) -> List:
        """
        Get a list of folders corresponding to the data samples

        Returns
        -------
        folder_order : list
            a list of string folder basenames corresponding to the data samples (e.g. 'folder2'
            if the corresponding file was read from '/path/to/folder1/folder2')
        """

        return [os.path.basename(self.get_folder(x)) for x in self.get_video_id_order()]

    def get_video_id_order(self) -> List:
        """
        Get a list of video ids corresponding to the data samples

        Returns
        -------
        video_id_order : list
            a list of string names of the video ids corresponding to the data samples
        """

        return [self.get_video_id(x) for x in self.get_original_coordinates()]

    def get_tag(self, idx: int) -> Union[int, None]:
        """
        Return a tag object corresponding to an index

        Tags can carry meta information (like annotator id) and are accepted by models that require
        that information and by metrics (some metrics have options for averaging over the tags).
        When a tag is `None`, it is not passed to the model.

        Parameters
        ----------
        idx : int
            the index

        Returns
        -------
        tag : int
            the tag index
        """

        return None

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

        return list(range(len(self)))

    def get_tags(self) -> List:
        """
        Get a list of all meta tags

        Returns
        -------
        tags: List
            a list of unique meta tag values
        """

        return [None]


class AnnotationStore(Store):
    """
    A class that generates annotation from video information and stores it

    Processes input video information and generates ordered arrays of annotation samples and corresponding unique
    original coordinates, as well as some meta objects.
    It is assumed that the input videos are separated into clips (e.g. corresponding to different individuals).
    Each video and each clip inside the video has a unique id (video_id and clip_id, correspondingly).
    The original coordinates object contains information about the video_id, clip_id and start time of the
    samples in the original input data.
    An AnnotationStore has to be fully defined with a tuple of key objects.
    The annotation array can be accessed with integer indices.
    The samples can be stored as a torch.Tensor in RAM or as an array of file paths to be loaded on runtime.
    When no arguments are passed a blank class instance should be created that can later be filled with
    information from key objects
    """

    required_objects = []
    """
    A list of string names of the objects required from the input store
    """

    @abstractmethod
    def __init__(
        self,
        video_order: List = None,
        key_objects: Tuple = None,
        annotation_path: Union[str, Set] = None,
        *args,
        **kwargs
    ):
        """
        Parameters
        ----------
        video_order : list, optional
            a list of video ids that should be processed in the same order (not passed if creating from key objects)
        key_objects : tuple, optional
            a tuple of key objects
        annotation_path : str | set, optional
            the path or the set of paths to the folder where the annotation files are stored (not passed if creating
            from key objects)
        """

        if key_objects is not None:
            self.load_from_key_objects(key_objects)

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def count_classes(
        self, frac: bool = False, zeros: bool = False, bouts: bool = False
    ) -> Dict:
        """
        Get a dictionary with class-wise frame counts

        Parameters
        ----------
        frac : bool, default False
            if `True`, a fraction of the total frame count is returned
        zeros : bool. default False
            if `True`, the number of known negative samples is counted (only if the annotation is multi-label)
        bouts : bool, default False
            if `True`, instead of frame counts segment counts are returned

        Returns
        -------
        count_dictionary : dict
            a dictionary with class indices as keys and frame counts as values

        """

    @abstractmethod
    def behaviors_dict(self) -> Dict:
        """
        Get a dictionary of class names

        Returns
        -------
        behavior_dictionary: dict
            a dictionary with class indices as keys and class names as values
        """

    @abstractmethod
    def annotation_class(self) -> str:
        """
        Get the type of annotation ('exclusive_classification', 'nonexclusive_classification', more coming soon)

        Returns
        -------
        annotation_class : str
            the type of annotation
        """

    @abstractmethod
    def size(self) -> int:
        """
        Get the total number of frames in the data

        Returns
        -------
        size : int
            the total number of frames
        """

    @abstractmethod
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

    @abstractmethod
    def set_pseudo_labels(self, labels: torch.Tensor) -> None:
        """
        Set pseudo labels to the unlabeled data

        Parameters
        ----------
        labels : torch.Tensor
            a tensor of pseudo-labels for the unlabeled data
        """


class PoseInputStore(InputStore):
    """
    A subclass of InputStore for pose estimation data

    Contains methods used by pose estimation feature extractors.
    All methods receive a data dictionary as input. This dictionary is the same as what is passed to the
    feature extractor and the only limitations for the structure are that it has to relate to one video id
    and have clip ids as keys. Read the documentation at `dlc2action.data` to find out more about videos
    and clips.
    """

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

        return None

    @abstractmethod
    def get_coords(self, data_dict: Dict, clip_id: str, bodypart: str) -> np.ndarray:
        """
        Get the coordinates array of a specific body part in a specific clip

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
        coords : np.ndarray
            the coordinates array of shape (#timesteps, #coordinates)
        """

    @abstractmethod
    def get_bodyparts(self) -> List:
        """
        Get a list of bodypart names

        Returns
        -------
        bodyparts : list
            a list of string or integer body part names
        """

    @abstractmethod
    def get_n_frames(self, data_dict: Dict, clip_id: str) -> int:
        """
        Get the length of the clip

        Parameters
        ----------
        data_dict : dict
            the data dictionary
        clip_id : str
            the clip id

        Returns
        -------
        n_frames : int
            the length of the clip
        """
