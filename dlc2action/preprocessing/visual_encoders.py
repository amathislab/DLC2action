#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version.
# A copy is included in dlc2action/LICENSE.AGPL.
#
"""Modular visual encoder registry for preprocessing video data into feature arrays.

Summary
-------
This module implements a lightweight *encoder registry* pattern, making it easy to add
new visual backbone encoders over time without modifying existing code.  Each encoder
is a concrete subclass of :class:`VisualEncoder` and is registered via the
:func:`register_encoder` decorator.  The high-level function
:func:`get_visual_features` iterates over a video folder, calls the requested encoder
on every video, pads the output, and saves the result as a ``.npy`` file in the
dictionary format expected by
:class:`~dlc2action.data.input_store.LoadedFeaturesInputStore`.

Output format
-------------
Every ``.npy`` file is a pickled dictionary of the form::

    {"ind0": np.ndarray(shape=(T + encoder.default_pad_frames, D), dtype=float32)}

where ``T`` is the number of video frames and ``D`` is the feature dimension of the
chosen encoder.  The padding length is **model-specific** and is set via the
:attr:`VisualEncoder.default_pad_frames` class attribute on each encoder subclass.
The padding copies the last frame's feature vector, which aligns the temporal length
with pose-estimation inputs that are padded in the same way.

Adding a new encoder
--------------------
Subclass :class:`VisualEncoder`, set a unique :attr:`VisualEncoder.name` class
attribute, set :attr:`VisualEncoder.default_pad_frames` to the correct value for
that model, implement :meth:`VisualEncoder.encode_video`, and apply the
:func:`register_encoder` decorator::

    @register_encoder
    class MyEncoder(VisualEncoder):
        name = "my_encoder"
        default_suffix = "_my_encoder.npy"
        default_pad_frames = 8  # model-specific padding

        def encode_video(self, video_path: Union[str, Path]) -> np.ndarray:
            ...  # Return float32 array of shape (T, D)

"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Type, Union

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Registry infrastructure
# ---------------------------------------------------------------------------

_ENCODER_REGISTRY: Dict[str, Type["VisualEncoder"]] = {}


def register_encoder(cls: Type["VisualEncoder"]) -> Type["VisualEncoder"]:
    """Class decorator that registers a :class:`VisualEncoder` in the global registry.

    Parameters
    ----------
    cls:
        A concrete :class:`VisualEncoder` subclass with a unique ``name`` attribute.

    Returns
    -------
    cls:
        The unmodified class (decorator is side-effect only).

    Raises
    ------
    ValueError
        If an encoder with the same :attr:`VisualEncoder.name` is already registered.

    """
    if cls.name in _ENCODER_REGISTRY:
        raise ValueError(
            f"An encoder named '{cls.name}' is already registered. "
            "Choose a different name or remove the duplicate."
        )
    _ENCODER_REGISTRY[cls.name] = cls
    return cls


def list_encoders() -> list:
    """Return the names of all registered visual encoders.

    Returns
    -------
    names : list of str
        Sorted list of encoder names that can be passed to :func:`get_visual_features`.

    """
    return sorted(_ENCODER_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class VisualEncoder(ABC):
    """Abstract base class for visual encoders.

    Subclasses must implement :meth:`encode_video` and set a unique string
    :attr:`name` and a :attr:`default_suffix` for the output files.  They
    are registered automatically with the :func:`register_encoder` decorator.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model identifier or local checkpoint path.  Defaults to the
        encoder's built-in :attr:`default_model_name`.
    device : str, optional
        PyTorch device string (e.g. ``"cuda:0"`` or ``"cpu"``).  Auto-detected
        when ``None``.

    """

    name: str = ""
    """Unique registry key used to select this encoder."""

    default_suffix: str = "_visual.npy"
    """Default output file suffix (falls back to this when no suffix is provided)."""

    default_model_name: str = ""
    """Default HuggingFace model checkpoint for this encoder."""

    default_pad_frames: int = 0
    """Number of frames to append (last-frame repeat) after encoding.

    This is a model-specific constant — set it on each concrete subclass to the
    value that is appropriate for that encoder's temporal receptive field or the
    downstream pipeline's expectations.  :func:`get_visual_features` reads this
    attribute automatically so callers never have to specify it manually.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name or self.default_model_name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    @abstractmethod
    def encode_video(self, video_path: Union[str, Path]) -> np.ndarray:
        """Encode a video file into a per-frame feature matrix.

        Parameters
        ----------
        video_path : str or Path
            Absolute or relative path to the video file.

        Returns
        -------
        features : np.ndarray
            Float32 array of shape ``(T, D)`` where ``T`` is the number of
            frames successfully decoded and ``D`` is the feature dimension.

        """


# ---------------------------------------------------------------------------
# DINOv3 encoder
# ---------------------------------------------------------------------------


@register_encoder
class DinoV3Encoder(VisualEncoder):
    """Per-frame visual feature extractor using DINOv3 / DINO-based backbones.

    Uses the HuggingFace ``transformers`` ``image-feature-extraction`` pipeline
    to extract CLS-token features from every frame of a video.  The default
    checkpoint is the ConvNeXt-tiny model pre-trained on LVD-142M.

    The padding length (:attr:`default_pad_frames`) is set to **8**, matching the
    temporal context expected by downstream DLC2Action models when using this
    encoder.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model identifier.  Defaults to
        ``"facebook/dinov3-convnext-tiny-pretrain-lvd1689m"``.
    device : str, optional
        PyTorch device string.  Auto-detected when ``None``.

    """

    name: str = "dinov3"
    default_suffix: str = "_dino.npy"
    default_model_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
    default_pad_frames: int = 8

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        # Lazy import so that the package stays importable even without
        # ``transformers`` installed (a meaningful error is raised at runtime).
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' package is required for the DINOv3 encoder. "
                "Install it with:  pip install transformers accelerate"
            ) from exc

        self._pipeline = hf_pipeline(
            model=self.model_name,
            task="image-feature-extraction",
            device=self.device,
        )

    def encode_video(self, video_path: Union[str, Path]) -> np.ndarray:
        """Extract per-frame DINOv3 features from a video file.

        Parameters
        ----------
        video_path : str or Path
            Path to the input video file.

        Returns
        -------
        features : np.ndarray
            Float32 array of shape ``(T, D)``.

        Raises
        ------
        RuntimeError
            If the video file cannot be opened by OpenCV.

        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = os.path.basename(video_path)
        frame_features = []

        for _ in tqdm(range(n_frames), desc=f"Encoding {video_name}", leave=False):
            ret, img = cap.read()
            if not ret:
                break
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            # The pipeline returns a nested list: [[token_0_features, ...]]
            # The CLS token is at index 0 of the inner list.
            raw = self._pipeline(pil_img)
            token_features = torch.tensor(raw[0])  # shape: (n_tokens, D)
            frame_features.append(token_features[0])  # CLS token

        cap.release()

        if not frame_features:
            raise RuntimeError(
                f"No frames could be decoded from '{video_path}'. "
                "Check that the file is a valid video."
            )

        features = torch.stack(frame_features, dim=0)  # (T, D)
        return features.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_visual_features(
    video_folder: Union[str, Path],
    video_suffix: str,
    output_folder: Union[str, Path],
    encoder: str = "dinov3",
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    clip_id: str = "ind0",
    overwrite: bool = False,
) -> None:
    """Extract per-frame visual features from all videos in a folder.

    Searches *video_folder* for every file whose name ends with *video_suffix*,
    runs the selected visual encoder on each video, pads the output, and saves
    the result as a ``.npy`` dictionary file compatible with
    :class:`~dlc2action.data.input_store.LoadedFeaturesInputStore`.

    The padding length is **model-specific** and is read automatically from the
    encoder's :attr:`VisualEncoder.default_pad_frames` class attribute (e.g. 8
    for :class:`DinoV3Encoder`).  There is no user-facing ``pad_frames``
    parameter because the correct value is part of the encoder's definition.

    The saved dictionary has the form::

        {"ind0": np.ndarray(shape=(T + encoder.default_pad_frames, D), dtype=float32)}

    where ``T`` is the number of video frames and ``D`` is the encoder's feature
    dimension, so it can be used directly in a project with
    ``data_type="features"`` and ``feature_suffix=encoder.default_suffix``.

    Parameters
    ----------
    video_folder : str or Path
        Directory that contains the input video files.
    video_suffix : str
        File extension (or any suffix) used to identify video files, e.g.
        ``".mp4"`` or ``"_cropped.avi"``.
    output_folder : str or Path
        Directory where the ``.npy`` feature files will be saved.  Created
        automatically if it does not yet exist.
    encoder : str, default ``"dinov3"``
        Name of the registered visual encoder to use.  Call
        :func:`list_encoders` to see all available options.
    model_name : str, optional
        Override the encoder's default HuggingFace model checkpoint.
    device : str, optional
        PyTorch device string (e.g. ``"cuda:0"``).  Auto-detected when ``None``.
    clip_id : str, default ``"ind0"``
        The clip / individual identifier used as the dictionary key inside the
        saved ``.npy`` file.  Change this if you need a different agent name.
    overwrite : bool, default False
        If ``True``, re-extract features even if an output file already exists.

    Raises
    ------
    ValueError
        If *encoder* is not a registered encoder name.
    RuntimeError
        If no video files matching *video_suffix* are found in *video_folder*.

    Examples
    --------
    Extract DINOv3 features from all ``.mp4`` files::

        import dlc2action

        dlc2action.get_visual_features(
            video_folder="/path/to/videos",
            video_suffix=".mp4",
            output_folder="/path/to/features",
        )

    Use a custom device and skip already-processed files::

        dlc2action.get_visual_features(
            video_folder="/path/to/videos",
            video_suffix=".mp4",
            output_folder="/path/to/features",
            device="cuda:1",
            overwrite=False,
        )

    """
    if encoder not in _ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder '{encoder}'. "
            f"Available encoders: {list_encoders()}. "
            "Use dlc2action.preprocessing.list_encoders() for an up-to-date list."
        )

    video_folder = Path(video_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Collect matching video files
    video_paths = sorted(
        p for p in video_folder.iterdir() if p.name.endswith(video_suffix)
    )
    if not video_paths:
        raise RuntimeError(
            f"No files ending with '{video_suffix}' were found in '{video_folder}'."
        )

    # Instantiate the encoder once (model loading can be expensive)
    enc: VisualEncoder = _ENCODER_REGISTRY[encoder](
        model_name=model_name,
        device=device,
    )

    for video_path in tqdm(video_paths, desc="Processing videos"):
        # Derive output file name: {stem}{encoder.default_suffix}
        stem = video_path.name[: -len(video_suffix)]
        out_file = output_folder / (stem + enc.default_suffix)

        if out_file.exists() and not overwrite:
            print(f"Skipping '{video_path.name}' — output already exists.")
            continue

        features = enc.encode_video(video_path)  # (T, D)

        # Pad by copying the last frame enc.default_pad_frames times.
        # The padding length is model-specific and defined on the encoder class.
        if enc.default_pad_frames > 0:
            padding = np.repeat(features[-1:], enc.default_pad_frames, axis=0)
            features = np.concatenate([features, padding], axis=0)

        # Save as a dictionary: {clip_id: array} — compatible with LoadedFeaturesInputStore
        out_dict = {clip_id: features}
        np.save(str(out_file), out_dict)
