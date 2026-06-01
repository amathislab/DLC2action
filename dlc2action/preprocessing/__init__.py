#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version.
# A copy is included in dlc2action/LICENSE.AGPL.
#
"""Preprocessing utilities for importing external features into DLC2Action.

This module provides helpers for converting raw video files into per-frame feature
arrays that are compatible with the :class:`~dlc2action.data.input_store.LoadedFeaturesInputStore`
(i.e. ``data_type="features"`` projects).

Example
-------
Extract DINOv2 features from all ``.mp4`` files in a folder and save them
alongside an existing DLC2Action project::

    import dlc2action

    dlc2action.get_visual_features(
        video_folder="/path/to/videos",
        video_suffix=".mp4",
        output_folder="/path/to/features",
    )

"""

from dlc2action.preprocessing.visual_encoders import get_visual_features

__all__ = ["get_visual_features"]
