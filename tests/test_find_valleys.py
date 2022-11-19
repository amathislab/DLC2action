#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#
import torch
import pytest
from dlc2action.data.dataset import BehaviorDataset

data_parameters = {
    "data_type": "dlc_track",
    "annotation_type": "boris",
    "data_path": "/home/liza/data/cricket",
    "annotation_path": "/home/liza/data/cricket",
    "behaviors": ["Grooming", "Search", "Pursuit"],
    "annotation_suffix": {".csv"},
    "data_suffix": {
        "DLC_resnet50_preycapSep30shuffle1_20000_bx_filtered.h5",
    },
    "canvas_shape": [2250, 1250],
    "default_agent_name": "mouse+single",
    "interactive": True,
    "feature_extraction_pars": {"interactive": True},
}


@pytest.mark.parametrize("method", ["hard_threshold", "hysteresis", "hysteresis_max"])
@pytest.mark.parametrize("low", [True, False])
@pytest.mark.parametrize("add_error", [True, False])
@pytest.mark.parametrize("add_min_frames", [True, False])
@pytest.mark.parametrize("add_min_frames_error", [True, False])
def test_find_valleys(
    method: str,
    low: bool,
    add_error: bool,
    add_min_frames: bool,
    add_min_frames_error: bool,
):
    """
    Test the `dlc2action.data.dataset.BehaviorDataset.find_valleys` function
    """

    dataset = BehaviorDataset(**data_parameters)
    l = len(dataset)
    f = dataset.len_segment()
    if add_error:
        predicted_error = torch.zeros(l, 2, f)
        predicted_error[0, 1, f // 4 :] = 1
    else:
        predicted_error = None
    if add_min_frames_error:
        min_frames_error = f // 2
    else:
        min_frames_error = 0
    if low:
        prediction = torch.ones(l, 3, f)
        value = 0
        soft_value = 0.55
    else:
        prediction = torch.zeros(l, 3, f)
        value = 1
        soft_value = 0.45
    if method == "hard_threshold":
        prediction[0, 0, : f // 2] = value
        prediction[-2, 0, f // 4 : -f // 4] = value
        hysteresis = False
        threshold_diff = 0
    elif method.startswith("hysteresis"):
        if method.endswith("max"):
            threshold_diff = None
            prediction[:, 1:, :] = 0.01
        else:
            threshold_diff = 0.1
        prediction[0, 0, : f // 2] = soft_value
        prediction[0, 0, 0] = value
        prediction[-2, 0, f // 4 : -f // 4] = soft_value
        prediction[-2, 0, f // 2] = value
        hysteresis = True
    else:
        raise ValueError(f"Method {method} unrecognized!")
    if add_min_frames:
        min_frames = f + 1
    else:
        min_frames = 0
    if low and method == "hysteresis_max":
        with pytest.raises(ValueError):
            dataset.find_valleys(
                predicted=prediction,
                threshold=0.5,
                main_class=0,
                low=low,
                min_frames=min_frames,
                predicted_error=predicted_error,
                hysteresis=hysteresis,
                threshold_diff=threshold_diff,
                min_frames_error=min_frames_error,
            )
    else:
        valleys = dataset.find_valleys(
            predicted=prediction,
            threshold=0.5,
            main_class=0,
            low=low,
            min_frames=min_frames,
            predicted_error=predicted_error,
            hysteresis=hysteresis,
            threshold_diff=threshold_diff,
            min_frames_error=min_frames_error,
        )
        coords_first = dataset.input_store.get_original_coordinates()[0]
        coords_last = dataset.input_store.get_original_coordinates()[-2]
        clip_id_first = dataset.input_store.get_clip_id(coords_first)
        video_id_first = dataset.input_store.get_video_id(coords_first)
        start_first, _ = dataset.input_store.get_clip_start_end(coords_first)
        clip_id_last = dataset.input_store.get_clip_id(coords_last)
        video_id_last = dataset.input_store.get_video_id(coords_last)
        start_last, end_last = dataset.input_store.get_clip_start_end(coords_last)
        if add_min_frames:
            answer = []
        else:
            answer = [
                (video_id_last, clip_id_last, start_last + f // 4, end_last - f // 4)
            ]
            if add_error and not add_min_frames_error:
                answer.append(
                    (video_id_first, clip_id_first, start_first, start_first + f // 4)
                )
            elif not add_error:
                answer.append(
                    (video_id_first, clip_id_first, start_first, start_first + f // 2)
                )
        for video_id, clip_id, start, end in answer:
            assert valleys[video_id] == [[start, end, clip_id]]
        answer = [x[0] for x in answer]
        for video_id in valleys:
            if video_id not in answer:
                assert valleys[video_id] == []


# test_find_valleys('hard_threshold', low=True, add_error=True, add_min_frames=False, add_min_frames_error=True)
