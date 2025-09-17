#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. 
# A copy is included in dlc2action/LICENSE.AGPL.
#
import numpy as np
import pandas as pd
import cv2
import os
from dataclasses import dataclass
from tqdm import tqdm
from multiprocessing import Pool


@dataclass
class ProjectConfig:
    pose_path: str
    video_path: str
    mapping_file_path: str
    output_dir: str


def bbox_from_keypoints(
    keypoints: np.ndarray,
    image_h: int,
    image_w: int,
    margin: int,
) -> np.ndarray:
    """
    Computes bounding boxes from keypoints. Copied from https://github.com/DeepLabCut/DeepLabCut/blob/763135804206702344ff9bf91a044eb305636a7d/deeplabcut/pose_estimation_pytorch/data/utils.py#L33

    Args:
        keypoints: (..., num_keypoints, xy) the keypoints from which to get bboxes
        image_h: the height of the image
        image_w: the width of the image
        margin: the bounding box margin

    Returns:
        the bounding boxes for the keypoints, of shape (..., 4) in the xywh format
    """
    squeeze = False

    # we do not estimate bbox on keypoints that have 0 or -1 flag
    keypoints = np.copy(keypoints)
    keypoints[keypoints[..., -1] <= 0] = np.nan

    if len(keypoints.shape) == 2:
        squeeze = True
        keypoints = np.expand_dims(keypoints, axis=0)

    bboxes = np.full((keypoints.shape[0], 4), np.nan)
    bboxes[:, :2] = np.nanmin(keypoints[..., :2], axis=1) - margin  # X1, Y1
    bboxes[:, 2:4] = np.nanmax(keypoints[..., :2], axis=1) + margin  # X2, Y2

    # can have NaNs if some individuals have no visible keypoints
    bboxes = np.nan_to_num(bboxes, nan=0)

    bboxes = np.clip(
        bboxes,
        a_min=[0, 0, 0, 0],
        a_max=[image_w, image_h, image_w, image_h],
    )
    bboxes[..., 2] = bboxes[..., 2] - bboxes[..., 0]  # to width
    bboxes[..., 3] = bboxes[..., 3] - bboxes[..., 1]  # to height
    if squeeze:
        return bboxes[0]

    return bboxes


def crop_frame(video_capture: cv2.VideoCapture, bbox: list, num_frames: int) -> list:
    """Crop frames from the video based on the bounding boxes."""
    frame_count = 0
    frames = []
    with tqdm(total=num_frames, desc="Cropping frames") as pbar:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            # Get the bounding box for the current frame
            x, y, w, h = bbox[frame_count]
            # Crop the frame
            crop_frame = frame[int(y) : int(y + h), int(x) : int(x + w)]
            # Resize the cropped frame to 224x224
            crop_frame = cv2.resize(crop_frame, (224, 224))
            frames.append(crop_frame)
            frame_count += 1
            pbar.update(1)

    return frames


def crop_video(config: ProjectConfig, video: str, pose_file: str) -> None:
    video_path = os.path.join(config.video_path, video)

    video_capture = cv2.VideoCapture(video_path)
    image_h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    image_w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    pose_path = os.path.join(config.pose_path, pose_file)
    pose_data = pd.read_csv(pose_path)
    pose_data = pose_data.values[
        2:, 1:
    ]  # Remove the first column (frame number) and the first 3 rows
    pose_data = pose_data.reshape(pose_data.shape[0], -1, 3)
    pose_data = pose_data.astype(float)
    pose_data = pose_data[:, 7:, :]  # Remove the arena keypoints

    bbox = bbox_from_keypoints(
        pose_data[:, :, :2],
        image_h=image_h,
        image_w=image_w,
        margin=10,
    )

    frames = crop_frame(video_capture, bbox, num_frames)

    output_path = os.path.join(
        config.output_dir,
        pose_file.split("DeepCut_resnet50_Blockcourse1May9shuffle1_1030000.csv")[0]
        + "_cropped.mp4",
    )
    os.makedirs(config.output_dir, exist_ok=True)
    print(output_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    height, width, _ = frames[0].shape
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        video_writer.write(frame)

    video_writer.release()
    video_capture.release()


if __name__ == "__main__":
    config = ProjectConfig(
        pose_path="path/to/OFT/Output_DLC",
        video_path="path/to/OFT/videos",
        mapping_file_path="path/to/OFT/Labels/AllLabDataOFT_final.csv",
        output_dir="/path/to/OFT/mouse_cropped_videos",
    )

    # Load the mapping file
    mapping_df = pd.read_csv(config.mapping_file_path, sep=";")
    mapping_df = mapping_df[["ID", "DLCFile"]]
    mapping_df = mapping_df.drop_duplicates()

    for video in os.listdir(config.video_path):
        if not video.endswith(".mp4"):
            continue
        pose_file = mapping_df[mapping_df["ID"] == video.split(".mp4")[0]][
            "DLCFile"
        ].values[0]
        crop_video(config, video, pose_file)
