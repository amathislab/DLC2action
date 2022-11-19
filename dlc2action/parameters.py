#
# Copyright 2020-2022 by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version. A copy is included in dlc2action/LICENSE.AGPL.
#

# parameters_dict = {
#     "annotation": {
#         "boris": {
#             "min_frames_action": ("int", 0, 10),
#             "visibility_min_score": ("float", 0, 0.8),
#             "visibility_min_frac": ("float", 0.5, 1),
#         },
#         "calms21": {},
#         "dlc": {
#             "min_frames_action": ("int", 0, 10),
#             "visibility_min_score": ("float", 0, 0.8),
#             "visibility_min_frac": ("float", 0.5, 1),
#         },
#         "pku-mmd": {},
#     },
#     "augmentations": {
#         "heatmap": {},
#         "kinematic": {},
#     },
#     "data": {
#         "calms21": {},
#         "dlc_track": {
#             "likelihood_threshold": ("float", 0, 0.8)
#         },
#         "dlc_tracklet": {
#             "likelihood_threshold": ("float", 0, 0.8),
#             "frame_limit": ("int", 0, 100),
#         },
#         "np_3d": {
#             "frame_limit": ("int", 0, 100),
#         },
#         "pku-mmd": {},
#     },
#     "features": {
#         "heatmap": {
#             "channel_policy": ("categorical", ["color", "black&white", "bp"]),
#             "heatmap_width": ("categorical", [128, 256]),
#             "sigma": ("float", 0, 1),
#         },
#         "kinematic": {}
#     },
#     "model": {
#         "asformer": {
#             "num_decoders": ("int", 1, 4),
#             "num_layers": ("int", 5, 15),
#             "r1": ("int", 1, 4),
#             "r2": ("int", 1, 4),
#             "num_f_maps": ("int_log", 16, 128),
#             "channel_masking_rate": ("float")
# num_layers: 10
# r1: 2
# r2: 2
# num_f_maps: 64
# input_dim: 'dataset_features'
# channel_masking_rate: 0.4"
#         }
#     }
# }
