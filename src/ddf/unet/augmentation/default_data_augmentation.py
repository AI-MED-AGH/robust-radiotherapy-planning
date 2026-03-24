#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
from copy import deepcopy

import numpy as np


default_3D_augmentation_params = {
    "selected_data_channels": None,
    "selected_seg_channels": None,

    "do_elastic": False,
    "elastic_deform_alpha": (0., 900.),
    "elastic_deform_sigma": (9., 13.),
    "p_eldef": 0.2,

    "do_scaling": True,
    "scale_range": (0.85, 1.25),
    "independent_scale_factor_for_each_axis": False,
    "p_independent_scale_per_axis": 1,
    "p_scale": 0.2,

    "do_rotation": True,
    "rotation_x": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_y": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_z": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_p_per_axis": 1,
    "p_rot": 0.2,

    "random_crop": False,
    "random_crop_dist_to_border": None,

    "do_gamma": False,
    "gamma_retain_stats": True,
    "gamma_range": (0.7, 1.5),
    "p_gamma": 0.3,

    "do_mirror": True,
    "mirror_axes": (0, 1, 2),

    "dummy_2D": False,
    "mask_was_used_for_normalization": None,
    "border_mode_data": "constant",

    "all_segmentation_labels": None,  # used for cascade
    "move_last_seg_chanel_to_data": False,  # used for cascade
    "cascade_do_cascade_augmentations": False,  # used for cascade
    "cascade_random_binary_transform_p": 0.4,
    "cascade_random_binary_transform_p_per_label": 1,
    "cascade_random_binary_transform_size": (1, 8),
    "cascade_remove_conn_comp_p": 0.2,
    "cascade_remove_conn_comp_max_size_percent_threshold": 0.15,
    "cascade_remove_conn_comp_fill_with_other_class_p": 0.0,

    "do_additive_brightness": False,
    "additive_brightness_p_per_sample": 0.15,
    "additive_brightness_p_per_channel": 0.5,
    "additive_brightness_mu": 0.0,
    "additive_brightness_sigma": 0.1,

    "num_threads": 12 if 'nnUNet_n_proc_DA' not in os.environ else int(os.environ['nnUNet_n_proc_DA']),
    "num_cached_per_thread": 1,
}

default_2D_augmentation_params = deepcopy(default_3D_augmentation_params)

default_2D_augmentation_params["elastic_deform_alpha"] = (0., 200.)
default_2D_augmentation_params["elastic_deform_sigma"] = (9., 13.)
default_2D_augmentation_params["rotation_x"] = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
default_2D_augmentation_params["rotation_y"] = (-0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi)
default_2D_augmentation_params["rotation_z"] = (-0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi)

# sometimes you have 3d data and a 3d net but cannot augment them properly in 3d due to anisotropy (which is currently
# not supported in batchgenerators). In that case you can 'cheat' and transfer your 3d data into 2d data and
# transform them back after augmentation
default_2D_augmentation_params["dummy_2D"] = False
default_2D_augmentation_params["mirror_axes"] = (0, 1)  # this can be (0, 1, 2) if dummy_2D=True


def get_patch_size(final_patch_size, rot_x, rot_y, rot_z, scale_range):
    if isinstance(rot_x, (tuple, list)):
        rot_x = max(np.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(np.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(np.abs(rot_z))
    rot_x = min(90 / 360 * 2. * np.pi, rot_x)
    rot_y = min(90 / 360 * 2. * np.pi, rot_y)
    rot_z = min(90 / 360 * 2. * np.pi, rot_z)
    from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d
    coords = np.array(final_patch_size)
    final_shape = np.copy(coords)
    if len(coords) == 3:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0)
    elif len(coords) == 2:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0)
    final_shape /= min(scale_range)
    return final_shape.astype(int)
