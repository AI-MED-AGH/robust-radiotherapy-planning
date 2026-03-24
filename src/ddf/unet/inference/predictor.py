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

import sys
import shutil
import numpy as np
from collections import OrderedDict
from typing import Tuple, List, Union
from _warnings import warn
from time import time, sleep
from abc import abstractmethod
from datetime import datetime
#from tqdm import trange
#from multiprocessing import Pool

from augmentation.default_data_augmentation import default_2D_augmentation_params, get_patch_size, default_3D_augmentation_params

from inference.segmentation_export import save_segmentation_nifti_from_softmax
from preprocessing.preprocessing import *

from training.generic_UNet import *
from training.utils import InitWeights_He
from training.neural_network import SegmentationNetwork
from training.utils import softmax_helper
from training.utils import maybe_to_torch, to_cuda
from training.utils import sum_tensor

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler, autocast
import torch.backends.cudnn as cudnn

from batchgenerators.utilities.file_and_folder_operations import *

import importlib


class Predictor(object):
    def __init__(self, plans_file, architecture_name,stage=None, deterministic=False, fp16=True):

        self.fp16 = fp16

        if deterministic:
            np.random.seed(12345)
            torch.manual_seed(12345)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(12345)
            cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

        self.deterministic = deterministic

        self.network = None
        self.was_initialized = False

        self.stage = stage
        self.plans_file = plans_file
        self.plans = None

        self.num_input_channels = self.num_classes = self.net_pool_per_axis = self.patch_size = self.batch_size = \
            self.threeD = self.base_num_features = self.intensity_properties = self.normalization_schemes = \
            self.net_num_pool_op_kernel_sizes = self.net_conv_kernel_sizes = None  # loaded automatically from plans_file

        self.basic_generator_patch_size = self.data_aug_params = self.transpose_forward = self.transpose_backward = None


        self.classes = self.do_dummy_2D_aug = self.use_mask_for_norm = self.only_keep_largest_connected_component = \
            self.min_region_size_per_class = self.min_size_per_class = None

        self.inference_pad_border_mode = "constant"
        self.inference_pad_kwargs = {'constant_values': 0}

        self.pad_all_sides = None

        self.conv_per_stage = None
        self.regions_class_order = None

        self.pin_memory = True

        self.architecture_name = architecture_name

    def initialize(self, training=True, force_load_plans=False):

        if not self.was_initialized:

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            self.initialize_network()

        self.was_initialized = True

    def initialize_network(self):
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0.1, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

        m = importlib.import_module('training.generic_UNet')
        architecture_class = getattr(m, self.architecture_name)
        print(architecture_class,self.num_input_channels)

############################################
        self.num_input_channels = 1
        self.num_classes = 3
        #self.num_input_channels = self.num_input_channels -1
        #self.num_classes = 1
###########################################

        self.network = architecture_class(self.num_input_channels, self.base_num_features, self.num_classes,
                                          len(self.net_num_pool_op_kernel_sizes),
                                          self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                          dropout_op_kwargs,
                                          net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                          self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = lambda x: x

    def setup_DA_params(self):

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                print("Using dummy2d data augmentation",flush=True)
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
            patch_size_for_spatialtransform = self.patch_size[1:]
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            patch_size_for_spatialtransform = self.patch_size

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform

        self.data_aug_params["num_cached_per_thread"] = 2

    def load_plans_file(self):
        """
        This is what actually configures the entire experiment. The plans file is generated by experiment planning
        :return:
        """
        self.plans = load_pickle(self.plans_file)

    def process_plans(self, plans):
        if self.stage is None:
            assert len(list(plans['plans_per_stage'].keys())) == 1, \
                "If self.stage is None then there can be only one stage in the plans file. That seems to not be the " \
                "case. Please specify which stage of the cascade must be trained"
            self.stage = list(plans['plans_per_stage'].keys())[0]
        self.plans = plans

        stage_plans = self.plans['plans_per_stage'][self.stage]
        self.batch_size = stage_plans['batch_size']
        self.net_pool_per_axis = stage_plans['num_pool_per_axis']
        self.patch_size = np.array(stage_plans['patch_size']).astype(int)
        self.do_dummy_2D_aug = stage_plans['do_dummy_2D_data_aug']

        if 'pool_op_kernel_sizes' not in stage_plans.keys():
            assert 'num_pool_per_axis' in stage_plans.keys()
            print("WARNING! old plans file with missing pool_op_kernel_sizes. Attempting to fix it...",flush=True)
            self.net_num_pool_op_kernel_sizes = []
            for i in range(max(self.net_pool_per_axis)):
                curr = []
                for j in self.net_pool_per_axis:
                    if (max(self.net_pool_per_axis) - j) <= i:
                        curr.append(2)
                    else:
                        curr.append(1)
                self.net_num_pool_op_kernel_sizes.append(curr)
        else:
            self.net_num_pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']

        if 'conv_kernel_sizes' not in stage_plans.keys():
            print("WARNING! old plans file with missing conv_kernel_sizes. Attempting to fix it...",flush=True)
            self.net_conv_kernel_sizes = [[3] * len(self.net_pool_per_axis)] * (max(self.net_pool_per_axis) + 1)
        else:
            self.net_conv_kernel_sizes = stage_plans['conv_kernel_sizes']

        self.pad_all_sides = None  # self.patch_size
        self.intensity_properties = plans['dataset_properties']['intensityproperties']
        self.normalization_schemes = plans['normalization_schemes']
        self.base_num_features = plans['base_num_features']
        self.num_input_channels = plans['num_modalities']
        self.num_classes = plans['num_classes'] + 1  # background is no longer in num_classes
        self.classes = plans['all_classes']
        self.use_mask_for_norm = plans['use_mask_for_norm']
        self.only_keep_largest_connected_component = plans['keep_only_largest_region']
        self.min_region_size_per_class = plans['min_region_size_per_class']
        self.min_size_per_class = None  # DONT USE THIS. plans['min_size_per_class']

        if plans.get('transpose_forward') is None or plans.get('transpose_backward') is None:
            print("WARNING! You seem to have data that was preprocessed with a previous version of nnU-Net. "
                  "You should rerun preprocessing. We will proceed and assume that both transpose_foward "
                  "and transpose_backward are [0, 1, 2]. If that is not correct then weird things will happen!")
            plans['transpose_forward'] = [0, 1, 2]
            plans['transpose_backward'] = [0, 1, 2]
        self.transpose_forward = plans['transpose_forward']
        self.transpose_backward = plans['transpose_backward']

        if len(self.patch_size) == 2:
            self.threeD = False
        elif len(self.patch_size) == 3:
            self.threeD = True
        else:
            raise RuntimeError("invalid patch size in plans file: %s" % str(self.patch_size))

        if "conv_per_stage" in plans.keys():  # this ha sbeen added to the plans only recently
            self.conv_per_stage = plans['conv_per_stage']
        else:
            self.conv_per_stage = 2


    def preprocess_patient(self, input_files):

        #preprocessor_classes = {"GenericPreprocessor":GenericPreprocessor,"PreprocessorFor2D":PreprocessorFor2D,
        #                       "Preprocessor3DDifferentResampling":Preprocessor3DDifferentResampling,
        #                       "Preprocessor3DBetterResampling":Preprocessor3DBetterResampling,
        #                       "PreprocessorFor2D_noNormalization":PreprocessorFor2D_noNormalization}

        preprocessor_name = self.plans.get('preprocessor_name')
        if preprocessor_name is None:
            if self.threeD:
                preprocessor_name = "GenericPreprocessor"
            else:
                preprocessor_name = "PreprocessorFor2D"

        print("using preprocessor", preprocessor_name)

        m = importlib.import_module('preprocessing.preprocessing')
        preprocessor_class = getattr(m, preprocessor_name)

        #preprocessor_class = preprocessor_classes[preprocessor_name]
        preprocessor = preprocessor_class(self.normalization_schemes, self.use_mask_for_norm,
                                          self.transpose_forward, self.intensity_properties)

        d, s, properties = preprocessor.preprocess_test_case(input_files,
                                                             self.plans['plans_per_stage'][self.stage][
                                                                 'current_spacing'])
        return d, s, properties


    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = True,
                                                         verbose: bool = True, mixed_precision: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param data:
        :param do_mirroring:
        :param mirror_axes:
        :param use_sliding_window:
        :param step_size:
        :param use_gaussian:
        :param pad_border_mode:
        :param pad_kwargs:
        :param all_in_gpu:
        :param verbose:
        :return:
        """

        ds = self.network.do_ds
        self.network.do_ds = False

        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        if do_mirroring and mirror_axes is None:
            mirror_axes = self.data_aug_params['mirror_axes']

        if do_mirroring:
            assert self.data_aug_params["do_mirror"], "Cannot do mirroring as test time augmentation when training " \
                                                      "was done without mirroring"


        current_mode = self.network.training
        #self.network.eval()
        self.network.train()
        ret = self.network.predict_3D(data, do_mirroring, mirror_axes, use_sliding_window, step_size, self.patch_size,
                                      self.regions_class_order, use_gaussian, pad_border_mode, pad_kwargs,
                                      all_in_gpu, verbose, mixed_precision=mixed_precision)

        print('return from predict_preprocessed_data_return_seg_and_softmax',ret[0].shape,ret[1].shape)

        self.network.train(current_mode)
        self.network.do_ds = ds

        return ret



    def load_checkpoint_ram(self, checkpoint, train=False):

        self.network.load_state_dict(checkpoint['state_dict'])


    def save_segmentation(self,segmentation_softmax: Union[str, np.ndarray], out_fname: str,
                          properties_dict: dict, order: int = 1,
                          region_class_order: Tuple[Tuple[int]] = None,
                          seg_postprogess_fn: callable = None, seg_postprocess_args: tuple = None,
                          resampled_npz_fname: str = None,
                          non_postprocessed_fname: str = None, force_separate_z: bool = None,
                          interpolation_order_z: int = 0, verbose: bool = True):

        save_segmentation_nifti_from_softmax(segmentation_softmax, out_fname,
                                             properties_dict, order,region_class_order,seg_postprogess_fn, seg_postprocess_args,
                                             resampled_npz_fname,non_postprocessed_fname, force_separate_z,interpolation_order_z, verbose)

    
