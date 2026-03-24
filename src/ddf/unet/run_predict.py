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


import argparse
import torch
import nibabel as nib
#from pycimg import CImg
from copy import deepcopy
from typing import Tuple, Union, List
import numpy as np
import shutil
import importlib
import pkgutil
import json
import pickle

from batchgenerators.utilities.file_and_folder_operations import join, isdir
from batchgenerators.augmentations.utils import resize_segmentation
from batchgenerators.utilities.file_and_folder_operations import *

from inference.predictor import Predictor


def predict_cases(plan_file,architecture_name,checkpoints, stage, output_folder, list_of_lists, output_filenames, 
                  do_tta=True, step_size=0.5):

    assert len(list_of_lists) == len(output_filenames)

    print('#################################################')
    print("emptying cuda cache")
    torch.cuda.empty_cache()

    trainer = Predictor(plan_file, architecture_name,stage, False, True)
    trainer.process_plans(load_pickle(plan_file))

    trainer.output_folder = output_folder
    trainer.output_folder_base = output_folder
    trainer.initialize(False)

    params = [torch.load(i, map_location=torch.device('cpu'),weights_only=False) for i in checkpoints]

    #print('Trainer params')
    #print(trainer.normalization_schemes, trainer.use_mask_for_norm,trainer.transpose_forward, trainer.intensity_properties)
    #print('#####################################')

    if 'segmentation_export_params' in trainer.plans.keys():
        force_separate_z = trainer.plans['segmentation_export_params']['force_separate_z']
        interpolation_order = trainer.plans['segmentation_export_params']['interpolation_order']
        interpolation_order_z = trainer.plans['segmentation_export_params']['interpolation_order_z']
    else:
        force_separate_z = None
        interpolation_order = 1
        interpolation_order_z = 0


    print("starting prediction...")
    for list_of_list,output_filename in zip(list_of_lists, output_filenames):

        print(list_of_list)
        for l in list_of_list:
            dum = nib.load(l).get_fdata()
            #print(l,np.mean(dum),np.std(dum),np.quantile(dum,(0.025,0.975)))

        d, _, dct = trainer.preprocess_patient(list_of_list)

        #for i in range(d.shape[0]):
        #    print(i,np.mean(d[i]),np.std(d[i]),np.quantile(d[i],(0.025,0.975)))

        d = d[0:1]
        #d = d[:trainer.num_input_channels]

        #print('stats',np.mean(d),np.std(d))

        print("processing ", output_filename,d.shape,dct)
        print("predicting", output_filename)
        softmax = []
        for num,p in enumerate(params):
            trainer.load_checkpoint_ram(p, False)
            #out1,out2 = trainer.predict_preprocessed_data_return_seg_and_softmax(d, do_tta, trainer.data_aug_params[
            #    'mirror_axes'], True, step_size=step_size, use_gaussian=True, all_in_gpu=False,mixed_precision=True)

            #print('aaaa',type(out1),type(out2),out1.shape,out2.shape,np.mean(out1),np.std(out1),np.mean(out2),np.std(out2))
            #niftiImage = nib.Nifti1Image(out1, affine=np.eye(4))
            #nib.save(niftiImage,os.path.dirname(output_filename) + '/predicted1_' + str(num) + '_' + os.path.basename(output_filename))
            #niftiImage = nib.Nifti1Image(out2[0], affine=np.eye(4))
            #nib.save(niftiImage,os.path.dirname(output_filename) + '/predicted2_' + str(num) + '_' + os.path.basename(output_filename))

            softmax.append(trainer.predict_preprocessed_data_return_seg_and_softmax(d, do_tta, trainer.data_aug_params[
                'mirror_axes'], True, step_size=step_size, use_gaussian=True, all_in_gpu=False,mixed_precision=True)[1][None])

        softmax = np.vstack(softmax)
        softmax_mean = np.mean(softmax, 0)

        #print('prediction',np.mean(softmax_mean),np.std(softmax_mean),np.quantile(softmax_mean,(0.025,0.975)))

        transpose_forward = trainer.plans.get('transpose_forward')
        if transpose_forward is not None:
            transpose_backward = trainer.plans.get('transpose_backward')
            softmax_mean = softmax_mean.transpose([0] + [i + 1 for i in transpose_backward])

        #niftiImage = nib.Nifti1Image(softmax_mean[0], affine=np.eye(4))
        #nib.save(niftiImage,os.path.dirname(output_filename) + '/predicted_' + os.path.basename(output_filename))

        if hasattr(trainer, 'regions_class_order'):
            region_class_order = trainer.regions_class_order
        else:
            region_class_order = None

        #print(np.mean(softmax_mean),np.std(softmax_mean))
        trainer.save_segmentation(softmax_mean, output_filename, dct, interpolation_order, region_class_order,None, None,None, None, force_separate_z, interpolation_order_z)



def check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities):
    print("This model expects %d input modalities for each image" % expected_num_modalities)
    files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)

    maybe_case_ids = np.unique([i[:-12] for i in files])

    remaining = deepcopy(files)
    missing = []

    assert len(files) > 0, "input folder did not contain any images (expected to find .nii.gz file endings)"

    # now check if all required files are present and that no unexpected files are remaining
    for c in maybe_case_ids:
        for n in range(expected_num_modalities):
            expected_output_file = c + "_%04.0d.nii.gz" % n
            if not isfile(join(input_folder, expected_output_file)):
                missing.append(expected_output_file)
            else:
                remaining.remove(expected_output_file)

    print("Found %d unique case ids, here are some examples:" % len(maybe_case_ids),
          np.random.choice(maybe_case_ids, min(len(maybe_case_ids), 10)))
    print("If they don't look right, make sure to double check your filenames. They must end with _0000.nii.gz etc")

    if len(remaining) > 0:
        print("found %d unexpected remaining files in the folder. Here are some examples:" % len(remaining),
              np.random.choice(remaining, min(len(remaining), 10)))

    if len(missing) > 0:
        print("Some files are missing:")
        print(missing)
        raise RuntimeError("missing files in input_folder")

    return maybe_case_ids





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config",required = True, help ="json file with inference configuration")

    args = parser.parse_args()
    config_file = args.config

    with open(config_file, 'r') as config_file:
        configs = json.load(config_file)

    input_folder = configs['input_folder']
    output_folder = configs['output_folder']
    disable_tta = configs['disable_tta']
    plan_file = configs['plans_file_path']
    checkpoints = list(configs['checkpoints'].values())

    maybe_mkdir_p(output_folder)

    # check input folder integrity
    expected_num_modalities = load_pickle(plan_file)['num_modalities']

    case_ids = check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities)

    # output file names
    output_files = [join(output_folder, i + ".nii.gz") for i in case_ids]
    all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)

    # original input image names in a form of a list of lists, where j-th sublist contains n input image names for j-th case, these n inputs correspond to n Unet inputs 
    list_of_lists = [[join(input_folder, i) for i in all_files if i[:len(j)].startswith(j) and
                      len(i) == (len(j) + 12)] for j in case_ids]

    # some configs
    with open(configs['plans_file_path'], 'rb') as plans_file:
        plans = pickle.load(plans_file)

    network_type = configs['network_type']
    assert network_type in ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres'], "Incorrect network type!"

    possible_stages = list(plans['plans_per_stage'].keys())

    if (network_type == '3d_cascade_fullres' or network_type == "3d_lowres") and len(possible_stages) == 1:
        raise RuntimeError("3d_lowres/3d_cascade_fullres only applies if there is more than one stage. Run 3d_fullres.")

    if network_type == '2d' or network_type == "3d_lowres":
        stage = 0
    else:
        stage = possible_stages[-1]

    if 'architecture_name' in configs.keys():
        architecture_name = configs['architecture_name']
    else:
        architecture_name = "Generic_UNet"

    # start prediction
    step_size = 0.5
    predict_cases(plan_file,architecture_name,checkpoints, stage, output_folder, list_of_lists, output_files, not disable_tta, step_size=step_size)


if __name__ == "__main__":
    main()
