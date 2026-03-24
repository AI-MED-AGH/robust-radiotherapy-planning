import json
import pickle
import os
import numpy as np
from time import time
import argparse



########################################################################
##############    READ CONFIG FILES                    #################
########################################################################

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', required=True, help='json file with the configuration of the data needed to perform augmentation')
args = parser.parse_args()
config_train = args.config


with open(config_train, 'r') as config_file:
    configs = json.load(config_file)

with open(configs['plans_file_path'], 'rb') as plans_file:
    plans = pickle.load(plans_file)

########################################################################
##########                    PREPARE NETWORK             ##############
########################################################################

network_type = configs['network_type']
assert network_type in ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres'], "Incorrect network type!"

possible_stages = list(plans['plans_per_stage'].keys())

if (network_type == '3d_cascade_fullres' or network_type == "3d_lowres") and len(possible_stages) == 1:
    raise RuntimeError("3d_lowres/3d_cascade_fullres only applies if there is more than one stage. Run 3d_fullres.")

if network_type == '2d' or network_type == "3d_lowres":
    stage = 0
else:
    stage = possible_stages[-1]

batch_size = plans['plans_per_stage'][stage]['batch_size']
stage_plans = plans['plans_per_stage'][stage]
patch_size = np.array(stage_plans['patch_size']).astype(int)
net_num_pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']
net_conv_kernel_sizes = stage_plans['conv_kernel_sizes']

print(batch_size,patch_size,net_num_pool_op_kernel_sizes,net_conv_kernel_sizes)


plans['plans_per_stage'][stage]['patch_size'] = np.array([80,256,256]).astype(int)
#plans['plans_per_stage'][stage]['pool_op_kernel_sizes'] = plans['plans_per_stage'][stage]['pool_op_kernel_sizes'][1:]
#plans['plans_per_stage'][stage]['conv_kernel_sizes'] = plans['plans_per_stage'][stage]['conv_kernel_sizes'][1:]
#plans['plans_per_stage'][stage]['batch_size'] = 16

#plans['plans_per_stage'][stage]['pool_op_kernel_sizes'][-1] = [2,2,2]
#plans['plans_per_stage'][stage]['conv_kernel_sizes'][-1] = [3,3,3]


#batch_size = plans['plans_per_stage'][stage]['batch_size']
#stage_plans = plans['plans_per_stage'][stage]
#patch_size = np.array(stage_plans['patch_size']).astype(int)
#net_num_pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']
#net_conv_kernel_sizes = stage_plans['conv_kernel_sizes']

#print(batch_size,patch_size,net_num_pool_op_kernel_sizes,net_conv_kernel_sizes)

dirName = os.path.dirname(configs['plans_file_path'])
baseName = os.path.basename(configs['plans_file_path'])
plans_fname = dirName + '/modified_' + baseName

with open(plans_fname, 'wb') as f:
    pickle.dump(plans, f)

