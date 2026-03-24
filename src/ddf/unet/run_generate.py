#!/net/software/local/python/3.7/bin/python3
# Data augmentation starting point.

import json
import pickle
import numpy as np
import argparse
from augmentation.generators import prepare_generators

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', required=True, help='json file with the configuration of the data needed to perform augmentation')
args = parser.parse_args()
config_gen = args.config


with open(config_gen, 'r') as config_file:
    configs = json.load(config_file)

with open(configs['plans_file_path'], 'rb') as plans_file:
    plans = pickle.load(plans_file)

#print(plans)

possible_stages = list(plans['plans_per_stage'].keys())
stage = possible_stages[-1]

stage_plans = plans['plans_per_stage'][stage]
net_num_pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']

if configs['deep_supervision']:
    deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
                np.vstack(net_num_pool_op_kernel_sizes), axis=0))[:-1]
else:
    deep_supervision_scales = None


tr_gen, val_gen  = prepare_generators(config_gen,deep_supervision_scales)

print("generators ready")

for _ in range(20):
    data_dict = next(tr_gen)
    #data_dict = next(val_gen)

    input_channels = 1
    data = data_dict['data'].detach().cpu().numpy()[:,:input_channels]
    target = data_dict['data'].detach().cpu().numpy()[:,input_channels:]

    
    print('SHAPE',target.shape)
    #print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    #print('MINMAX',end = ' ')
    #for h in range(target.shape[0]):
    #    print( np.min(target[h]),np.max(target[h]),target.shape,end=' ')
    #print('\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

#print(possible_stages,stage)
#print(type(train_batch))

#print('Training batch, keys:')

#for key in train_batch.keys():
#    print(key,type(train_batch[key]))

#print('input shape:', train_batch['data'].shape)
#print('target shapes:')
#if type(train_batch['target']).__name__ == 'list':
#    for el in train_batch['target']:
#        print(el.shape)
#else:
#    print(train_batch['target'].shape)

#print('Validation batch, keys:')

#for key in val_batch.keys():
#    print(key,type(val_batch[key]))

#print('input shape:', val_batch['data'].shape)
#print('target shapes:')
#if type(val_batch['target']).__name__ == 'list':
#    for el in val_batch['target']:
#        print(el.shape)
#else:
#    print(val_batch['target'].shape)


#with open(configs['split_file'], 'rb') as splits_file:
#    splits = pickle.load(splits_file)

#print(splits)


