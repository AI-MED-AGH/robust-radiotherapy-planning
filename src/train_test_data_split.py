import os
import numpy as np
import glob
import json

########################################################

data_dir = '../data/CT/'
data_files = sorted(glob.glob(f'{data_dir}*/Patient_*'))
ids = [os.path.basename(i).split('_')[1] for i in data_files]

SEED = 42
np.random.seed(SEED)
np.random.shuffle(ids)

train_fraction = 0.8
train_ids = ids[:int(len(ids)*train_fraction)]
test_ids = ids[int(len(ids)*train_fraction):]

#########################################################

val_fraction = 0.2
val_num = int(val_fraction*len(train_ids))

data_dict = {}

for fold in range(5):

    val_ids = train_ids[fold*val_num:min((fold+1)*val_num,len(train_ids))]

    train_files = []
    for i in train_ids:
        if i in val_ids:
            continue
        dname = f'{data_dir}/Patient_{i}/'
        fraction_names = [name for name in sorted(glob.glob(f'{dname}/*.nii.gz')) if '_fraction_1_.nii.gz' not in name]
        first_fraction_name = f'{data_dir}/Patient_{i}/Patient_{i}_fraction_1_.nii.gz'
        data_dirs = [{
                "moving_image": first_fraction_name,
                "fixed_image":n
                } for n in fraction_names]
        train_files.extend(data_dirs)

    val_files = []
    for i in val_ids:
        dname = f'{data_dir}/Patient_{i}/'
        fraction_names = [name for name in sorted(glob.glob(f'{dname}/*.nii.gz')) if '_fraction_1_.nii.gz' not in name]
        first_fraction_name = f'{data_dir}/Patient_{i}/Patient_{i}_fraction_1_.nii.gz'
        data_dirs = [{
                "moving_image": first_fraction_name,
                "fixed_image":n
                } for n in fraction_names]
        val_files.extend(data_dirs)

    data_dict[fold] = {}
    data_dict[fold]['train'] = train_files
    data_dict[fold]['val'] = val_files

test_files = []
for i in test_ids:
    dname = f'{data_dir}/Patient_{i}/'
    fraction_names = [name for name in sorted(glob.glob(f'{dname}/*.nii.gz')) if '_fraction_1_.nii.gz' not in name]
    first_fraction_name = f'{data_dir}/Patient_{i}/Patient_{i}_fraction_1_.nii.gz'
    data_dirs = [{
            "moving_image": first_fraction_name,
            "fixed_image":n
            } for n in fraction_names]
    test_files.extend(data_dirs)

data_dict['test'] = test_files

with open('data_dict.json','w') as f:
    json.dump(data_dict,f,indent = 4)

