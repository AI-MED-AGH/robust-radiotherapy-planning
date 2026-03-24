import numpy as np
import nibabel as nib
import os
import glob
import json

fname = '/net/tscratch/people/plgztabor/ROBUST_PLANNING/CODE/data_dict.json'
with open(fname,'r') as f:
    data_dict = json.load(f)['test']

ids = sorted(set([os.path.basename(item["moving_image"]).split('_')[1] for item in data_dict]))

save_dir = '/net/tscratch/people/plgztabor/ROBUST_PLANNING/CODE/DEFORMATIONS/APPLY_TRANSFORMS/STRUCTURES_Ts/'
aff = np.eye(4)

for pid in ids:

    os.makedirs(f'{save_dir}/Patient_{pid}/PROBABILIY_MAPS', exist_ok=True)

    for sid in [1,2,3,4,5]:
        fnames = glob.glob(f'{save_dir}/Patient_{pid}/GT_Variants/Variant_*_STRUCTURE_{sid}_Patient_*.nii.gz')
        imgs = []
        for fname in fnames:
            imgs.append(nib.load(fname).get_fdata())
        imgs = np.asarray(imgs,dtype=np.float32)
        imgs = np.mean(imgs,axis=0)

        niftiImage = nib.Nifti1Image(imgs, affine=aff)
        sname = f'{save_dir}/Patient_{pid}/PROBABILIY_MAPS/GT_Patient_{pid}_STRUCTURE_{sid}.nii.gz'
        nib.save(niftiImage,sname)

        del imgs

        fnames = glob.glob(f'{save_dir}/Patient_{pid}/Variants/Variant_*_STRUCTURE_{sid}_Patient_*.nii.gz')
        imgs = []
        for fname in fnames:
            imgs.append(nib.load(fname).get_fdata())
        imgs = np.asarray(imgs,dtype=np.float32)
        imgs = np.mean(imgs,axis=0)

        niftiImage = nib.Nifti1Image(imgs, affine=aff)
        sname = f'{save_dir}/Patient_{pid}/PROBABILIY_MAPS/PRED_Patient_{pid}_STRUCTURE_{sid}.nii.gz'
        nib.save(niftiImage,sname)

        del imgs
        



