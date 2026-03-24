import json
import numpy as np
import nibabel as nib
from skimage.transform import resize
import os

fname = '/net/tscratch/people/plgztabor/ROBUST_PLANNING/CODE/data_dict.json'
with open(fname,'r') as f:
    data_dict = json.load(f)['test']

ids = set([os.path.basename(item["moving_image"]).split('_')[1] for item in data_dict])

data_dir = '/net/tscratch/people/plgztabor/ROBUST_PLANNING/DATA/CT/'
save_dir = '/net/tscratch/people/plgztabor/ROBUST_PLANNING/CODE/DEFORMATIONS/APPLY_TRANSFORMS/moving_images/'

DSHAPE = (256,256)
FACTOR = 2


for n,pid in enumerate(ids):

        print(pid)

        fname = f'{data_dir}/Patient_{pid}/Patient_{pid}_fraction_1_.nii.gz'

        fixed_img = nib.load(fname).get_fdata()
        aff = nib.load(fname).affine

        resized_fixed_img = resize(fixed_img,DSHAPE + (fixed_img.shape[2],),anti_aliasing=True,preserve_range=True)
        aff[0,0] *= FACTOR
        aff[1,1] *= FACTOR

        niftiImage = nib.Nifti1Image(resized_fixed_img, affine=aff)

        fname = f'{save_dir}/image_{pid}_0000.nii.gz'
        nib.save(niftiImage,fname)

        fname = f'{save_dir}/image_{pid}_0001.nii.gz'
        nib.save(niftiImage,fname)

        fname = f'{save_dir}/image_{pid}_0002.nii.gz'
        nib.save(niftiImage,fname)

        fname = f'{save_dir}/image_{pid}_0003.nii.gz'
        nib.save(niftiImage,fname)

