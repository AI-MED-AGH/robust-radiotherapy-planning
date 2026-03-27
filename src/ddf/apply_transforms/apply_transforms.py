import json
import numpy as np
import nibabel as nib
from skimage.transform import resize
import os
import SimpleITK as sitk
import glob

fname = '/net/tscratch/people/plgztabor/ROBUST_PLANNING/CODE/data_dict.json'
with open(fname,'r') as f:
    data_dict = json.load(f)['test']

ids = sorted(set([os.path.basename(item["moving_image"]).split('_')[1] for item in data_dict]))

data_dir = '/net/tscratch/people/plgztabor/ROBUST_PLANNING/DATA/'
save_dir = '/net/tscratch/people/plgztabor/ROBUST_PLANNING/CODE/DEFORMATIONS/APPLY_TRANSFORMS/'

paths = [f'{save_dir}/predictionsTs_{i}/{j}' for i in [0,1] for j in [0,1,2,3,4]]

spacing = (1.171875, 1.171875, 3.0)
origin = (0.0, 0.0, 0.0)
direction = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
aff = np.eye(4)

for n,pid in enumerate(ids):

    print(pid)
    basename = f'Patient_{pid}_fraction_1_.nii.gz'

    os.makedirs(f'{save_dir}/CT_Ts/Patient_{pid}', exist_ok=True)
    os.makedirs(f'{save_dir}/STRUCTURES_Ts/Patient_{pid}', exist_ok=True)

    os.makedirs(f'{save_dir}/CT_Ts/Patient_{pid}/Variants', exist_ok=True)
    os.makedirs(f'{save_dir}/STRUCTURES_Ts/Patient_{pid}/Variants', exist_ok=True)
    os.makedirs(f'{save_dir}/STRUCTURES_Ts/Patient_{pid}/GT_Variants', exist_ok=True)

    fname = f'{data_dir}/CT/Patient_{pid}/{basename}'
    img = nib.load(fname).get_fdata().swapaxes(2,1).swapaxes(1,0).swapaxes(2,1)
    niftiImage = nib.Nifti1Image(img, affine=aff)
    sname = f'{save_dir}/CT_Ts/Patient_{pid}/{basename}'
    nib.save(niftiImage,sname)

    sitk_img = sitk.GetImageFromArray(img)
    sitk_img.SetOrigin(origin)
    sitk_img.SetSpacing(spacing)
    sitk_img.SetDirection(direction)    

    fname = f'{data_dir}/STRUCTURES/Patient_{pid}/{basename}'
    structure = nib.load(fname).get_fdata().swapaxes(2,1).swapaxes(1,0).swapaxes(2,1)
    labels = [1,2,3,4,5]
    sitk_structures = []
    for l in labels:
        print(f'{l=}')
        dum = np.zeros(structure.shape,dtype=np.uint8)
        if l>1:
            dum[structure==l] = 1
        else:
            dum[structure != 0] = 1
        sitk_dum = sitk.GetImageFromArray(dum)
        sitk_dum.SetOrigin(origin)
        sitk_dum.SetSpacing(spacing)
        sitk_dum.SetDirection(direction)
        sitk_structures.append(sitk_dum)
        niftiImage = nib.Nifti1Image(dum, affine=aff)
        sname = f'{save_dir}/STRUCTURES_Ts/Patient_{pid}/STRUCTURE_{l}_{basename}'
        nib.save(niftiImage,sname)

    fnames = glob.glob(f'{data_dir}/STRUCTURES/Patient_{pid}/*.nii.gz')
    for nfname,fname in enumerate(fnames):
        print(f'{nfname=}')
        structure = nib.load(fname).get_fdata().swapaxes(2,1).swapaxes(1,0).swapaxes(2,1)
        labels = [1,2,3,4,5]
        for l in labels:
            dum = np.zeros(structure.shape,dtype=np.uint8)
            if l > 0:
                dum[structure==l] = 1
            else:
                dum[structure != 0] = 1
            niftiImage = nib.Nifti1Image(dum, affine=aff)
            sname = f'{save_dir}/STRUCTURES_Ts/Patient_{pid}/GT_Variants/Variant_{nfname}_STRUCTURE_{l}_{basename}'
            nib.save(niftiImage,sname)
        

    for npath, path in enumerate(paths):
        print(f'{npath=}')
        tname = f'{path}/image_{pid}.nii.gz'
        ddf = nib.load(tname).get_fdata().squeeze().swapaxes(1,0).swapaxes(3,2)
        resized_ddf = resize(ddf, (ddf.shape[0],) + (512,521,3) ,anti_aliasing=True,preserve_range=True)
        sitk_ddf = sitk.GetImageFromArray(resized_ddf)
        sitk_ddf.SetOrigin(origin)
        sitk_ddf.SetSpacing(spacing)
        sitk_ddf.SetDirection(direction)
        dt = sitk.DisplacementFieldTransform(sitk.InvertDisplacementField(sitk_ddf))

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitk_img)  # Use fixed image as reference
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)  # Background pixel value
        resampler.SetTransform(dt)
        warped_image = resampler.Execute(sitk_img)
        warped_image = sitk.GetArrayFromImage(warped_image)
        niftiImage = nib.Nifti1Image(warped_image, affine=aff)
        sname = f'{save_dir}/CT_Ts/Patient_{pid}/Variants/Variant_{npath}_{basename}'
        nib.save(niftiImage,sname)
        for nstruct, sitk_structure in enumerate(sitk_structures):
            print(f'\t{nstruct=}')
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(sitk_structure)  # Use fixed image as reference
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)  # Background pixel value
            resampler.SetTransform(dt)
            warped_image = resampler.Execute(sitk_structure)
            warped_image = sitk.GetArrayFromImage(warped_image)
            niftiImage = nib.Nifti1Image(warped_image, affine=aff)
            sname = f'{save_dir}/STRUCTURES_Ts/Patient_{pid}/Variants/Variant_{npath}_STRUCTURE_{nstruct+1}_{basename}'
            nib.save(niftiImage,sname)
            
        
