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
ids = ['03','17','25','27','33','42','59','68','71','76','77']
flip_flags = [1,0,1,1,0,0,1,1,1,1,0]
    
dose_dir = '/net/tscratch/people/plgztabor/ROBUST_PLANNING/DATA/DOSES/'

save_dir = '/net/tscratch/people/plgztabor/ROBUST_PLANNING/CODE/DEFORMATIONS/APPLY_TRANSFORMS/'

paths = [f'{save_dir}/predictionsTs_{i}/{j}' for i in [0,1] for j in [0,1,2,3,4]]

spacing = (1.171875, 1.171875, 3.0)
origin = (0.0, 0.0, 0.0)
direction = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
aff = np.eye(4)
TH_LOW = -700
structures_ids = [2,3,4,5]

os.makedirs(f'{save_dir}/Doses', exist_ok=True)
mapping = {0:'rectum',1:'bladder',2:'prostate',3:'femur heads'}

for n,(pid,flip) in enumerate(zip(ids,flip_flags)):

    #if pid != '03':
    #    continue
        
    os.makedirs(f'{save_dir}/Doses/Patient_{pid}', exist_ok=True)
    
    print(pid)
    basename = f'Patient_{pid}_fraction_1_.nii.gz'

    fname = f'{save_dir}/CT_Ts/Patient_{pid}/{basename}'
    ct = nib.load(fname).get_fdata().swapaxes(2,1).swapaxes(1,0).swapaxes(2,1)
    ct[ct<TH_LOW] = TH_LOW
    
    fname = f'{dose_dir}/Patient_{pid}/{basename}'
    dose = nib.load(fname).get_fdata()
    if flip:
        dose = dose[:,:,::-1]

    fnames = [f'{save_dir}/STRUCTURES_Ts/Patient_{pid}/STRUCTURE_{i}_{basename}' 
              for i in structures_ids]
    structures = [nib.load(fname).get_fdata().swapaxes(2,1).swapaxes(1,0).swapaxes(2,1) 
                  for fname in fnames]

    handle = open("dose_stats.txt","a")
    print('%'*20,file=handle)
    print( f'Patient_{pid}',file=handle)
    print('\tPlanned doses means',file=handle)
    for i in range(4):
        print('\t',mapping[i],np.mean(dose[structures[i]==1]),file=handle)    
    handle.close()

    dose_variants = []
    for npath, path in enumerate(paths):
        print(f'{npath=}')
        tname = f'{path}/image_{pid}.nii.gz'
        ddf = nib.load(tname).get_fdata().squeeze().swapaxes(1,0).swapaxes(3,2)
        resized_ddf = resize(ddf, (ddf.shape[0],) + (512,521,3) ,anti_aliasing=True,preserve_range=True)
        sitk_ddf = sitk.GetImageFromArray(resized_ddf)
        sitk_ddf.SetOrigin(origin)
        sitk_ddf.SetSpacing(spacing)
        sitk_ddf.SetDirection(direction)
        dt = sitk.DisplacementFieldTransform(sitk_ddf)
        
        fname = f'{save_dir}/CT_Ts/Patient_{pid}/Variants/Variant_{npath}_{basename}'
        ct_variant = nib.load(fname).get_fdata().swapaxes(2,1).swapaxes(1,0).swapaxes(2,1)
        ct_variant[ct_variant<TH_LOW] = TH_LOW
        dose_corrected = dose*(1 + ct_variant/1000)/(1 + ct/1000)
    
        sitk_dose = sitk.GetImageFromArray(dose_corrected)
        sitk_dose.SetOrigin(origin)
        sitk_dose.SetSpacing(spacing)
        sitk_dose.SetDirection(direction)      

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitk_dose)  # Use fixed image as reference
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)  # Background pixel value
        resampler.SetTransform(dt)
        warped_dose = resampler.Execute(sitk_dose)

        warped_dose = sitk.GetArrayFromImage(warped_dose)
        dose_variants.append(warped_dose)
        
    dose_variants = np.asarray(dose_variants,dtype=np.float32)
    dose_mean = np.mean(dose_variants,axis=0)
    dose_std = np.std(dose_variants,axis=0)

    handle = open("dose_stats.txt","a")
    print('\tDose means from anatomical variants',file=handle)

    for structure,str_id in zip(structures,structures_ids):
        dose_copy_mean = np.copy(dose_mean)
        dose_copy_mean[structure==0] = 0
        niftiImage = nib.Nifti1Image(dose_copy_mean, affine=aff)
        sname = f'{save_dir}/Doses/Patient_{pid}/MeanDose_STRUCTURE_{str_id}_{basename}'
        nib.save(niftiImage,sname)
        
        dose_copy_std = np.copy(dose_std)
        dose_copy_std[structure==0] = 0        
        niftiImage = nib.Nifti1Image(dose_copy_std, affine=aff)
        sname = f'{save_dir}/Doses/Patient_{pid}/StdDose_STRUCTURE_{str_id}_{basename}'
        nib.save(niftiImage,sname)
        
        print('\t',mapping[str_id-2],np.mean(dose_copy_mean[dose_copy_mean!=0]),file=handle)

    handle.close()
