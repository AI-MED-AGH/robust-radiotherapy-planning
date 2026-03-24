import json
import numpy as np
import nibabel as nib
from skimage.transform import resize
import os

fname = '../data_dict.json'
with open(fname,'r') as f:
    data_dict = json.load(f)

dataDict = {
"name": "ABDOMEN",
"description": "ABDOMEN",
"reference": "Holly Cross",
"licence":"NA",
"relase":"NA",
"tensorImageSize": "3D",
"modality": {
   "0": "CT",
   "1": "MRI",
   "2": "MRI",
   "3": "MRI"
 },
 "labels": {
   "0": "background",
   "1": "body"
 },
 "numTraining": 0,
 "numTest": 0,
 "training":[],
 "test": []
}


save_transforms_dir = '/net/tscratch/people/plgztabor/ROBUST_PLANNING/CODE/SIMPLE_ITK_REGISTRATION/TRANSFORMS/'

train_items = data_dict['0']['train'] +  data_dict['0']['val']
test_items = data_dict['test']

DSHAPE = (256,256)
FACTOR = 2

image_dirs = ['./imagesTr/','./imagesTs/']
label_dirs = ['./labelsTr/','./labelsTs/']
moving_image_dirs = ['./moving_imagesTr/','./moving_imagesTs/']

for n,all_items in enumerate([train_items,test_items]):

    for nitem, item in enumerate(all_items):
        fixedImName = item['fixed_image']
        movingImName = item['moving_image']

        patient_id = os.path.basename(fixedImName).split('_')[1]
        fixed_id = os.path.basename(fixedImName).split('_')[3]
        moving_id = os.path.basename(movingImName).split('_')[3]

        transformImageName = save_transforms_dir + f'transform_patient_{patient_id}_fixed_{fixed_id}_moving_{moving_id}.nii.gz'
        fname = fixedImName.replace('plgztabor/DATA','plgztabor/ROBUST_PLANNING/DATA')
        mname = movingImName.replace('plgztabor/DATA','plgztabor/ROBUST_PLANNING/DATA')

        if not os.path.isfile(transformImageName):
            print(f'no tranform file {transformImageName}')
            continue

        if not os.path.isfile(fname):
            print(f'no fixed file {fname}')
            continue

        if not os.path.isfile(mname):
            print(f'no moving file {mname}')
            continue

        fixed_img = nib.load(fname).get_fdata()
        moving_img = nib.load(mname).get_fdata()
        aff = nib.load(fname).affine
        transform = nib.load(transformImageName).get_fdata().squeeze()

        resized_fixed_img = resize(fixed_img,DSHAPE + (fixed_img.shape[2],),anti_aliasing=True,preserve_range=True)
        resized_moving_img = resize(moving_img,DSHAPE + (moving_img.shape[2],),anti_aliasing=True,preserve_range=True)
        resized_transform = resize(transform,DSHAPE + (transform.shape[2],transform.shape[3]),anti_aliasing=True,preserve_range=True)
        aff[0,0] *= FACTOR
        aff[1,1] *= FACTOR

        print(n,nitem,len(all_items),transform.shape,resized_transform.shape)

        label = np.zeros(resized_fixed_img.shape,dtype=np.float32)
        label[10:-10,10:-10,10:-10] = 1
        
        if n==0:
            fname = f'image_{nitem}.nii.gz'
            dataItem = {
                "image": "./imagesTr/" + fname,
                "label": "./labelsTr/" + fname
                }
            dataDict['training'].append(dataItem)

        fname = f'image_{nitem}.nii.gz'
        niftiImage = nib.Nifti1Image(label, affine=aff)
        nib.save(niftiImage,label_dirs[n] + fname)

        fname = f'image_{nitem}_0000.nii.gz'
        niftiImage = nib.Nifti1Image(resized_fixed_img, affine=aff)
        nib.save(niftiImage,image_dirs[n] + fname)

        fname = f'moving_image_{nitem}_0000.nii.gz'
        niftiImage = nib.Nifti1Image(resized_moving_img, affine=aff)
        nib.save(niftiImage,moving_image_dirs[n] + fname)

        fname = f'image_{nitem}_0001.nii.gz'
        niftiImage = nib.Nifti1Image(resized_transform[:,:,:,0], affine=aff)
        nib.save(niftiImage,image_dirs[n] + fname)

        fname = f'image_{nitem}_0002.nii.gz'
        niftiImage = nib.Nifti1Image(resized_transform[:,:,:,1], affine=aff)
        nib.save(niftiImage,image_dirs[n] + fname)

        fname = f'image_{nitem}_0003.nii.gz'
        niftiImage = nib.Nifti1Image(resized_transform[:,:,:,2], affine=aff)
        nib.save(niftiImage,image_dirs[n] + fname)


    if n==0:
        dataDict['numTraining'] = len(dataDict['training'])
        with open('dataset.json','w') as f:
            json.dump(dataDict,f,indent=4)





