import numpy as np
import nibabel as nib
import os
import glob
import json

def iou(x,y):
    dum = x+y
    dum[dum>0] = 1
    bar = np.copy(x)
    bar[bar>0] = 1
    foo = np.copy(y)
    foo[foo>0] = 1
    return np.sum(foo*bar)/np.sum(dum)

fname = '/net/tscratch/people/plgztabor/ROBUST_PLANNING/CODE/data_dict.json'
with open(fname,'r') as f:
    data_dict = json.load(f)['test']

ids = sorted(set([os.path.basename(item["moving_image"]).split('_')[1] for item in data_dict]))

save_dir = '/net/tscratch/people/plgztabor/ROBUST_PLANNING/CODE/DEFORMATIONS/APPLY_TRANSFORMS/STRUCTURES_Ts/'

dices = {2:[], 3:[], 4:[], 5:[]}
adices = {2:[], 3:[], 4:[], 5:[]}
geds = {2:[], 3:[], 4:[], 5:[]}

THS = [0.1 + i*0.1 for i in range(10)]

for sid in [2,3,4,5]:
    print(f'{sid=}')
    for pid in ids:

        # READ PROBABILITY MAPS
        sname = f'{save_dir}/Patient_{pid}/PROBABILIY_MAPS/GT_Patient_{pid}_STRUCTURE_{sid}.nii.gz'
        gt = nib.load(sname).get_fdata()
        sname = f'{save_dir}/Patient_{pid}/PROBABILIY_MAPS/PRED_Patient_{pid}_STRUCTURE_{sid}.nii.gz'
        pred = nib.load(sname).get_fdata()

        # CALCULATE GRAY=LEVEL DICE
        dice = 2*np.sum(np.sqrt(gt*pred))/(np.sum(gt) + np.sum(pred))
        dices[sid].append(float(dice))

        # CALCULATE ADICE
        adice = []
        for TH in THS:
            th_gt = np.zeros(gt.shape, dtype=np.float32)
            th_gt[gt>=TH] = 1
            th_pred = np.zeros(pred.shape, dtype=np.float32)
            th_pred[pred>=TH] = 1
            a = 2*np.sum(th_gt*th_pred)/(np.sum(th_gt) + np.sum(th_pred))
            adice.append(a)
        adice = np.mean(adice)
        adices[sid].append(float(adice))

        # CALCULATE GED
        fnames = glob.glob(f'{save_dir}/Patient_{pid}/GT_Variants/Variant_*_STRUCTURE_{sid}_Patient_*.nii.gz')
        gt_imgs = [nib.load(fname).get_fdata() for fname in fnames]

        fnames = glob.glob(f'{save_dir}/Patient_{pid}/Variants/Variant_*_STRUCTURE_{sid}_Patient_*.nii.gz')
        pred_imgs = [nib.load(fname).get_fdata() for fname in fnames]
    
        sum1 = 0
        for i in range(len(gt_imgs)):
            for j in range(len(pred_imgs)):
                sum1 += 1-iou(gt_imgs[i],pred_imgs[j])
        sum1 /= (len(gt_imgs)*len(pred_imgs))

        sum2 = 0
        for i in range(len(gt_imgs)-1):
            for j in range(i,len(gt_imgs)):
                sum2 += 1-iou(gt_imgs[i],gt_imgs[j])
        sum2 /= (len(gt_imgs)*(len(gt_imgs)-1)/2)

        sum3 = 0
        for i in range(len(pred_imgs)-1):
            for j in range(i,len(pred_imgs)):
                sum3 += 1-iou(pred_imgs[i],pred_imgs[j])
        sum3 /= (len(pred_imgs)*(len(pred_imgs)-1)/2)

        ged = 2*sum1 - sum2 - sum3
        geds[sid].append(float(ged))

        # SAVE RESULTS
        with open('results.json','w') as f:
            json.dump({'ids':ids, 'dices':dices, 'adices':adices, 'geds': geds}, f, indent = 4)
            
        print(f'\t{pid=}, {dice=}, {adice=}, {ged=}')

    print(sid, np.mean(dices[sid]), np.std(dices[sid]),np.mean(adices[sid]),np.std(adices[sid]),np.mean(geds[sid]), np.std(geds[sid]))

for key in dices.keys():
    print(key, np.mean(dices[key]), np.std(dices[key]))
        
for key in adices.keys():
    print(key, np.mean(adices[key]), np.std(adices[key]))

for key in geds.keys():
    print(key, np.mean(geds[key]), np.std(geds[key]))

with open('results.json','w') as f:
    json.dump({'ids':ids, 'dices':dices, 'adices':adices, 'geds': geds}, f, indent = 4)



