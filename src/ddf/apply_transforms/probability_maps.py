import glob
import json
import os

import nibabel as nib
import numpy as np

# Config
DATA_FILE = "src/data_full/data_dict.json"
DATA_DIR = "src/data_full/CT"

SAVE_DIR = "RESULTS/DEFORMATIONS/APPLY_TRANSFORMS/STRUCTURES_Ts/"
DOSE_FILE = "dose_stats.txt"

AFF = np.eye(4)
LABELS = [1, 2, 3, 4, 5]

with open(DATA_FILE, "r") as f:
    data_dict = json.load(f)["test"]

ids = sorted(
    set([os.path.basename(item["moving_image"]).split("_")[1] for item in data_dict])
)

for pid in ids:
    os.makedirs(f"{SAVE_DIR}/Patient_{pid}/PROBABILIY_MAPS", exist_ok=True)

    for sid in LABELS:
        fnames = glob.glob(
            f"{SAVE_DIR}/Patient_{pid}/GT_Variants/Variant_*_STRUCTURE_{sid}_Patient_*.nii.gz"
        )
        imgs = []
        for fname in fnames:
            imgs.append(nib.load(fname).get_fdata())  # type: ignore
        imgs_mean = np.mean(imgs, axis=0)

        niftiImage = nib.Nifti1Image(imgs_mean, affine=AFF)
        sname = f"{SAVE_DIR}/Patient_{pid}/PROBABILIY_MAPS/GT_Patient_{pid}_STRUCTURE_{sid}.nii.gz"
        nib.save(niftiImage, sname)

        del imgs

        fnames = glob.glob(
            f"{SAVE_DIR}/Patient_{pid}/Variants/Variant_*_STRUCTURE_{sid}_Patient_*.nii.gz"
        )
        imgs = []
        for fname in fnames:
            imgs.append(nib.load(fname).get_fdata())  # type: ignore
        imgs_mean = np.mean(imgs, axis=0)

        niftiImage = nib.Nifti1Image(imgs_mean, affine=AFF)
        sname = f"{SAVE_DIR}/Patient_{pid}/PROBABILIY_MAPS/PRED_Patient_{pid}_STRUCTURE_{sid}.nii.gz"
        nib.save(niftiImage, sname)

        del imgs
