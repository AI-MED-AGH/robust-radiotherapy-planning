import json
import os

import nibabel as nib
from skimage.transform import resize

# Config
DATA_FILE = "src/data_full/data_dict.json"
DATA_DIR = "src/data_full/CT"

SAVE_DIR = "RESULTS/DEFORMATIONS/APPLY_TRANSFORMS/moving_images/"
DOSE_FILE = "dose_stats.txt"

DSHAPE = (256, 256)
FACTOR = 2

with open(DATA_FILE, "r") as f:
    data_dict = json.load(f)["test"]

ids = sorted(
    set([os.path.basename(item["moving_image"]).split("_")[1] for item in data_dict])
)

for n, pid in enumerate(ids):
    print(pid)

    DATA_FILE = f"{DATA_DIR}/Patient_{pid}/Patient_{pid}_fraction_1_.nii.gz"

    fixed_img = nib.load(DATA_FILE).get_fdata()  # type: ignore
    aff = nib.load(DATA_FILE).affine  # type: ignore

    resized_fixed_img = resize(
        fixed_img,
        DSHAPE + (fixed_img.shape[2],),
        anti_aliasing=True,
        preserve_range=True,
    )
    aff[0, 0] *= FACTOR
    aff[1, 1] *= FACTOR

    niftiImage = nib.Nifti1Image(resized_fixed_img, affine=aff)

    DATA_FILE = f"{SAVE_DIR}/image_{pid}_0000.nii.gz"
    nib.save(niftiImage, DATA_FILE)

    DATA_FILE = f"{SAVE_DIR}/image_{pid}_0001.nii.gz"
    nib.save(niftiImage, DATA_FILE)

    DATA_FILE = f"{SAVE_DIR}/image_{pid}_0002.nii.gz"
    nib.save(niftiImage, DATA_FILE)

    DATA_FILE = f"{SAVE_DIR}/image_{pid}_0003.nii.gz"
    nib.save(niftiImage, DATA_FILE)
