import glob
import json
import os

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from skimage.transform import resize

# Config
DATA_FILE = "DATA/data_dict.json"
DATA_DIR = "DATA/"

SAVE_DIR = "RESULTS/DEFORMATIONS/APPLY_TRANSFORMS/"

SPACING = (1.171875, 1.171875, 3.0)
ORIGIN = (0.0, 0.0, 0.0)
DIRECTION = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
AFF = np.eye(4)

LABELS = [1, 2, 3, 4, 5]


with open(DATA_FILE, "r") as f:
    data_dict = json.load(f)["test"]
ids = sorted(
    set([os.path.basename(item["moving_image"]).split("_")[1] for item in data_dict])
)
paths = [f"{SAVE_DIR}/predictionsTs_{i}/{j}" for i in [0, 1] for j in [0, 1, 2, 3, 4]]

for n, pid in enumerate(ids):
    print(pid)
    basename = f"Patient_{pid}_fraction_1_.nii.gz"

    os.makedirs(f"{SAVE_DIR}/CT_Ts/Patient_{pid}", exist_ok=True)
    os.makedirs(f"{SAVE_DIR}/STRUCTURES_Ts/Patient_{pid}", exist_ok=True)

    os.makedirs(f"{SAVE_DIR}/CT_Ts/Patient_{pid}/Variants", exist_ok=True)
    os.makedirs(f"{SAVE_DIR}/STRUCTURES_Ts/Patient_{pid}/Variants", exist_ok=True)
    os.makedirs(f"{SAVE_DIR}/STRUCTURES_Ts/Patient_{pid}/GT_Variants", exist_ok=True)

    DATA_FILE = f"{DATA_DIR}/CT/Patient_{pid}/{basename}"
    img = nib.load(DATA_FILE).get_fdata().swapaxes(2, 1).swapaxes(1, 0).swapaxes(2, 1)  # type: ignore
    niftiImage = nib.Nifti1Image(img, affine=AFF)
    sname = f"{SAVE_DIR}/CT_Ts/Patient_{pid}/{basename}"
    nib.save(niftiImage, sname)

    sitk_img = sitk.GetImageFromArray(img)
    sitk_img.SetOrigin(ORIGIN)
    sitk_img.SetSpacing(SPACING)
    sitk_img.SetDirection(DIRECTION)

    DATA_FILE = f"{DATA_DIR}/STRUCTURES/Patient_{pid}/{basename}"
    structure = (
        nib.load(DATA_FILE).get_fdata().swapaxes(2, 1).swapaxes(1, 0).swapaxes(2, 1)  # type: ignore
    )
    sitk_structures = []
    for label in LABELS:
        print(f"{label=}")
        dum = np.zeros(structure.shape, dtype=np.uint8)
        if label > 1:
            dum[structure == label] = 1
        else:
            dum[structure != 0] = 1
        sitk_dum = sitk.GetImageFromArray(dum)
        sitk_dum.SetOrigin(ORIGIN)
        sitk_dum.SetSpacing(SPACING)
        sitk_dum.SetDirection(DIRECTION)
        sitk_structures.append(sitk_dum)
        niftiImage = nib.Nifti1Image(dum, affine=AFF)
        sname = f"{SAVE_DIR}/STRUCTURES_Ts/Patient_{pid}/STRUCTURE_{label}_{basename}"
        nib.save(niftiImage, sname)

    fnames = glob.glob(f"{DATA_DIR}/STRUCTURES/Patient_{pid}/*.nii.gz")
    for nfname, DATA_FILE in enumerate(fnames):
        print(f"{nfname=}")
        structure = (
            nib.load(DATA_FILE).get_fdata().swapaxes(2, 1).swapaxes(1, 0).swapaxes(2, 1)  # type: ignore
        )
        for label in LABELS:
            dum = np.zeros(structure.shape, dtype=np.uint8)
            if label > 0:
                dum[structure == label] = 1
            else:
                dum[structure != 0] = 1
            niftiImage = nib.Nifti1Image(dum, affine=AFF)
            sname = f"{SAVE_DIR}/STRUCTURES_Ts/Patient_{pid}/GT_Variants/Variant_{nfname}_STRUCTURE_{label}_{basename}"
            nib.save(niftiImage, sname)

    for npath, path in enumerate(paths):
        print(f"{npath=}")
        tname = f"{path}/image_{pid}.nii.gz"
        ddf = nib.load(tname).get_fdata().squeeze().swapaxes(1, 0).swapaxes(3, 2)  # type: ignore
        resized_ddf = resize(
            ddf,
            (ddf.shape[0],) + (512, 521, 3),
            anti_aliasing=True,
            preserve_range=True,
        )
        sitk_ddf = sitk.GetImageFromArray(resized_ddf)
        sitk_ddf.SetOrigin(ORIGIN)
        sitk_ddf.SetSpacing(SPACING)
        sitk_ddf.SetDirection(DIRECTION)
        dt = sitk.DisplacementFieldTransform(sitk.InvertDisplacementField(sitk_ddf))

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitk_img)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(dt)
        warped_image = resampler.Execute(sitk_img)
        warped_image = sitk.GetArrayFromImage(warped_image)
        niftiImage = nib.Nifti1Image(warped_image, affine=AFF)
        sname = f"{SAVE_DIR}/CT_Ts/Patient_{pid}/Variants/Variant_{npath}_{basename}"
        nib.save(niftiImage, sname)
        for nstruct, sitk_structure in enumerate(sitk_structures):
            print(f"\t{nstruct=}")
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(sitk_structure)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(dt)
            warped_image = resampler.Execute(sitk_structure)
            warped_image = sitk.GetArrayFromImage(warped_image)
            niftiImage = nib.Nifti1Image(warped_image, affine=AFF)
            sname = f"{SAVE_DIR}/STRUCTURES_Ts/Patient_{pid}/Variants/Variant_{npath}_STRUCTURE_{nstruct + 1}_{basename}"
            nib.save(niftiImage, sname)
