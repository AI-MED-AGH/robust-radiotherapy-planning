import glob
import json
import os

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize

# Config
DATA_FILE = "src/data_full/data_dict.json"
DATA_DIR = "src/data_full/"

SAVE_DIR = "RESULTS/DEFORMATIONS/APPLY_TRANSFORMS/"

SPACING = (1.171875, 1.171875, 3.0)
ORIGIN = (0.0, 0.0, 0.0)
DIRECTION = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
AFF = np.eye(4)

LABELS = [1, 2, 3, 4, 5]


with open(DATA_FILE) as f:
    data_dict = json.load(f)["test"]
ids = sorted(set([os.path.basename(item["moving_image"]).split("_")[1] for item in data_dict]))
paths = [f"{SAVE_DIR}/predictionsTs_{i}/{j}" for i in [0, 1] for j in [0, 1, 2, 3, 4]]

for pid in ids:
    print(pid)
    basename = f"Patient_{pid}_fraction_1_.nii.gz"

    os.makedirs(f"{SAVE_DIR}/CT_Ts/Patient_{pid}", exist_ok=True)
    os.makedirs(f"{SAVE_DIR}/STRUCTURES_Ts/Patient_{pid}", exist_ok=True)

    os.makedirs(f"{SAVE_DIR}/CT_Ts/Patient_{pid}/Variants", exist_ok=True)
    os.makedirs(f"{SAVE_DIR}/STRUCTURES_Ts/Patient_{pid}/Variants", exist_ok=True)
    os.makedirs(f"{SAVE_DIR}/STRUCTURES_Ts/Patient_{pid}/GT_Variants", exist_ok=True)

    DATA_FILE = f"{DATA_DIR}/CT/Patient_{pid}/{basename}"
    img = nib.load(DATA_FILE).get_fdata().swapaxes(2, 1).swapaxes(1, 0).swapaxes(2, 1)  # type: ignore
    niftiImage = nib.Nifti1Image(img, affine=AFF)  # type: ignore
    sname = f"{SAVE_DIR}/CT_Ts/Patient_{pid}/{basename}"
    nib.save(niftiImage, sname)

    sitk_img = sitk.GetImageFromArray(img)
    sitk_img.SetOrigin(ORIGIN)  # type: ignore
    sitk_img.SetSpacing(SPACING)  # type: ignore
    sitk_img.SetDirection(DIRECTION)  # type: ignore

    DATA_FILE = f"{DATA_DIR}/STRUCTURES/Patient_{pid}/{basename}"
    structure = nib.load(DATA_FILE).get_fdata().swapaxes(2, 1).swapaxes(1, 0).swapaxes(2, 1)  # type: ignore
    sitk_structures = []
    for label in LABELS:
        print(f"{label=}")
        dum = np.zeros(structure.shape, dtype=np.uint8)
        if label > 1:
            dum[structure == label] = 1
        else:
            dum[structure != 0] = 1
        sitk_dum = sitk.GetImageFromArray(dum)
        sitk_dum.SetOrigin(ORIGIN)  # type: ignore
        sitk_dum.SetSpacing(SPACING)  # type: ignore
        sitk_dum.SetDirection(DIRECTION)  # type: ignore
        sitk_structures.append(sitk_dum)
        niftiImage = nib.Nifti1Image(dum, affine=AFF)  # type: ignore
        sname = f"{SAVE_DIR}/STRUCTURES_Ts/Patient_{pid}/STRUCTURE_{label}_{basename}"
        nib.save(niftiImage, sname)

    fnames = glob.glob(f"{DATA_DIR}/STRUCTURES/Patient_{pid}/*.nii.gz")
    for nfname, DATA_FILE in enumerate(fnames):
        print(f"{nfname=}")
        structure = nib.load(DATA_FILE).get_fdata().swapaxes(2, 1).swapaxes(1, 0).swapaxes(2, 1)  # type: ignore
        for label in LABELS:
            dum = np.zeros(structure.shape, dtype=np.uint8)
            if label > 0:
                dum[structure == label] = 1
            else:
                dum[structure != 0] = 1
            niftiImage = nib.Nifti1Image(dum, affine=AFF)  # type: ignore
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
        )  # type: ignore
        sitk_ddf = sitk.GetImageFromArray(resized_ddf)
        sitk_ddf.SetOrigin(ORIGIN)  # type: ignore
        sitk_ddf.SetSpacing(SPACING)  # type: ignore
        sitk_ddf.SetDirection(DIRECTION)  # type: ignore
        dt = sitk.DisplacementFieldTransform(sitk.InvertDisplacementField(sitk_ddf))  # type: ignore

        resampler = sitk.ResampleImageFilter()  # type: ignore
        resampler.SetReferenceImage(sitk_img)  # type: ignore
        resampler.SetInterpolator(sitk.sitkLinear)  # type: ignore
        resampler.SetDefaultPixelValue(0)  # type: ignore
        resampler.SetTransform(dt)  # type: ignore
        warped_image = resampler.Execute(sitk_img)  # type: ignore
        warped_image = sitk.GetArrayFromImage(warped_image)
        niftiImage = nib.Nifti1Image(warped_image, affine=AFF)  # type: ignore
        sname = f"{SAVE_DIR}/CT_Ts/Patient_{pid}/Variants/Variant_{npath}_{basename}"
        nib.save(niftiImage, sname)
        for nstruct, sitk_structure in enumerate(sitk_structures):
            print(f"\t{nstruct=}")
            resampler = sitk.ResampleImageFilter()  # type: ignore
            resampler.SetReferenceImage(sitk_structure)  # type: ignore
            resampler.SetInterpolator(sitk.sitkLinear)  # type: ignore
            resampler.SetDefaultPixelValue(0)  # type: ignore
            resampler.SetTransform(dt)  # type: ignore
            warped_image = resampler.Execute(sitk_structure)  # type: ignore
            warped_image = sitk.GetArrayFromImage(warped_image)
            niftiImage = nib.Nifti1Image(warped_image, affine=AFF)  # type: ignore
            sname = (
                f"{SAVE_DIR}/STRUCTURES_Ts/Patient_{pid}/Variants/Variant_{npath}_STRUCTURE_{nstruct + 1}_{basename}"
            )
            nib.save(niftiImage, sname)
