import json
import os

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize

# Config
DATA_FILE = "DATA/data_dict.json"
DATA_DIR = "DATA/DOSES"

SAVE_DIR = "RESULTS/DEFORMATIONS/APPLY_TRANSFORMS"
DOSE_FILE = "dose_stats.txt"

SPACING = (1.171875, 1.171875, 3.0)
ORIGIN = (0.0, 0.0, 0.0)
DIRECTION = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
AFF = np.eye(4)
TH_LOW = -700
STRUCTURED_IDS = [2, 3, 4, 5]

with open(DATA_FILE, "r") as f:
    data_dict = json.load(f)["test"]

ids = sorted(
    set([os.path.basename(item["moving_image"]).split("_")[1] for item in data_dict])
)
# This line is hacky, but I can't figure out right now what it means
flip_flags = [1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0]

paths = [f"{SAVE_DIR}/predictionsTs_{i}/{j}" for i in [0, 1] for j in [0, 1, 2, 3, 4]]

os.makedirs(f"{SAVE_DIR}/Doses", exist_ok=True)
mapping = {0: "rectum", 1: "bladder", 2: "prostate", 3: "femur heads"}

for n, (pid, flip) in enumerate(zip(ids, flip_flags)):
    # if pid != '03':
    #    continue

    os.makedirs(f"{SAVE_DIR}/Doses/Patient_{pid}", exist_ok=True)

    print(pid)
    basename = f"Patient_{pid}_fraction_1_.nii.gz"

    fname = f"{SAVE_DIR}/CT_Ts/Patient_{pid}/{basename}"
    ct = nib.load(fname).get_fdata().swapaxes(2, 1).swapaxes(1, 0).swapaxes(2, 1)  # type: ignore
    ct[ct < TH_LOW] = TH_LOW

    fname = f"{DATA_DIR}/Patient_{pid}/{basename}"
    dose = nib.load(fname).get_fdata()  # type: ignore
    if flip:
        dose = dose[:, :, ::-1]

    fnames = [
        f"{SAVE_DIR}/STRUCTURES_Ts/Patient_{pid}/STRUCTURE_{i}_{basename}"
        for i in STRUCTURED_IDS
    ]
    structures = [
        nib.load(fname).get_fdata().swapaxes(2, 1).swapaxes(1, 0).swapaxes(2, 1)  # type: ignore
        for fname in fnames
    ]

    handle = open(DOSE_FILE, "a")
    print("%" * 20, file=handle)
    print(f"Patient_{pid}", file=handle)
    print("\tPlanned doses means", file=handle)
    for i in range(len(mapping)):
        print("\t", mapping[i], np.mean(dose[structures[i] == 1]), file=handle)
    handle.close()

    dose_variants = []
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
        dt = sitk.DisplacementFieldTransform(sitk_ddf)

        fname = f"{SAVE_DIR}/CT_Ts/Patient_{pid}/Variants/Variant_{npath}_{basename}"
        ct_variant = (
            nib.load(fname).get_fdata().swapaxes(2, 1).swapaxes(1, 0).swapaxes(2, 1)  # type: ignore
        )
        ct_variant[ct_variant < TH_LOW] = TH_LOW
        dose_corrected = dose * (1 + ct_variant / 1000) / (1 + ct / 1000)

        sitk_dose = sitk.GetImageFromArray(dose_corrected)
        sitk_dose.SetOrigin(ORIGIN)
        sitk_dose.SetSpacing(SPACING)
        sitk_dose.SetDirection(DIRECTION)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitk_dose)  # Use fixed image as reference
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)  # Background pixel value
        resampler.SetTransform(dt)
        warped_dose = resampler.Execute(sitk_dose)

        warped_dose = sitk.GetArrayFromImage(warped_dose)
        dose_variants.append(warped_dose)

    dose_variants_array = np.asarray(dose_variants, dtype=np.float32)
    dose_mean = np.mean(dose_variants_array, axis=0)
    dose_std = np.std(dose_variants_array, axis=0)

    handle = open(DOSE_FILE, "a")
    print("\tDose means from anatomical variants", file=handle)

    for structure, str_id in zip(structures, STRUCTURED_IDS):
        dose_copy_mean = np.copy(dose_mean)
        dose_copy_mean[structure == 0] = 0
        niftiImage = nib.Nifti1Image(dose_copy_mean, affine=AFF)
        sname = f"{SAVE_DIR}/Doses/Patient_{pid}/MeanDose_STRUCTURE_{str_id}_{basename}"
        nib.save(niftiImage, sname)

        dose_copy_std = np.copy(dose_std)
        dose_copy_std[structure == 0] = 0
        niftiImage = nib.Nifti1Image(dose_copy_std, affine=AFF)
        sname = f"{SAVE_DIR}/Doses/Patient_{pid}/StdDose_STRUCTURE_{str_id}_{basename}"
        nib.save(niftiImage, sname)

        print(
            "\t",
            mapping[str_id - 2],
            np.mean(dose_copy_mean[dose_copy_mean != 0]),
            file=handle,
        )

    handle.close()
