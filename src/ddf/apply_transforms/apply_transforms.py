from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize


def apply_predicted_transforms(
    data_file: Path,
    data_dir: Path,
    save_dir: Path,
    spacing: tuple[float, float, float] = (1.171875, 1.171875, 3.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    direction: tuple[float, ...] = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0),
    aff: np.ndarray | None = None,
    labels: list[int] | None = None,
) -> None:
    """
    Apply predicted displacement fields to test CT images and structure masks, then save
    the warped image variants together with the corresponding reference and ground-truth files.

    Assumptions:
    - `data_file` points to a valid JSON file containing a `"test"` split.
    - Test items contain `"moving_image"` paths following the pattern `Patient_<id>_fraction_<id>_.nii.gz`.
    - CT images are stored in `data_dir / "CT" / Patient_<id> / ...`.
    - Structure masks are stored in `data_dir / "STRUCTURES" / Patient_<id> / ...`.
    - Predicted displacement fields are stored under `save_dir / "predictionsTs_i/j" / image_<pid>.nii.gz`.
    - CT images, masks, and displacement fields are spatially compatible after resizing.
    - Label values in the structure masks correspond to the values listed in `labels`.

    Returns
    -------
        None:
            The function saves reference CTs, warped CT variants, binary structure masks,
            warped structure variants, and ground-truth structure variants to disk.

    Parameters
    ----------
        data_file : Path
            Path to the JSON file containing the test split definition.
        data_dir : Path
            Root directory containing the `CT` and `STRUCTURES` subdirectories.
        save_dir : Path
            Root output directory where transformed images and masks will be saved.
        spacing : tuple[float, float, float]
            Voxel spacing assigned to SimpleITK images.
        origin : tuple[float, float, float]
            Image origin assigned to SimpleITK images.
        direction : tuple[float, ...]
            Direction cosine matrix assigned to SimpleITK images.
        aff : np.ndarray | None
            Affine matrix used when saving NIfTI files with nibabel. If `None`, identity is used.
        labels : list[int] | None
            Structure labels to extract and warp as binary masks. If `None`, defaults to
            `[1, 2, 3, 4, 5]`.

    Raises
    ------
        FileNotFoundError
            If the input JSON file, required data directories, or prediction directories do not exist.
        ValueError
            If the JSON structure is invalid or required fields are missing.
    """
    if not data_file.is_file():
        raise FileNotFoundError(f"Data split file does not exist: {data_file}")

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    ct_root = data_dir / "CT"
    structures_root = data_dir / "STRUCTURES"

    if not ct_root.is_dir():
        raise FileNotFoundError(f"CT directory does not exist: {ct_root}")

    if not structures_root.is_dir():
        raise FileNotFoundError(f"STRUCTURES directory does not exist: {structures_root}")

    if aff is None:
        aff = np.eye(4)

    if labels is None:
        labels = [1, 2, 3, 4, 5]

    with open(data_file) as f:
        data_dict = json.load(f)

    if "test" not in data_dict:
        raise ValueError(f'Missing required key "test" in JSON file: {data_file}')

    if not isinstance(data_dict["test"], list):
        raise ValueError(f'"test" must be a list in JSON file: {data_file}')

    for idx, item in enumerate(data_dict["test"]):
        if "moving_image" not in item:
            raise ValueError(f'Missing "moving_image" in test item at index {idx}')

    ids = sorted({os.path.basename(item["moving_image"]).split("_")[1] for item in data_dict["test"]})

    prediction_paths = [save_dir / f"predictionsTs_{i}" / str(j) for i in [0, 1] for j in [0, 1, 2, 3, 4]]

    for prediction_path in prediction_paths:
        if not prediction_path.is_dir():
            raise FileNotFoundError(f"Prediction directory does not exist: {prediction_path}")

    failed_patients: list[tuple[str, str]] = []

    for pid in ids:
        try:
            print(f"Processing patient {pid}")
            basename = f"Patient_{pid}_fraction_1_.nii.gz"

            (save_dir / "CT_Ts" / f"Patient_{pid}").mkdir(parents=True, exist_ok=True)
            (save_dir / "STRUCTURES_Ts" / f"Patient_{pid}").mkdir(parents=True, exist_ok=True)
            (save_dir / "CT_Ts" / f"Patient_{pid}" / "Variants").mkdir(parents=True, exist_ok=True)
            (save_dir / "STRUCTURES_Ts" / f"Patient_{pid}" / "Variants").mkdir(parents=True, exist_ok=True)
            (save_dir / "STRUCTURES_Ts" / f"Patient_{pid}" / "GT_Variants").mkdir(parents=True, exist_ok=True)

            ct_file = ct_root / f"Patient_{pid}" / basename
            if not ct_file.is_file():
                raise FileNotFoundError(f"CT file does not exist for patient {pid}: {ct_file}")

            img = nib.load(str(ct_file)).get_fdata().swapaxes(2, 1).swapaxes(1, 0).swapaxes(2, 1)  # type: ignore
            nifti_image = nib.Nifti1Image(img, affine=aff)  # type: ignore
            save_name = save_dir / "CT_Ts" / f"Patient_{pid}" / basename
            nib.save(nifti_image, str(save_name))

            sitk_img = sitk.GetImageFromArray(img)
            sitk_img.SetOrigin(origin)  # type: ignore
            sitk_img.SetSpacing(spacing)  # type: ignore
            sitk_img.SetDirection(direction)  # type: ignore

            structure_file = structures_root / f"Patient_{pid}" / basename
            if not structure_file.is_file():
                raise FileNotFoundError(f"Structure file does not exist for patient {pid}: {structure_file}")

            structure = nib.load(str(structure_file)).get_fdata().swapaxes(2, 1).swapaxes(1, 0).swapaxes(2, 1)  # type: ignore

            sitk_structures = []
            for label in labels:
                print(f"{label=}")
                mask = np.zeros(structure.shape, dtype=np.uint8)
                if label > 1:
                    mask[structure == label] = 1
                else:
                    mask[structure != 0] = 1

                sitk_mask = sitk.GetImageFromArray(mask)
                sitk_mask.SetOrigin(origin)  # type: ignore
                sitk_mask.SetSpacing(spacing)  # type: ignore
                sitk_mask.SetDirection(direction)  # type: ignore
                sitk_structures.append(sitk_mask)

                nifti_image = nib.Nifti1Image(mask, affine=aff)  # type: ignore
                save_name = save_dir / "STRUCTURES_Ts" / f"Patient_{pid}" / f"STRUCTURE_{label}_{basename}"
                nib.save(nifti_image, str(save_name))

            patient_structures_dir = structures_root / f"Patient_{pid}"
            structure_variants = glob.glob(str(patient_structures_dir / "*.nii.gz"))

            if not structure_variants:
                raise FileNotFoundError(
                    f"No structure variant files found for patient {pid} in: {patient_structures_dir}"
                )

            for variant_idx, variant_file in enumerate(structure_variants):
                print(f"{variant_idx=}")
                structure = nib.load(variant_file).get_fdata().swapaxes(2, 1).swapaxes(1, 0).swapaxes(2, 1)  # type: ignore

                for label in labels:
                    mask = np.zeros(structure.shape, dtype=np.uint8)
                    if label > 0:
                        mask[structure == label] = 1
                    else:
                        mask[structure != 0] = 1

                    nifti_image = nib.Nifti1Image(mask, affine=aff)  # type: ignore
                    save_name = (
                        save_dir
                        / "STRUCTURES_Ts"
                        / f"Patient_{pid}"
                        / "GT_Variants"
                        / f"Variant_{variant_idx}_STRUCTURE_{label}_{basename}"
                    )
                    nib.save(nifti_image, str(save_name))

            for path_idx, path in enumerate(prediction_paths):
                print(f"{path_idx=}")
                transform_file = path / f"image_{pid}.nii.gz"

                if not transform_file.is_file():
                    raise FileNotFoundError(
                        f"Predicted displacement field does not exist for patient {pid}: {transform_file}"
                    )

                ddf = nib.load(str(transform_file)).get_fdata().squeeze().swapaxes(1, 0).swapaxes(3, 2)  # type: ignore

                if ddf.ndim != 4:
                    raise ValueError(
                        f"Predicted displacement field for patient {pid} has invalid shape {ddf.shape}. "
                        "Expected a 4D array after squeeze and axis swaps."
                    )

                resized_ddf = resize(
                    ddf,
                    (ddf.shape[0], 512, 521, 3),
                    anti_aliasing=True,
                    preserve_range=True,
                )  # type: ignore

                sitk_ddf = sitk.GetImageFromArray(resized_ddf)
                sitk_ddf.SetOrigin(origin)  # type: ignore
                sitk_ddf.SetSpacing(spacing)  # type: ignore
                sitk_ddf.SetDirection(direction)  # type: ignore

                transform = sitk.DisplacementFieldTransform(sitk.InvertDisplacementField(sitk_ddf))  # type: ignore

                resampler = sitk.ResampleImageFilter()  # type: ignore
                resampler.SetReferenceImage(sitk_img)  # type: ignore
                resampler.SetInterpolator(sitk.sitkLinear)  # type: ignore
                resampler.SetDefaultPixelValue(0)  # type: ignore
                resampler.SetTransform(transform)  # type: ignore

                warped_ct = resampler.Execute(sitk_img)  # type: ignore
                warped_ct = sitk.GetArrayFromImage(warped_ct)

                nifti_image = nib.Nifti1Image(warped_ct, affine=aff)  # type: ignore
                save_name = save_dir / "CT_Ts" / f"Patient_{pid}" / "Variants" / f"Variant_{path_idx}_{basename}"
                nib.save(nifti_image, str(save_name))

                for struct_idx, sitk_structure in enumerate(sitk_structures):
                    print(f"\t{struct_idx=}")
                    resampler = sitk.ResampleImageFilter()  # type: ignore
                    resampler.SetReferenceImage(sitk_structure)  # type: ignore
                    resampler.SetInterpolator(sitk.sitkLinear)  # type: ignore
                    resampler.SetDefaultPixelValue(0)  # type: ignore
                    resampler.SetTransform(transform)  # type: ignore

                    warped_structure = resampler.Execute(sitk_structure)  # type: ignore
                    warped_structure = sitk.GetArrayFromImage(warped_structure)

                    nifti_image = nib.Nifti1Image(warped_structure, affine=aff)  # type: ignore
                    save_name = (
                        save_dir
                        / "STRUCTURES_Ts"
                        / f"Patient_{pid}"
                        / "Variants"
                        / f"Variant_{path_idx}_STRUCTURE_{struct_idx + 1}_{basename}"
                    )
                    nib.save(nifti_image, str(save_name))

            print(f"Finished patient {pid}")

        except Exception as e:
            print(f"Failed patient {pid}: {e}")
            failed_patients.append((pid, str(e)))
            continue

    if failed_patients:
        print("\nFailed patients summary:")
        for pid, error_msg in failed_patients:
            print(f"- Patient {pid}: {error_msg}")
    else:
        print("\nAll patients processed successfully.")
