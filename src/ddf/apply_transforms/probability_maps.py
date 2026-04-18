from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np


def create_structure_probability_maps(
    data_file: Path,
    save_dir: Path,
    aff: np.ndarray | None = None,
    labels: list[int] | None = None,
) -> None:
    """
    Create mean probability maps for ground-truth and predicted structure variants for each
    patient in the test split, then save them as NIfTI files.

    Assumptions:
    - `data_file` points to a valid JSON file containing a `"test"` split.
    - Test items contain `"moving_image"` paths following the pattern `Patient_<id>_fraction_<id>_.nii.gz`.
    - Ground-truth structure variants are stored in
      `save_dir / Patient_<id> / GT_Variants / Variant_*_STRUCTURE_<sid>_Patient_*.nii.gz`.
    - Predicted structure variants are stored in
      `save_dir / Patient_<id> / Variants / Variant_*_STRUCTURE_<sid>_Patient_*.nii.gz`.
    - All variant masks for a given patient and label have identical shapes.
    - The output directory is writable.

    Returns
    -------
        None:
            The function saves mean ground-truth and predicted probability maps to
            `save_dir / Patient_<id> / PROBABILIY_MAPS /`.

    Parameters
    ----------
        data_file : Path
            Path to the JSON file containing the test split definition.
        save_dir : Path
            Root directory containing per-patient structure variants.
        aff : np.ndarray | None
            Affine matrix used when saving NIfTI files with nibabel. If `None`, identity is used.
        labels : list[int] | None
            Structure labels for which probability maps will be created. If `None`,
            defaults to `[1, 2, 3, 4, 5]`.

    Raises
    ------
        FileNotFoundError
            If the input JSON file or save directory does not exist.
        ValueError
            If the JSON structure is invalid or required fields are missing.
    """
    if not data_file.is_file():
        raise FileNotFoundError(f"Data split file does not exist: {data_file}")

    if not save_dir.is_dir():
        raise FileNotFoundError(f"Save directory does not exist: {save_dir}")

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

    ids = sorted({Path(item["moving_image"]).name.split("_")[1] for item in data_dict["test"]})

    failed_patients: list[tuple[str, str]] = []

    for pid in ids:
        try:
            print(f"Processing patient {pid}")

            patient_dir = save_dir / f"Patient_{pid}"
            gt_variants_dir = patient_dir / "GT_Variants"
            pred_variants_dir = patient_dir / "Variants"
            probability_maps_dir = patient_dir / "PROBABILIY_MAPS"

            if not patient_dir.is_dir():
                raise FileNotFoundError(f"Patient directory does not exist: {patient_dir}")
            if not gt_variants_dir.is_dir():
                raise FileNotFoundError(f"GT variants directory does not exist: {gt_variants_dir}")
            if not pred_variants_dir.is_dir():
                raise FileNotFoundError(f"Predicted variants directory does not exist: {pred_variants_dir}")

            probability_maps_dir.mkdir(parents=True, exist_ok=True)

            for sid in labels:
                gt_variant_paths = sorted(gt_variants_dir.glob(f"Variant_*_STRUCTURE_{sid}_Patient_*.nii.gz"))
                if not gt_variant_paths:
                    raise FileNotFoundError(
                        f"No GT variant files found for patient {pid}, structure {sid} in: {gt_variants_dir}"
                    )

                gt_imgs = [nib.load(str(path)).get_fdata() for path in gt_variant_paths]  # type: ignore
                gt_shapes = {img.shape for img in gt_imgs}
                if len(gt_shapes) != 1:
                    raise ValueError(
                        f"GT variant files for patient {pid}, structure {sid} do not all share the same shape."
                    )

                gt_mean = np.mean(gt_imgs, axis=0)
                gt_nifti = nib.Nifti1Image(gt_mean, affine=aff)  # type: ignore
                gt_save_path = probability_maps_dir / f"GT_Patient_{pid}_STRUCTURE_{sid}.nii.gz"
                nib.save(gt_nifti, str(gt_save_path))

                pred_variant_paths = sorted(pred_variants_dir.glob(f"Variant_*_STRUCTURE_{sid}_Patient_*.nii.gz"))
                if not pred_variant_paths:
                    raise FileNotFoundError(
                        f"No predicted variant files found for patient {pid}, structure {sid} in: {pred_variants_dir}"
                    )

                pred_imgs = [nib.load(str(path)).get_fdata() for path in pred_variant_paths]  # type: ignore
                pred_shapes = {img.shape for img in pred_imgs}
                if len(pred_shapes) != 1:
                    raise ValueError(
                        f"Predicted variant files for patient {pid}, structure {sid} do not all share the same shape."
                    )

                pred_mean = np.mean(pred_imgs, axis=0)
                pred_nifti = nib.Nifti1Image(pred_mean, affine=aff)  # type: ignore
                pred_save_path = probability_maps_dir / f"PRED_Patient_{pid}_STRUCTURE_{sid}.nii.gz"
                nib.save(pred_nifti, str(pred_save_path))

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
