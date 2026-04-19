from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any

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

    prediction_paths = sorted(
        path
        for parent in save_dir.glob("predictionsTs_*")
        if parent.is_dir()
        for path in parent.iterdir()
        if path.is_dir()
    )

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
                        f"Predicted displacement field for patient {pid} has invalid shape {ddf.shape}."
                        "Expected a 4D array after squeeze and axis swaps"
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


def iou(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate intersection over union (IoU) for two binary arrays of the same shape.

    Assumptions:
    - `x` and `y` are NumPy arrays of identical shape.
    - Both arrays represent binary or binarizable segmentation masks.
    - Non-zero values are treated as foreground.

    Returns
    -------
        float:
            The IoU value.

    Parameters
    ----------
        x : np.ndarray
            First binary array.
        y : np.ndarray
            Second binary array.

    Raises
    ------
        ValueError
            If the input arrays are not of the same shape.
    """
    if x.shape != y.shape:
        raise ValueError(f"Input arrays must have the same shape, got {x.shape} and {y.shape}")

    union = x + y
    union[union > 0] = 1

    x_bin = np.copy(x)
    x_bin[x_bin > 0] = 1

    y_bin = np.copy(y)
    y_bin[y_bin > 0] = 1

    union_sum = np.sum(union)
    if union_sum == 0:
        return 1.0

    return float(np.sum(x_bin * y_bin) / union_sum)


def compute_gray_level_dice(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Compute gray-level Dice score for two probability maps.

    Assumptions:
    - `gt` and `pred` are NumPy arrays of identical shape.
    - Both arrays contain non-negative values.
    - Arrays represent probability-like or soft segmentation maps.

    Returns
    -------
        float:
            The gray-level Dice score.

    Parameters
    ----------
        gt : np.ndarray
            Ground-truth probability map.
        pred : np.ndarray
            Predicted probability map.

    Raises
    ------
        ValueError
            If the input arrays are not of the same shape.
    """
    if gt.shape != pred.shape:
        raise ValueError(f"Input arrays must have the same shape, got {gt.shape} and {pred.shape}")

    denominator = np.sum(gt) + np.sum(pred)
    if denominator == 0:
        return 1.0

    return float(2 * np.sum(np.sqrt(gt * pred)) / denominator)


def compute_adice(gt: np.ndarray, pred: np.ndarray, thresholds: list[float]) -> float:
    """
    Compute averaged Dice score over a set of thresholds.

    Assumptions:
    - `gt` and `pred` are NumPy arrays of identical shape.
    - `thresholds` is non-empty.
    - Thresholded arrays are evaluated as binary masks.

    Returns
    -------
        float:
            The averaged Dice score.

    Parameters
    ----------
        gt : np.ndarray
            Ground-truth probability map.
        pred : np.ndarray
            Predicted probability map.
        thresholds : list[float]
            Thresholds used to binarize the probability maps.

    Raises
    ------
        ValueError
            If `gt` and `pred` do not have the same shape, or if `thresholds` is empty.
    """
    if gt.shape != pred.shape:
        raise ValueError(f"Input arrays must have the same shape, got {gt.shape} and {pred.shape}")

    if not thresholds:
        raise ValueError("Threshold list must not be empty")

    scores = []
    for threshold in thresholds:
        th_gt = np.zeros(gt.shape, dtype=np.float32)
        th_gt[gt >= threshold] = 1

        th_pred = np.zeros(pred.shape, dtype=np.float32)
        th_pred[pred >= threshold] = 1

        denominator = np.sum(th_gt) + np.sum(th_pred)
        if denominator == 0:
            scores.append(1.0)
        else:
            scores.append(float(2 * np.sum(th_gt * th_pred) / denominator))

    return float(np.mean(scores))


def compute_ged(gt_imgs: list[np.ndarray], pred_imgs: list[np.ndarray]) -> float:
    """
    Compute generalized energy distance (GED) using IoU-based distances.

    Assumptions:
    - `gt_imgs` and `pred_imgs` are non-empty lists of masks.
    - All masks across both lists have identical shapes.
    - Non-zero values are treated as foreground for IoU computation.

    Returns
    -------
        float:
            The GED value.

    Parameters
    ----------
        gt_imgs : list[np.ndarray]
            Ground-truth segmentation variants.
        pred_imgs : list[np.ndarray]
            Predicted segmentation variants.

    Raises
    ------
        ValueError
            If `gt_imgs` or `pred_imgs` is empty.
    """
    if not gt_imgs:
        raise ValueError("Ground-truth image list must not be empty")
    if not pred_imgs:
        raise ValueError("Predicted image list must not be empty")

    reference_shape = gt_imgs[0].shape
    for img in gt_imgs + pred_imgs:
        if img.shape != reference_shape:
            raise ValueError("All images used for GED computation must have the same shape")

    sum1 = 0.0
    for gt_img in gt_imgs:
        for pred_img in pred_imgs:
            sum1 += 1 - iou(gt_img, pred_img)
    sum1 /= len(gt_imgs) * len(pred_imgs)

    if len(gt_imgs) < 2:
        sum2 = 0.0
    else:
        sum2 = 0.0
        pair_count = 0
        for i in range(len(gt_imgs) - 1):
            for j in range(i + 1, len(gt_imgs)):
                sum2 += 1 - iou(gt_imgs[i], gt_imgs[j])
                pair_count += 1
        sum2 /= pair_count

    if len(pred_imgs) < 2:
        sum3 = 0.0
    else:
        sum3 = 0.0
        pair_count = 0
        for i in range(len(pred_imgs) - 1):
            for j in range(i + 1, len(pred_imgs)):
                sum3 += 1 - iou(pred_imgs[i], pred_imgs[j])
                pair_count += 1
        sum3 /= pair_count

    return float(2 * sum1 - sum2 - sum3)


def evaluate_structure_probability_maps(
    data_file: Path,
    save_dir: Path,
    results_path: Path = Path("results.json"),
    thresholds: list[float] | None = None,
    labels: list[int] | None = None,
) -> dict[str, Any]:
    """
    Evaluate predicted structure probability maps on the test split using gray-level Dice,
    averaged Dice over thresholds (aDice), and generalized energy distance (GED). Results
    are aggregated per structure label and saved to disk.

    Assumptions:
    - `data_file` points to a valid JSON file containing a `"test"` split.
    - Test items contain `"moving_image"` paths following the pattern `Patient_<id>_fraction_<id>_.nii.gz`.
    - Probability maps are stored in `save_dir / Patient_<id> / PROBABILITY_MAPS /`.
    - Ground-truth variant masks are stored in `save_dir / Patient_<id> / GT_Variants /`.
    - Predicted variant masks are stored in `save_dir / Patient_<id> / Variants /`.
    - Probability maps and variant masks for the same patient and structure are shape-compatible.
    - The output directory for `results_path` is writable.

    Returns
    -------
        dict[str, object]:
            Dictionary containing patient ids and metric values grouped by structure label.

    Parameters
    ----------
        data_file : Path
            Path to the JSON file containing the test split definition.
        save_dir : Path
            Root directory containing saved structure probability maps and segmentation variants.
        results_path : Path
            Path to the JSON file where evaluation results will be saved.
        thresholds : list[float] | None
            Thresholds used to compute aDice. If `None`, defaults to `[0.1, ..., 1.0]`.
        labels : list[int] | None
            Structure labels to evaluate. If `None`, defaults to `[2, 3, 4, 5]`.

    Raises
    ------
        FileNotFoundError
            If `data_file` does not exist, if `save_dir` does not exist, or if any
            required patient subdirectories, probability maps, or variant files
            are missing.

        ValueError
            If the JSON file does not contain a valid `"test"` key, if `"test"` is
            not a list, or if any test item is missing the `"moving_image"` field.

        Exception
            Any exception raised during per-patient processing is caught and recorded
            in `failed_cases`, but does not stop execution.
    """
    if not data_file.is_file():
        raise FileNotFoundError(f"Data split file does not exist: {data_file}")

    if not save_dir.is_dir():
        raise FileNotFoundError(f"Save directory does not exist: {save_dir}")

    if thresholds is None:
        thresholds = [0.1 + i * 0.1 for i in range(10)]

    if labels is None:
        labels = [2, 3, 4, 5]

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

    dices: dict[int, list[float]] = {label: [] for label in labels}
    adices: dict[int, list[float]] = {label: [] for label in labels}
    geds: dict[int, list[float]] = {label: [] for label in labels}
    failed_cases: list[dict[str, str | int]] = []

    for label in labels:
        print(f"{label=}")

        for pid in ids:
            try:
                patient_dir = save_dir / f"Patient_{pid}"
                prob_maps_dir = patient_dir / "PROBABILITY_MAPS"
                gt_variants_dir = patient_dir / "GT_Variants"
                pred_variants_dir = patient_dir / "Variants"

                if not patient_dir.is_dir():
                    raise FileNotFoundError(f"Patient directory does not exist: {patient_dir}")
                if not prob_maps_dir.is_dir():
                    raise FileNotFoundError(f"Probability maps directory does not exist: {prob_maps_dir}")
                if not gt_variants_dir.is_dir():
                    raise FileNotFoundError(f"GT variants directory does not exist: {gt_variants_dir}")
                if not pred_variants_dir.is_dir():
                    raise FileNotFoundError(f"Predicted variants directory does not exist: {pred_variants_dir}")

                gt_prob_path = prob_maps_dir / f"GT_Patient_{pid}_STRUCTURE_{label}.nii.gz"
                pred_prob_path = prob_maps_dir / f"PRED_Patient_{pid}_STRUCTURE_{label}.nii.gz"

                if not gt_prob_path.is_file():
                    raise FileNotFoundError(f"Ground-truth probability map does not exist: {gt_prob_path}")
                if not pred_prob_path.is_file():
                    raise FileNotFoundError(f"Predicted probability map does not exist: {pred_prob_path}")

                gt = nib.load(str(gt_prob_path)).get_fdata()  # type: ignore
                pred = nib.load(str(pred_prob_path)).get_fdata()  # type: ignore

                dice = compute_gray_level_dice(gt, pred)
                adice = compute_adice(gt, pred, thresholds)

                gt_variant_paths = sorted(gt_variants_dir.glob(f"Variant_*_STRUCTURE_{label}_Patient_*.nii.gz"))
                pred_variant_paths = sorted(pred_variants_dir.glob(f"Variant_*_STRUCTURE_{label}_Patient_*.nii.gz"))

                if not gt_variant_paths:
                    raise FileNotFoundError(
                        f"No GT variant files found for patient {pid}, label {label} in: {gt_variants_dir}"
                    )
                if not pred_variant_paths:
                    raise FileNotFoundError(
                        f"No predicted variant files found for patient {pid}, label {label} in: {pred_variants_dir}"
                    )

                gt_imgs = [nib.load(str(path)).get_fdata() for path in gt_variant_paths]  # type: ignore
                pred_imgs = [nib.load(str(path)).get_fdata() for path in pred_variant_paths]  # type: ignore

                ged = compute_ged(gt_imgs, pred_imgs)

                dices[label].append(float(dice))
                adices[label].append(float(adice))
                geds[label].append(float(ged))

                partial_results = {
                    "ids": ids,
                    "dices": dices,
                    "adices": adices,
                    "geds": geds,
                    "failed_cases": failed_cases,
                }
                results_path.parent.mkdir(parents=True, exist_ok=True)
                with open(results_path, "w") as f:
                    json.dump(partial_results, f, indent=4)

                print(f"\tpid={pid}, dice={dice}, adice={adice}, ged={ged}")

            except Exception as e:
                print(f"\tFailed pid={pid}, label={label}: {e}")
                failed_cases.append(
                    {
                        "patient_id": pid,
                        "label": label,
                        "error": str(e),
                    }
                )
                continue

        if dices[label]:
            print(
                label,
                np.mean(dices[label]),
                np.std(dices[label]),
                np.mean(adices[label]),
                np.std(adices[label]),
                np.mean(geds[label]),
                np.std(geds[label]),
            )
        else:
            print(f"No successful evaluations for label {label}.")

    for key in dices:
        if dices[key]:
            print(key, np.mean(dices[key]), np.std(dices[key]))
        else:
            print(f"{key}: no Dice values.")

    for key in adices:
        if adices[key]:
            print(key, np.mean(adices[key]), np.std(adices[key]))
        else:
            print(f"{key}: no aDice values.")

    for key in geds:
        if geds[key]:
            print(key, np.mean(geds[key]), np.std(geds[key]))
        else:
            print(f"{key}: no GED values.")

    results = {
        "ids": ids,
        "dices": dices,
        "adices": adices,
        "geds": geds,
        "failed_cases": failed_cases,
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    return results


def prepare_moving_images_for_inference(
    data_file: Path,
    data_dir: Path,
    save_dir: Path,
    dshape: tuple[int, int] = (256, 256),
    factor: float = 2.0,
    num_channels: int = 4,
) -> None:
    """
    Prepare resized moving CT images for inference and save them as multiple channel-specific
    NIfTI files per patient.

    Assumptions:
    - `data_file` points to a valid JSON file containing a `"test"` split.
    - Test items contain `"moving_image"` paths following the pattern `Patient_<id>_fraction_<id>_.nii.gz`.
    - Moving CT images are stored in `data_dir / Patient_<id> / Patient_<id>_fraction_1_.nii.gz`.
    - Input CT images are 3D NIfTI volumes.
    - `dshape` defines the resized in-plane shape `(height, width)`.
    - `factor` matches the in-plane resizing ratio used to update the affine matrix.
    - The output directory is writable.

    Returns
    -------
        None:
            The function saves resized moving images to `save_dir` as
            `image_<pid>_0000.nii.gz`, ..., `image_<pid>_<channel>.nii.gz`.

    Parameters
    ----------
        data_file : Path
            Path to the JSON file containing the test split definition.
        data_dir : Path
            Directory containing patient CT subdirectories.
        save_dir : Path
            Output directory where resized moving images will be saved.
        dshape : tuple[int, int]
            Target in-plane shape `(height, width)` used during resizing.
        factor : float
            Scaling factor applied to the first two affine diagonal elements after resizing.
        num_channels : int
            Number of identical channel-specific output files to save per patient.

    Raises
    ------
        FileNotFoundError
            If `data_file` does not exist, if `data_dir` does not exist, or if a
            required moving image file for any patient is missing.

        ValueError
            If the JSON file does not contain a valid `"test"` key, if `"test"` is
            not a list, if any test item is missing the `"moving_image"` field, or
            if an input image is not 3-dimensional.
    """
    if not data_file.is_file():
        raise FileNotFoundError(f"Data split file does not exist: {data_file}")

    if not data_dir.is_dir():
        raise FileNotFoundError(f"CT data directory does not exist: {data_dir}")

    save_dir.mkdir(parents=True, exist_ok=True)

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

            image_path = data_dir / f"Patient_{pid}" / f"Patient_{pid}_fraction_1_.nii.gz"
            if not image_path.is_file():
                raise FileNotFoundError(f"Moving image does not exist for patient {pid}: {image_path}")

            nifti = nib.load(str(image_path))
            fixed_img = nifti.get_fdata()  # type: ignore
            aff = nifti.affine.copy()  # type: ignore

            if fixed_img.ndim != 3:
                raise ValueError(f"Expected a 3D image for patient {pid}, but got shape {fixed_img.shape}")

            resized_fixed_img = resize(
                fixed_img,
                dshape + (fixed_img.shape[2],),
                anti_aliasing=True,
                preserve_range=True,
            )  # type: ignore

            aff[0, 0] *= factor
            aff[1, 1] *= factor

            nifti_image = nib.Nifti1Image(resized_fixed_img, affine=aff)  # type: ignore

            for channel_idx in range(num_channels):
                output_path = save_dir / f"image_{pid}_{channel_idx:04d}.nii.gz"
                nib.save(nifti_image, str(output_path))

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
            If `data_file` does not exist or if `save_dir` does not exist.

        ValueError
            If the JSON file does not contain a valid `"test"` key, if `"test"` is
            not a list, or if any test item is missing the `"moving_image"` field.
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
                        f"GT variant files for patient {pid}, structure {sid} do not all share the same shape"
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
                        f"Predicted variant files for patient {pid}, structure {sid} do not all share the same shape"
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
