from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


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
    """
    if x.shape != y.shape:
        raise ValueError(f"Input arrays must have the same shape, got {x.shape} and {y.shape}.")

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
    """
    if gt.shape != pred.shape:
        raise ValueError(f"Input arrays must have the same shape, got {gt.shape} and {pred.shape}.")

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
    """
    if gt.shape != pred.shape:
        raise ValueError(f"Input arrays must have the same shape, got {gt.shape} and {pred.shape}.")

    if not thresholds:
        raise ValueError("Threshold list must not be empty.")

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
    """
    if not gt_imgs:
        raise ValueError("Ground-truth image list must not be empty.")
    if not pred_imgs:
        raise ValueError("Predicted image list must not be empty.")

    reference_shape = gt_imgs[0].shape
    for img in gt_imgs + pred_imgs:
        if img.shape != reference_shape:
            raise ValueError("All images used for GED computation must have the same shape.")

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
    - Probability maps are stored in `save_dir / Patient_<id> / PROBABILIY_MAPS /`.
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
                prob_maps_dir = patient_dir / "PROBABILIY_MAPS"
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
