import itertools
import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.metrics.regression import SSIMMetric
from tqdm import tqdm

from src.pipeline.config import MaisiTestingConfig
from src.pipeline.helpers.helpers import (
    LPIPSModel,
    _as_metric_tensor,
    _build_lpips_model,
    _cache_patient_cts,
    _evenly_spaced_slices_for_lpips,
    _exclude_planning_cts,
    _resolve_metric_device,
    _validate_3d_pair,
)

logger = logging.getLogger(__name__)


def mae_3d(pred: torch.Tensor | np.ndarray, ref: torch.Tensor | np.ndarray) -> float:
    """
    Calculate mean absolute error between two 3D images.

    MAE measures the average absolute voxel-wise difference between the
    predicted/generated CT and the reference CT.

    Parameters
    ----------
    pred : torch.Tensor | np.ndarray
        Predicted or generated 3D image.

    ref : torch.Tensor | np.ndarray
        Reference 3D image.

    Returns
    -------
    mae : float
        Mean absolute error between `pred` and `ref`.

    Raises
    ------
    ValueError
        If `pred` or `ref` is empty.
        If `pred` and `ref` have different shapes.
        If `pred` or `ref` is not 3-dimensional.
        If `pred` or `ref` contains NaN or infinite values.
    """

    device = _resolve_metric_device(pred, ref)
    pred_t = _as_metric_tensor(pred, device=device)
    ref_t = _as_metric_tensor(ref, device=device)
    _validate_3d_pair(pred_t, ref_t)

    return float(torch.mean(torch.abs(pred_t - ref_t)).item())


def ssim_3d(pred: torch.Tensor | np.ndarray, ref: torch.Tensor | np.ndarray, data_min: float, data_max: float) -> float:
    """
    Calculate the mean Structural Similarity Index (SSIM) for two 3D images.

    This function computes SSIM directly on the full 3D volume using MONAI's
    regression metric implementation. It is intended for comparing volumetric
    medical images such as CT scans.

    Parameters
    ----------
    pred : torch.Tensor | np.ndarray
        Predicted 3D image.

    ref : torch.Tensor | np.ndarray
        Reference (ground-truth) 3D image.

    data_min : float
        Minimum value allowed for the data.

    data_max : float
        Maximum value allowed for the data.

    Returns
    -------
    score : float
        Mean 3D SSIM score. Values closer to 1 indicate higher structural
        similarity.

    Raises
    ------
    ValueError
        If `pred` or `ref` is empty.
        If `pred` and `ref` have different shapes.
        If `pred` or `ref` is not 3-dimensional.
        If `pred` or `ref` contains NaN or infinite values.
        If `data_min` is greater than `data_max`.
        If data falls outside of [data_min, data_max].
    """

    device = _resolve_metric_device(pred, ref)
    pred_t = _as_metric_tensor(pred, device=device)
    ref_t = _as_metric_tensor(ref, device=device)
    _validate_3d_pair(pred_t, ref_t)

    if data_min >= data_max:
        raise ValueError(f"`data_min` must be smaller than `data_max`. Got data_min={data_min}, data_max={data_max}")

    pred_min = float(pred_t.min().item())
    pred_max = float(pred_t.max().item())
    ref_min = float(ref_t.min().item())
    ref_max = float(ref_t.max().item())

    if pred_min < data_min or pred_max > data_max:
        raise ValueError(f"`pred` values must be in the range [data_min, data_max], got min={pred_min}, max={pred_max}")

    if ref_min < data_min or ref_max > data_max:
        raise ValueError(f"`ref` values must be in the range [data_min, data_max], got min={ref_min}, max={ref_max}")

    window_size = min(11, *(int(dim) for dim in pred_t.shape))
    if window_size % 2 == 0:
        window_size -= 1

    if window_size < 3:
        raise ValueError(f"SSIM window is invalid for image shape {tuple(pred_t.shape)}")

    # MONAI expects [batch, channel, H, W, D]. The metric returns a tensor, so
    # reduce explicitly to a Python float for CSV-friendly output
    metric = SSIMMetric(
        spatial_dims=3,
        data_range=data_max - data_min,
        kernel_type="uniform",
        win_size=window_size,
    )

    with torch.no_grad():
        score = cast(
            torch.Tensor,
            metric(
                pred_t.unsqueeze(0).unsqueeze(0),
                ref_t.unsqueeze(0).unsqueeze(0),
            ),
        )

    return float(score.mean().item())


def psnr_3d(pred: torch.Tensor | np.ndarray, ref: torch.Tensor | np.ndarray, data_min: float, data_max: float) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two 3D images.

    PSNR measures reconstruction quality from the mean squared voxel-wise
    difference between the predicted/generated CT and the reference CT.
    Higher values indicate closer agreement. Identical volumes return
    positive infinity, matching the standard PSNR definition.

    Parameters
    ----------
    pred : torch.Tensor | np.ndarray
        Predicted or generated 3D image.

    ref : torch.Tensor | np.ndarray
        Reference 3D image.

    data_min : float
        Minimum value allowed for the data.

    data_max : float
        Maximum value allowed for the data.

    Returns
    -------
    psnr : float
        Peak Signal-to-Noise Ratio in decibels.

    Raises
    ------
    ValueError
        If `pred` or `ref` is empty.
        If `pred` and `ref` have different shapes.
        If `pred` or `ref` is not 3-dimensional.
        If `pred` or `ref` contains NaN or infinite values.
        If `data_min` is greater than or equal to `data_max`.
        If data falls outside of [data_min, data_max].
    """

    device = _resolve_metric_device(pred, ref)
    pred_t = _as_metric_tensor(pred, device=device)
    ref_t = _as_metric_tensor(ref, device=device)
    _validate_3d_pair(pred_t, ref_t)

    if data_min >= data_max:
        raise ValueError(f"`data_min` must be smaller than `data_max`. Got data_min={data_min}, data_max={data_max}")

    pred_min = float(pred_t.min().item())
    pred_max = float(pred_t.max().item())
    ref_min = float(ref_t.min().item())
    ref_max = float(ref_t.max().item())

    if pred_min < data_min or pred_max > data_max:
        raise ValueError(f"`pred` values must be in the range [data_min, data_max], got min={pred_min}, max={pred_max}")

    if ref_min < data_min or ref_max > data_max:
        raise ValueError(f"`ref` values must be in the range [data_min, data_max], got min={ref_min}, max={ref_max}")

    mse = torch.mean((pred_t - ref_t).square())

    if float(mse.item()) == 0.0:
        return float("inf")

    data_range = data_max - data_min
    psnr = 10.0 * torch.log10(torch.as_tensor((data_range**2), dtype=pred_t.dtype, device=device) / mse)

    return float(psnr.item())


def sobel_edge_map_3d(arr: torch.Tensor | np.ndarray) -> torch.Tensor:
    """
    Calculate a 3D Sobel edge magnitude map.

    This function applies the Sobel operator along all three spatial axes and
    combines the resulting gradients into one edge-magnitude image.

    It can be used for SOB / Sobel-based edge similarity, where the goal is to
    compare whether two CT images have similar anatomical edge structure.

    Parameters
    ----------
    arr : torch.Tensor | np.ndarray
        Input 3D image.

    Returns
    -------
    edge : torch.Tensor
        3D Sobel edge magnitude map.

    Raises
    ------
    ValueError
        If `arr` is empty.
        If `arr` is not 3-dimensional.
        If `arr` contains NaN or infinite values.
    """

    arr_t = _as_metric_tensor(arr)

    if arr_t.numel() == 0:
        raise ValueError("`arr` cannot be empty")

    if arr_t.ndim != 3:
        raise ValueError(f"`arr` must be a 3D tensor with shape [H, W, Z]. Got shape {tuple(arr_t.shape)}")

    if not torch.isfinite(arr_t).all():
        raise ValueError("`arr` contains NaN or infinite values")

    derivative = torch.tensor([-1.0, 0.0, 1.0], dtype=arr_t.dtype, device=arr_t.device)
    smoothing = torch.tensor([1.0, 2.0, 1.0], dtype=arr_t.dtype, device=arr_t.device)

    # Apply x/y/z Sobel filters together
    kernel_x = derivative[:, None, None] * smoothing[None, :, None] * smoothing[None, None, :]
    kernel_y = smoothing[:, None, None] * derivative[None, :, None] * smoothing[None, None, :]
    kernel_z = smoothing[:, None, None] * smoothing[None, :, None] * derivative[None, None, :]
    kernels = torch.stack([kernel_x, kernel_y, kernel_z]).unsqueeze(1)

    volume = arr_t.unsqueeze(0).unsqueeze(0)
    gradients = F.conv3d(volume, kernels, padding=1)

    return cast(torch.Tensor, torch.linalg.vector_norm(gradients.squeeze(0), dim=0))


def sob_3d(pred: torch.Tensor | np.ndarray, ref: torch.Tensor | np.ndarray) -> float:
    """
    Calculate SOB / Sobel-based edge difference between two 3D images.

    This metric computes Sobel edge maps for the predicted/generated image
    and the reference image, then returns the mean absolute difference between
    those edge maps.

    Lower values mean the generated image has a more similar edge or structure
    pattern to the reference CT.

    Parameters
    ----------
    pred : torch.Tensor | np.ndarray
        Predicted or generated 3D image.

    ref : torch.Tensor | np.ndarray
        Reference 3D image.

    Returns
    -------
    sob : float
        Mean absolute difference between Sobel edge maps.

    Raises
    ------
    ValueError
        If `pred` or `ref` is empty.
        If `pred` or `ref` is not 3-dimensional.
        If `pred` and `ref` have different shapes.
        If `pred` or `ref` contains NaN or infinite values.
    """

    device = _resolve_metric_device(pred, ref)
    pred_t = _as_metric_tensor(pred, device=device)
    ref_t = _as_metric_tensor(ref, device=device)
    _validate_3d_pair(pred_t, ref_t)

    pred_edge = sobel_edge_map_3d(pred_t)
    ref_edge = sobel_edge_map_3d(ref_t)

    return float(torch.mean(torch.abs(pred_edge - ref_edge)).item())


def lpips_3d(
    lpips_model: LPIPSModel,
    pred: torch.Tensor | np.ndarray,
    ref: torch.Tensor | np.ndarray,
    data_min: float,
    data_max: float,
    max_slices: int,
) -> float:
    """
    Calculate slice-wise LPIPS between two 3D images and average the result.

    LPIPS is normally defined for 2D RGB images. For 3D CT volumes, this
    function selects evenly spaced slices across the full volume, converts them
    to LPIPS input format, computes LPIPS slice-wise, and returns the mean score.

    Using evenly spaced slices makes the metric less sensitive to one noisy or
    unrepresentative center slice.

    Interpretation:
    - Higher LPIPS means greater perceptual difference.
    - When comparing generated variants with each other, higher pairwise LPIPS
    usually suggests higher visual diversity.
    - When comparing generated CTs to reference CTs, lower LPIPS means higher
    perceptual similarity.

    Parameters
    ----------
    lpips_model : LPIPSModel
        Initialized LPIPS model.

    pred : torch.Tensor | np.ndarray
        Predicted or generated 3D CT array normalized to [data_min, data_max].

    ref : torch.Tensor | np.ndarray
        Reference 3D CT array normalized to [data_min, data_max].

    data_min : float
        Minimum value allowed for the data.

    data_max : float
        Maximum value allowed for the data.

    max_slices : int
        Maximum number of evenly spaced slices used for LPIPS calculation.

    Returns
    -------
    lpips_score : float
        Mean LPIPS score across selected slices.

    Raises
    ------
    ValueError
        If `pred` or `ref` is empty.
        If `pred` or `ref` is not 3-dimensional.
        If `pred` and `ref` have different shapes.
        If `pred` or `ref` contains NaN or infinite values.
        If `pred` or `ref` is not normalized to [data_min, data_max].
        If `max_slices` is less than 1.
    """

    device = _resolve_metric_device(pred, ref)
    pred_t = _as_metric_tensor(pred, device=device)
    ref_t = _as_metric_tensor(ref, device=device)
    _validate_3d_pair(pred_t, ref_t)

    if data_min >= data_max:
        raise ValueError(f"`data_min` must be smaller than `data_max`. Got data_min={data_min}, data_max={data_max}")

    if pred_t.min() < data_min or pred_t.max() > data_max:
        raise ValueError("`pred` must be normalized to [data_min, data_max]")

    if ref_t.min() < data_min or ref_t.max() > data_max:
        raise ValueError("`ref` must be normalized to [data_min, data_max]")

    # LPIPS is 2D, so compare a bounded slice sample
    pred_tensor = _evenly_spaced_slices_for_lpips(
        (pred_t - data_min) / (data_max - data_min),
        max_slices=max_slices,
    )

    ref_tensor = _evenly_spaced_slices_for_lpips(
        (ref_t - data_min) / (data_max - data_min),
        max_slices=max_slices,
    )

    with torch.no_grad():
        score = lpips_model(pred_tensor, ref_tensor)

    return float(score.mean().item())


def calculate_similarity_metrics(
    generated: dict[str, list[Path]],
    originals: dict[str, list[Path]],
    config: MaisiTestingConfig,
    lpips_model: LPIPSModel | None = None,
) -> None:
    """
    Calculate generated-vs-real CT similarity metrics.

    This function compares generated CT variants with available non-planning
    original/reference CT tensors for the same patient. It is robust to missing
    generated variants and calculates only comparisons that are possible.

    Metrics:
    - MAE: lower is better
    - SSIM: higher is better
    - PSNR: higher is better
    - SOB/Sobel MAE: lower is better
    - LPIPS: higher means greater perceptual difference

    Outputs:
    - generated_vs_real_metrics.csv
        Per-comparison metrics.
    - generated_vs_real_summary.csv
        Patient-level summary statistics.

    Parameters
    ----------
    generated : dict[str, list[Path]]
        Generated CT tensor paths grouped by patient ID.

    originals : dict[str, list[Path]]
        Original/reference CT tensor paths grouped by patient ID. Planning CTs
        are excluded before generated-vs-real comparisons are calculated.

    config : MaisiTestingConfig
        Configuration object containing evaluation settings and directory paths.

    lpips_model : LPIPSModel | None, optional
        Optional initialized LPIPS model. If ``None``, LPIPS values are left as
        NaN even when ``config.use_lpips`` is enabled.

    Raises
    ------
    ValueError
        If no valid generated-vs-real comparisons can be calculated.
    """

    device = torch.device(config.device)
    rows = []
    warnings: list[str] = []

    logger.info("Image similarity metrics will run on %s", device)
    for patient_id, gen_paths in tqdm(
        generated.items(),
        desc="Calculating generated vs real image metrics",
    ):
        all_ref_paths = originals.get(patient_id, [])
        ref_paths = _exclude_planning_cts({patient_id: all_ref_paths})[patient_id]

        if len(all_ref_paths) == 0:
            warnings.append(f"Skipping {patient_id}: no original/reference CT found")
            continue

        if len(ref_paths) == 0:
            warnings.append(f"Skipping {patient_id}: no non-planning original/reference CT found")
            continue

        patient_paths = [*gen_paths, *ref_paths]
        # Image metrics reuse each generated/reference CT multiple times for a
        # patient, so caching the patient tensors avoids repeated disk loads
        patient_tensors = _cache_patient_cts(patient_paths, config=config, device=device)

        for gen_path in gen_paths:
            gen_arr = patient_tensors[gen_path]
            for ref_path in ref_paths:
                ref_arr = patient_tensors[ref_path]

                if gen_arr.shape != ref_arr.shape:
                    warnings.append(
                        "Skipping shape mismatch for "
                        f"{patient_id}: {gen_path.name} {tuple(gen_arr.shape)} vs "
                        f"{ref_path.name} {tuple(ref_arr.shape)}"
                    )
                    continue

                row = {
                    "patient_id": patient_id,
                    "comparison_type": "generated_vs_real",
                    "generated_path": str(gen_path),
                    "reference_path": str(ref_path),
                    "mae": mae_3d(gen_arr, ref_arr),
                    "ssim": ssim_3d(gen_arr, ref_arr, config.data_min, config.data_max),
                    "psnr": psnr_3d(gen_arr, ref_arr, config.data_min, config.data_max),
                    "sob": sob_3d(gen_arr, ref_arr),
                    "lpips": np.nan,
                }

                if config.use_lpips and lpips_model is not None:
                    row["lpips"] = lpips_3d(
                        lpips_model=lpips_model,
                        pred=gen_arr,
                        ref=ref_arr,
                        data_min=config.data_min,
                        data_max=config.data_max,
                        max_slices=config.max_lpips_slices,
                    )

                rows.append(row)

    for message in warnings:
        logger.warning(message)

    if len(rows) == 0:
        raise ValueError(
            "No valid generated-vs-real comparisons were calculated. "
            "Check whether patient IDs match and tensor shapes are compatible"
        )

    df = pd.DataFrame(rows)
    metrics_path = config.metrics_dir / "generated_vs_real_metrics.csv"
    df.to_csv(metrics_path, index=False)
    logger.info("Saved generated-vs-real image metrics: %s", metrics_path)

    summary = df.groupby("patient_id")[["mae", "ssim", "psnr", "sob", "lpips"]].agg(
        ["mean", "std", "min", "max", "count"]
    )
    summary_path = config.metrics_dir / "generated_vs_real_summary.csv"
    summary.to_csv(summary_path)
    logger.info("Saved generated-vs-real image metric summary: %s", summary_path)


def calculate_pairwise_variety_metrics(
    ct_groups: dict[str, list[Path]],
    config: MaisiTestingConfig,
    lpips_model: LPIPSModel | None = None,
) -> None:
    """
    Calculate pairwise variety metrics within groups of CT images.

    This function compares all possible generated image pairs within each
    patient group.

    Metrics:
    - MAE: lower means more similar voxel intensities
    - SSIM: higher means more similar structure
    - PSNR: higher means lower squared voxel-wise error
    - SOB: lower means more similar edge structure
    - LPIPS: higher usually means more visual/perceptual difference

    Parameters
    ----------
    ct_groups : dict[str, list[Path]]
        Dictionary mapping patient IDs to CT tensor paths.

    config : MaisiTestingConfig
        Configuration object containing evaluation settings.

    lpips_model : LPIPSModel | None, optional
        Optional initialized LPIPS model. Reusing the model from
        ``calculate_similarity_metrics`` avoids constructing it twice.

    Raises
    ------
    ValueError
        If `ct_groups` is empty.
        If `max_lpips_slices` is less than 1.
        If the inputs are invalid. If no valid pairwise comparisons can be
        calculated, the metric file is skipped.
    """

    if len(ct_groups) == 0:
        raise ValueError("`ct_groups` cannot be empty")

    device = torch.device(config.device)
    if config.use_lpips and lpips_model is None:
        lpips_model = _build_lpips_model(config)
    rows = []
    warnings: list[str] = []

    logger.info("Pairwise variety metrics will run on %s", device)
    for patient_id, paths in tqdm(ct_groups.items(), desc="Calculating pairwise variety metrics"):
        if len(paths) < 2:
            warnings.append(
                f"Skipping pairwise generated variety metrics for {patient_id}: only {len(paths)} image(s)"
            )
            continue

        # Pairwise metrics compare every generated CT pair for a patient, so
        # each tensor is loaded once and reused across all combinations
        patient_tensors = _cache_patient_cts(paths, config=config, device=device)

        for path_a, path_b in itertools.combinations(paths, 2):
            arr_a = patient_tensors[path_a]
            arr_b = patient_tensors[path_b]
            if arr_a.shape != arr_b.shape:
                warnings.append(
                    "Skipping shape mismatch for "
                    f"{patient_id}: {path_a.name} {tuple(arr_a.shape)} vs "
                    f"{path_b.name} {tuple(arr_b.shape)}"
                )
                continue

            row = {
                "patient_id": patient_id,
                "path_a": str(path_a),
                "path_b": str(path_b),
                "mae": mae_3d(arr_a, arr_b),
                "ssim": ssim_3d(arr_a, arr_b, config.data_min, config.data_max),
                "psnr": psnr_3d(arr_a, arr_b, config.data_min, config.data_max),
                "sob": sob_3d(arr_a, arr_b),
                "lpips": np.nan,
            }

            if config.use_lpips and lpips_model is not None:
                row["lpips"] = lpips_3d(
                    pred=arr_a,
                    ref=arr_b,
                    lpips_model=lpips_model,
                    data_min=config.data_min,
                    data_max=config.data_max,
                    max_slices=config.max_lpips_slices,
                )

            rows.append(row)

    for message in warnings:
        logger.warning(message)

    if len(rows) == 0:
        logger.warning("Skipping generated pairwise variety metrics: no valid pairwise comparisons were calculated")
        return

    df = pd.DataFrame(rows)
    metrics_path = config.metrics_dir / "generated_pairwise_variety_metrics.csv"
    df.to_csv(metrics_path, index=False)
    logger.info("Saved generated pairwise variety metrics: %s", metrics_path)

    summary = df.groupby("patient_id")[["mae", "ssim", "psnr", "sob", "lpips"]].agg(
        ["mean", "std", "min", "max", "count"]
    )
    summary_path = config.metrics_dir / "generated_pairwise_variety_metrics_summary.csv"
    summary.to_csv(summary_path)
    logger.info("Saved generated pairwise variety metric summary: %s", summary_path)
