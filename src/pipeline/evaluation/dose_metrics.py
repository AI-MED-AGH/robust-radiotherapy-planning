import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.pipeline.config import MaisiTestingConfig
from src.pipeline.evaluation.structure_metrics import (
    WarpedStructureMaskCache,
)
from src.pipeline.helpers.helpers import (
    _collect_dose_distributions,
    _dose_at_volume_from_values,
    _dose_volume_histogram_from_values,
    _generated_ct_lookup,
    _get_dose_transform,
    _get_structure_label_transform,
    _label_to_binary_mask,
    _load_dose_comparison_data,
    _masked_dose_values,
    _masked_error_metrics_from_values,
    _maximum_dose_from_values,
    _mean_dose_from_values,
    _metric_row,
    _normalised_image_key,
    _reference_lookup,
    _volume_at_dose_from_values,
)

logger = logging.getLogger(__name__)


@dataclass
class _DoseComparisonData:
    """
    Store CPU-loaded data for one predicted/reference dose comparison.

    Dose maps and structure labels are loaded on CPU because NIfTI decoding and
    MONAI transforms are CPU-bound. The metric loop moves only the current
    comparison to the configured metric device, which keeps GPU memory bounded.
    """

    pred_path: Path
    ref_path: Path
    pred_dose: torch.Tensor | None
    ref_dose: torch.Tensor | None
    ref_label_map: torch.Tensor | None
    pred_masks: dict[int, torch.Tensor] | None
    pred_structure_source: str
    ref_structure_source: str
    info_message: str | None = None
    nonfatal_warning: str | None = None
    warning: str | None = None


def dose_volume_histogram(
    dose: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
    bin_width: float,
) -> pd.DataFrame:
    """
    Calculate a cumulative dose-volume histogram for a masked volume.

    The returned `volume_percent` is the percentage of structure voxels
    receiving at least each dose-bin value.

    Parameters
    ----------
    dose : torch.Tensor | np.ndarray
        3D dose map.

    mask : torch.Tensor | np.ndarray
        Binary mask selecting the evaluated target/OAR volume.

    bin_width : float
        DVH dose-bin width in Gy.

    Returns
    -------
    dvh : pd.DataFrame
        DataFrame with `dose_gy` and cumulative `volume_percent` columns.

    Raises
    ------
    ValueError
        If `bin_width` is non-positive or dose/mask shapes differ.
    """

    if bin_width <= 0:
        raise ValueError("`bin_width` must be greater than 0")

    structure_dose = _masked_dose_values(dose, mask)
    return _dose_volume_histogram_from_values(structure_dose, bin_width)


def dose_at_volume(
    dose: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
    volume_percent: float,
) -> float:
    """
    Calculate Dx: minimum dose received by at least x percent of a volume.

    For example, D95 is the 5th percentile of masked dose values.

    Parameters
    ----------
    dose : torch.Tensor | np.ndarray
        3D dose map.

    mask : torch.Tensor | np.ndarray
        Binary mask selecting the evaluated target/OAR volume.

    volume_percent : float
        Percent volume used by the Dx definition. D95 corresponds to
        `volume_percent=95`.

    Returns
    -------
    dose_value : float
        Dose value in the same units as `dose`, usually Gy. Empty masks return
        NaN.

    Raises
    ------
    ValueError
        If `volume_percent` is outside `[0, 100]`.
    """

    if volume_percent < 0 or volume_percent > 100:
        raise ValueError(f"`volume_percent` must be in [0, 100]. Got {volume_percent}")

    return _dose_at_volume_from_values(_masked_dose_values(dose, mask), volume_percent)


def volume_at_dose(
    dose: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
    dose_threshold: float,
) -> float:
    """
    Calculate Vx: percent of a volume receiving at least x Gy.

    Parameters
    ----------
    dose : torch.Tensor | np.ndarray
        3D dose map.

    mask : torch.Tensor | np.ndarray
        Binary mask selecting the evaluated target/OAR volume.

    dose_threshold : float
        Dose threshold in the same units as `dose`, usually Gy.

    Returns
    -------
    volume_percent : float
        Percent of masked voxels receiving at least `dose_threshold`. Empty
        masks return NaN.
    """

    return _volume_at_dose_from_values(_masked_dose_values(dose, mask), dose_threshold)


def mean_dose(dose: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray) -> float:
    """
    Calculate mean dose inside a masked volume.

    Parameters
    ----------
    dose : torch.Tensor | np.ndarray
        3D dose map.

    mask : torch.Tensor | np.ndarray
        Binary mask selecting the evaluated target/OAR volume.

    Returns
    -------
    mean_value : float
        Mean masked dose. Empty masks return NaN.
    """

    return _mean_dose_from_values(_masked_dose_values(dose, mask))


def maximum_dose(dose: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray) -> float:
    """
    Calculate maximum dose inside a masked volume.

    Parameters
    ----------
    dose : torch.Tensor | np.ndarray
        3D dose map.

    mask : torch.Tensor | np.ndarray
        Binary mask selecting the evaluated target/OAR volume.

    Returns
    -------
    max_value : float
        Maximum masked dose. Empty masks return NaN.
    """

    return _maximum_dose_from_values(_masked_dose_values(dose, mask))


def calculate_dose_evaluation_metrics(
    predicted: dict[str, list[Path]],
    references: dict[str, list[Path]],
    config: MaisiTestingConfig,
    model_name: str,
    warped_mask_cache: WarpedStructureMaskCache | None = None,
) -> None:
    """
    Calculate dose evaluation metrics for predicted-vs-reference doses.

    Dose files and structure label maps are loaded and cached on CPU. Metric
    reductions run on `config.device`, so CUDA can accelerate masked dose
    errors, Dx/Vx, mean/max dose, and DVH binning. When CUDA is used and a
    patient has multiple comparisons, a single background worker prepares the
    next CPU dose comparison while the current one is evaluated on the metric
    device, matching the CPU/GPU cooperation pattern used by structure metrics.

    Outputs:
    - `dose_distribution_metrics.csv`: voxel, mean-dose, and max-dose rows.
    - `dose_clinical_metrics.csv`: Dx and Vx rows.
    - `dose_dvh.csv`: cumulative DVH rows for plotting or downstream analysis.
    - summary CSVs grouped by model, patient, structure label, and metric.

    Parameters
    ----------
    predicted : dict[str, list[Path]]
        Predicted dose paths grouped by patient ID.

    references : dict[str, list[Path]]
        Reference/ground-truth dose paths grouped by patient ID.

    config : MaisiTestingConfig
        Pipeline configuration containing metric device, structure labels,
        dose metric settings, and output paths.

    model_name : str
        Name written to output rows for the evaluated model or baseline.

    warped_mask_cache : WarpedStructureMaskCache | None, optional
        Optional cache produced by structure metrics. When provided, dose
        metrics reuse generated-space target/OAR masks from structure
        evaluation instead of repeating DVF registration and mask warping.

    Raises
    ------
    RuntimeError
        If dose metrics are configured to use CUDA but CUDA is unavailable.
    """

    dose_transform = _get_dose_transform(config)
    structure_transform = _get_structure_label_transform(config)
    generated_ct_lookup = _generated_ct_lookup(config)
    shared_generated_mask_cache = warped_mask_cache if warped_mask_cache is not None else {}

    distribution_rows: list[dict[str, Any]] = []
    clinical_rows: list[dict[str, Any]] = []
    dvh_rows: list[dict[str, Any]] = []
    info_messages: list[str] = []
    warnings: list[str] = []
    metric_device = torch.device(config.device)
    if metric_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Dose metrics requested CUDA, but torch.cuda.is_available() is False")

    logger.info(
        "Calculating dose metrics: "
        f"{sum(len(paths) for paths in predicted.values())} predicted dose(s), "
        f"{sum(len(paths) for paths in references.values())} reference dose(s), "
        f"labels={config.structure_labels}, metric_device={metric_device}"
    )

    def append_metrics_for_comparison(result: _DoseComparisonData, current_patient_id: str) -> None:
        """
        Move one CPU-loaded dose comparison to the metric device and append rows.

        Only the current predicted/reference dose pair and one mask at a time
        are moved to the metric device. This keeps memory bounded while still
        allowing CUDA to accelerate reductions and quantiles.

        Parameters
        ----------
        result : _DoseComparisonData
            CPU-loaded data for one dose comparison.

        current_patient_id : str
            Patient identifier written to output rows and warnings.
        """

        if result.warning is not None:
            warnings.append(result.warning)
            return

        if result.info_message is not None:
            info_messages.append(result.info_message)

        if result.nonfatal_warning is not None:
            warnings.append(result.nonfatal_warning)

        if result.pred_dose is None or result.ref_dose is None:
            warnings.append(f"Skipping dose metrics for {current_patient_id}: failed to load {result.pred_path}")
            return

        pred_dose = result.pred_dose.to(metric_device)
        ref_dose = result.ref_dose.to(metric_device)

        with torch.inference_mode():
            masks: list[tuple[str, int | None, torch.Tensor, torch.Tensor, str, str]] = [
                (
                    "overall",
                    None,
                    torch.ones_like(pred_dose, dtype=torch.bool, device=metric_device),
                    torch.ones_like(ref_dose, dtype=torch.bool, device=metric_device),
                    "full_predicted_dose_grid",
                    "full_reference_dose_grid",
                )
            ]

            if result.ref_label_map is not None:
                ref_label_map = result.ref_label_map.to(metric_device)
                for label in config.structure_labels:
                    ref_mask = _label_to_binary_mask(ref_label_map, label)
                    if result.pred_masks is None:
                        pred_mask = ref_mask
                        pred_structure_source = result.pred_structure_source
                    else:
                        pred_mask = result.pred_masks[label].to(metric_device)
                        pred_structure_source = result.pred_structure_source

                    masks.append(
                        (
                            f"label_{label}",
                            label,
                            pred_mask,
                            ref_mask,
                            pred_structure_source,
                            result.ref_structure_source,
                        )
                    )

            for structure_name, structure_label, pred_mask, ref_mask, pred_mask_source, ref_mask_source in masks:
                pred_structure_dose = pred_dose[pred_mask.bool()]
                ref_structure_dose = ref_dose[ref_mask.bool()]

                if torch.equal(pred_mask, ref_mask):
                    voxel_mae, voxel_rmse = _masked_error_metrics_from_values(
                        pred_structure_dose,
                        ref_structure_dose,
                    )
                else:
                    voxel_mae, voxel_rmse = float("nan"), float("nan")

                base = {
                    "model_name": model_name,
                    "patient_id": current_patient_id,
                    "comparison_type": "predicted_dose_vs_reference_dose",
                    "predicted_path": str(result.pred_path),
                    "reference_path": str(result.ref_path),
                    "structure_name": structure_name,
                    "structure_label": structure_label,
                    "predicted_structure_source": pred_mask_source,
                    "reference_structure_source": ref_mask_source,
                }

                distribution_rows.extend(
                    [
                        _metric_row("voxel_mae", voxel_mae, 0.0, base),
                        _metric_row("voxel_rmse", voxel_rmse, 0.0, base),
                        _metric_row(
                            "mean_dose",
                            _mean_dose_from_values(pred_structure_dose),
                            _mean_dose_from_values(ref_structure_dose),
                            base,
                        ),
                        _metric_row(
                            "maximum_dose",
                            _maximum_dose_from_values(pred_structure_dose),
                            _maximum_dose_from_values(ref_structure_dose),
                            base,
                        ),
                    ]
                )

                for volume_percent in config.dose_dx_volume_percents:
                    clinical_rows.append(
                        _metric_row(
                            f"D{volume_percent:g}",
                            _dose_at_volume_from_values(pred_structure_dose, volume_percent),
                            _dose_at_volume_from_values(ref_structure_dose, volume_percent),
                            base,
                        )
                    )

                for dose_threshold in config.dose_vx_thresholds:
                    clinical_rows.append(
                        _metric_row(
                            f"V{dose_threshold:g}Gy",
                            _volume_at_dose_from_values(pred_structure_dose, dose_threshold),
                            _volume_at_dose_from_values(ref_structure_dose, dose_threshold),
                            base,
                        )
                    )

                pred_dvh = _dose_volume_histogram_from_values(pred_structure_dose, config.dose_dvh_bin_width)
                ref_dvh = _dose_volume_histogram_from_values(ref_structure_dose, config.dose_dvh_bin_width)
                merged_dvh = pred_dvh.merge(ref_dvh, on="dose_gy", how="outer", suffixes=("_predicted", "_reference"))
                merged_dvh = merged_dvh.sort_values("dose_gy").fillna(0.0)
                for row in merged_dvh.to_dict("records"):
                    pred_volume = float(row["volume_percent_predicted"])
                    ref_volume = float(row["volume_percent_reference"])
                    dvh_rows.append(
                        {
                            **base,
                            "dose_gy": float(row["dose_gy"]),
                            "volume_percent_predicted": pred_volume,
                            "volume_percent_reference": ref_volume,
                            "volume_percent_difference": pred_volume - ref_volume,
                        }
                    )

    for patient_id, pred_paths in tqdm(predicted.items(), desc="Calculating dose metrics"):
        ref_lookup = _reference_lookup(references.get(patient_id, []))
        if len(ref_lookup) == 0:
            warnings.append(f"Skipping dose metrics for {patient_id}: no reference dose found")
            continue

        comparison_paths: list[tuple[Path, Path]] = []
        for pred_path in pred_paths:
            ref_path = ref_lookup.get(_normalised_image_key(pred_path))
            if ref_path is None:
                warnings.append(f"Skipping dose metrics for {pred_path}: no matching reference dose found")
                continue
            comparison_paths.append((pred_path, ref_path))

        if len(comparison_paths) == 0:
            continue

        reference_dose_cache: dict[Path, torch.Tensor] = {}
        structure_cache: dict[Path, torch.Tensor | None] = {}
        planning_ct_cache: dict[str, torch.Tensor] = {}
        planning_label_map_cache: dict[str, torch.Tensor] = {}
        generated_mask_cache: WarpedStructureMaskCache = shared_generated_mask_cache

        if metric_device.type == "cuda" and len(comparison_paths) > 1:
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="dose-loading") as executor:
                next_pred_path, next_ref_path = comparison_paths[0]
                next_future = executor.submit(
                    _load_dose_comparison_data,
                    next_pred_path,
                    next_ref_path,
                    patient_id,
                    dose_transform,
                    structure_transform,
                    generated_ct_lookup,
                    reference_dose_cache,
                    structure_cache,
                    planning_ct_cache,
                    planning_label_map_cache,
                    generated_mask_cache,
                    config,
                )

                for next_index in range(1, len(comparison_paths) + 1):
                    result = next_future.result()
                    if next_index < len(comparison_paths):
                        next_pred_path, next_ref_path = comparison_paths[next_index]
                        next_future = executor.submit(
                            _load_dose_comparison_data,
                            next_pred_path,
                            next_ref_path,
                            patient_id,
                            dose_transform,
                            structure_transform,
                            generated_ct_lookup,
                            reference_dose_cache,
                            structure_cache,
                            planning_ct_cache,
                            planning_label_map_cache,
                            generated_mask_cache,
                            config,
                        )

                    append_metrics_for_comparison(result, patient_id)

        else:
            for pred_path, ref_path in comparison_paths:
                result = _load_dose_comparison_data(
                    pred_path=pred_path,
                    ref_path=ref_path,
                    patient_id=patient_id,
                    dose_transform=dose_transform,
                    structure_transform=structure_transform,
                    generated_ct_lookup=generated_ct_lookup,
                    reference_dose_cache=reference_dose_cache,
                    structure_cache=structure_cache,
                    planning_ct_cache=planning_ct_cache,
                    planning_label_map_cache=planning_label_map_cache,
                    generated_mask_cache=generated_mask_cache,
                    config=config,
                )
                append_metrics_for_comparison(result, patient_id)

    for message in info_messages:
        logger.info(message)

    for message in warnings:
        logger.warning(message)

    if len(distribution_rows) == 0 and len(clinical_rows) == 0 and len(dvh_rows) == 0:
        logger.warning("Skipping dose metrics: no valid predicted-vs-reference dose comparisons were calculated")
        return

    if len(distribution_rows) > 0:
        distribution_df = pd.DataFrame(distribution_rows)
        distribution_path = config.metrics_dir / "dose_distribution_metrics.csv"
        distribution_df.to_csv(distribution_path, index=False)
        logger.info("Saved dose distribution metrics: %s", distribution_path)

        summary = distribution_df.groupby(["model_name", "patient_id", "structure_label", "metric"], dropna=False)[
            ["difference", "absolute_difference"]
        ].agg(["mean", "std", "min", "max", "count"])
        summary_path = config.metrics_dir / "dose_distribution_summary.csv"
        summary.to_csv(summary_path)
        logger.info("Saved dose distribution metric summary: %s", summary_path)

    if len(clinical_rows) > 0:
        clinical_df = pd.DataFrame(clinical_rows)
        clinical_path = config.metrics_dir / "dose_clinical_metrics.csv"
        clinical_df.to_csv(clinical_path, index=False)
        logger.info("Saved dose clinical metrics: %s", clinical_path)

        summary = clinical_df.groupby(["model_name", "patient_id", "structure_label", "metric"], dropna=False)[
            ["difference", "absolute_difference"]
        ].agg(["mean", "std", "min", "max", "count"])
        summary_path = config.metrics_dir / "dose_clinical_summary.csv"
        summary.to_csv(summary_path)
        logger.info("Saved dose clinical metric summary: %s", summary_path)

    if len(dvh_rows) > 0:
        dvh_df = pd.DataFrame(dvh_rows)
        dvh_path = config.metrics_dir / "dose_dvh.csv"
        dvh_df.to_csv(dvh_path, index=False)
        logger.info("Saved dose DVH rows: %s", dvh_path)


def evaluate_predicted_doses(
    config: MaisiTestingConfig,
    warped_mask_cache: WarpedStructureMaskCache | None = None,
) -> None:
    """
    Collect configured predicted/reference dose directories and run metrics.

    Missing prediction/reference folders are treated as a skipped optional
    baseline so the CT evaluation pipeline can run before dose predictors are
    available.

    Parameters
    ----------
    config : MaisiTestingConfig
        Pipeline configuration containing dose directories, metric settings,
        and output paths.

    warped_mask_cache : WarpedStructureMaskCache | None, optional
        Optional generated-space mask cache returned by structure metrics.
        Passing it avoids repeating planning-to-generated CT registration for
        dose structure metrics.
    """

    if not config.use_dose_metrics:
        logger.info("Skipping dose metrics: disabled by configuration")
        return

    if not config.predicted_dose_dir.exists():
        logger.warning("Skipping dose metrics: predicted dose directory does not exist: %s", config.predicted_dose_dir)
        return

    if not config.reference_dose_dir.exists():
        logger.warning("Skipping dose metrics: reference dose directory does not exist: %s", config.reference_dose_dir)
        return

    try:
        predicted = _collect_dose_distributions(config.predicted_dose_dir)
        references = _collect_dose_distributions(config.reference_dose_dir)
    except ValueError as exc:
        logger.warning("Skipping dose metrics: %s", exc)
        return

    calculate_dose_evaluation_metrics(
        predicted=predicted,
        references=references,
        config=config,
        model_name=config.dose_model_name,
        warped_mask_cache=warped_mask_cache,
    )
