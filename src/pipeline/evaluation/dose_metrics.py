import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.pipeline.config import MaisiTestingConfig
from src.pipeline.helpers.helpers_doses import (
    _clinical_metric_values,
    _collect_single_dose_per_patient,
    _contains_dose_files,
    _DoseComparisonData,
    _evaluate_dose_on_generated_structures,
    _evaluate_dose_on_real_structures,
    _generated_ct_lookup,
    _get_dose_transform,
    _load_dose_comparison_data,
    _load_dose_distribution,
    _masked_dose_values,
    _masked_error_metrics_from_values,
    _metric_row,
    _planning_masks_for_dose,
    _write_scenario_outputs,
    _write_smoke_test_outputs,
)
from src.pipeline.helpers.helpers_structures import (
    WarpedStructureMaskCache,
    _get_structure_label_transform,
    _label_to_binary_mask,
)

logger = logging.getLogger(__name__)


def dose_volume_histogram(
    dose: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
    bin_width: float,
) -> pd.DataFrame:
    """Calculate a cumulative DVH for a masked volume.

    For each regularly spaced dose threshold, the cumulative DVH reports the
    percentage of selected structure voxels receiving at least that dose.

    Parameters
    ----------
    dose : torch.Tensor | np.ndarray
        Dose distribution in physical dose units, normally Gy.

    mask : torch.Tensor | np.ndarray
        Binary structure mask on the same grid as ``dose``.

    bin_width : float
        Positive spacing between consecutive dose thresholds, in the same
        units as ``dose``.

    Returns
    -------
    dvh : pd.DataFrame
        Table with ``dose_gy`` and cumulative ``volume_percent`` columns. An
        empty mask returns an empty table with those columns.

    Raises
    ------
    ValueError
        If ``bin_width`` is not positive or the dose and mask shapes differ.
    """

    if bin_width <= 0:
        raise ValueError("`bin_width` must be greater than 0")

    structure_dose = _masked_dose_values(dose, mask)
    if structure_dose.numel() == 0:
        return pd.DataFrame(columns=["dose_gy", "volume_percent"])

    max_value = float(structure_dose.max().item())
    bins = torch.arange(0.0, max_value + bin_width, bin_width, dtype=torch.float32, device=structure_dose.device)
    sorted_dose = torch.sort(structure_dose).values
    first_ge_indices = torch.searchsorted(sorted_dose, bins, right=False)
    volume_percent = (structure_dose.numel() - first_ge_indices).float() / structure_dose.numel() * 100.0
    return pd.DataFrame(
        {
            "dose_gy": bins.detach().cpu().numpy(),
            "volume_percent": volume_percent.detach().cpu().numpy(),
        }
    )


def dose_at_volume(
    dose: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
    volume_percent: float,
) -> float:
    """Calculate Dx, the dose received by at least x percent of a volume.

    Dx is evaluated as the lower-tail quantile ``(100 - x) / 100`` using
    linear interpolation. For example, D95 is the fifth percentile of the
    selected dose values.

    Parameters
    ----------
    dose : torch.Tensor | np.ndarray
        Dose distribution in physical dose units, normally Gy.

    mask : torch.Tensor | np.ndarray
        Binary structure mask on the same grid as ``dose``.

    volume_percent : float
        Percentage of the structure that must receive at least the returned
        dose. Must lie in ``[0, 100]``.

    Returns
    -------
    dose_value : float
        Dx in the same units as ``dose``. An empty mask returns ``NaN``.

    Raises
    ------
    ValueError
        If ``volume_percent`` is outside ``[0, 100]`` or the dose and mask
        shapes differ.
    """

    if volume_percent < 0 or volume_percent > 100:
        raise ValueError(f"`volume_percent` must be in [0, 100]. Got {volume_percent}")

    structure_dose = _masked_dose_values(dose, mask)
    if structure_dose.numel() == 0:
        return float("nan")

    quantile = (100.0 - volume_percent) / 100.0
    rank = quantile * (structure_dose.numel() - 1)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    lower_value = structure_dose.kthvalue(lower_index + 1).values
    if lower_index == upper_index:
        return float(lower_value.item())

    upper_value = structure_dose.kthvalue(upper_index + 1).values
    interpolated = lower_value + (upper_value - lower_value) * (rank - lower_index)
    return float(interpolated.item())


def volume_at_dose(
    dose: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
    dose_threshold: float,
) -> float:
    """Calculate Vx, the percentage of a volume receiving at least x dose.

    Parameters
    ----------
    dose : torch.Tensor | np.ndarray
        Dose distribution in physical dose units, normally Gy.

    mask : torch.Tensor | np.ndarray
        Binary structure mask on the same grid as ``dose``.

    dose_threshold : float
        Inclusive dose threshold in the same units as ``dose``.

    Returns
    -------
    volume_percent : float
        Percentage of selected voxels for which dose is greater than or equal
        to ``dose_threshold``. An empty mask returns ``NaN``.

    Raises
    ------
    ValueError
        If the dose and mask shapes differ.
    """

    structure_dose = _masked_dose_values(dose, mask)
    if structure_dose.numel() == 0:
        return float("nan")
    return float((structure_dose >= dose_threshold).float().mean().item() * 100.0)


def mean_dose(dose: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray) -> float:
    """Calculate the arithmetic mean dose inside a structure mask.

    Parameters
    ----------
    dose : torch.Tensor | np.ndarray
        Dose distribution in physical dose units, normally Gy.

    mask : torch.Tensor | np.ndarray
        Binary structure mask on the same grid as ``dose``.

    Returns
    -------
    mean_value : float
        Mean selected dose in the same units as ``dose``. An empty mask
        returns ``NaN``.

    Raises
    ------
    ValueError
        If the dose and mask shapes differ.
    """

    structure_dose = _masked_dose_values(dose, mask)
    if structure_dose.numel() == 0:
        return float("nan")
    return float(structure_dose.mean().item())


def maximum_dose(dose: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray) -> float:
    """Calculate the maximum dose inside a structure mask.

    Parameters
    ----------
    dose : torch.Tensor | np.ndarray
        Dose distribution in physical dose units, normally Gy.

    mask : torch.Tensor | np.ndarray
        Binary structure mask on the same grid as ``dose``.

    Returns
    -------
    maximum_value : float
        Maximum selected dose in the same units as ``dose``. An empty mask
        returns ``NaN``.

    Raises
    ------
    ValueError
        If the dose and mask shapes differ.
    """

    structure_dose = _masked_dose_values(dose, mask)
    if structure_dose.numel() == 0:
        return float("nan")
    return float(structure_dose.max().item())


def evaluate_original_anatomy_comparison(config: MaisiTestingConfig) -> None:
    """Compare patient-level candidate and clinical doses on fraction-1 structures.

    For every patient with an unambiguous candidate and clinical planning
    dose, calculate the configured clinical metrics using the original
    fraction-1 structure masks. Exactly one candidate dose and one clinical
    fraction-1 dose are accepted per patient; duplicates raise ``ValueError``.
    Detailed values and differences are written to
    ``original_anatomy_comparison.csv`` in ``config.metrics_dir``. Patients
    with missing candidate doses, masks, or incompatible dose shapes are
    skipped.

    Parameters
    ----------
    config : MaisiTestingConfig
        Pipeline configuration containing dose directories, structure labels,
        clinical metric settings, preprocessing options, and the output path.
    """

    candidates = _collect_single_dose_per_patient(config.predicted_dose_dir, planning_only=False)
    clinical_doses = _collect_single_dose_per_patient(config.dose_root, planning_only=True)
    dose_transform = _get_dose_transform(config)
    structure_transform = _get_structure_label_transform(config)
    rows: list[dict[str, Any]] = []
    for patient_id, clinical_path in clinical_doses.items():
        candidate_path = candidates.get(patient_id)
        if candidate_path is None:
            logger.warning("Skipping original anatomy comparison for %s: no candidate dose found", patient_id)
            continue
        clinical = _load_dose_distribution(clinical_path, dose_transform)
        candidate = _load_dose_distribution(candidate_path, dose_transform)
        masks = _planning_masks_for_dose(clinical_path, structure_transform, config)
        if masks is None or clinical.shape != candidate.shape:
            continue
        for label, mask in masks.items():
            clinical_metrics = _clinical_metric_values(clinical, mask, config)
            candidate_metrics = _clinical_metric_values(candidate, mask, config)
            for metric, candidate_value in candidate_metrics.items():
                clinical_value = clinical_metrics[metric]
                rows.append(
                    {
                        "patient_id": patient_id,
                        "structure_label": label,
                        "metric": metric,
                        "candidate_value": candidate_value,
                        "clinical_fraction_1_value": clinical_value,
                        "difference": candidate_value - clinical_value,
                        "absolute_difference": abs(candidate_value - clinical_value),
                        "candidate_dose_path": str(candidate_path),
                        "clinical_dose_path": str(clinical_path),
                        "structure_source": "original_fraction_1_structure",
                    }
                )
    if rows:
        pd.DataFrame(rows).to_csv(config.metrics_dir / "original_anatomy_comparison.csv", index=False)
    else:
        logger.warning("Skipping original anatomy comparison: no patient-level candidate doses were found")


def evaluate_scenario_robustness(
    config: MaisiTestingConfig,
    warped_mask_cache: WarpedStructureMaskCache | None = None,
) -> None:
    """Evaluate configured candidate doses across real and generated anatomies.

    Each patient's single candidate dose is held fixed while metrics are
    evaluated on observed real-fraction and generated-anatomy structure masks.
    If a patient has no candidate dose, its clinical fraction-1 dose is used as
    a fallback. Detailed and across-scenario summary CSV files are written
    under ``config.metrics_dir``.

    Parameters
    ----------
    config : MaisiTestingConfig
        Pipeline configuration containing predicted and reference dose paths,
        generated CT paths, structure labels, and clinical metric settings.

    warped_mask_cache : WarpedStructureMaskCache | None, optional
        Reusable generated-space masks keyed by generated CT path. If omitted,
        a new cache is created for this evaluation.
    """

    cache = warped_mask_cache if warped_mask_cache is not None else {}
    references = _collect_single_dose_per_patient(config.dose_root, planning_only=True)
    predicted = (
        _collect_single_dose_per_patient(config.predicted_dose_dir, planning_only=False)
        if _contains_dose_files(config.predicted_dose_dir)
        else {}
    )
    rows = _evaluate_dose_on_real_structures(predicted, references, config, "candidate")
    rows.extend(_evaluate_dose_on_generated_structures(predicted, references, config, "candidate", cache))
    _write_scenario_outputs(rows, config, "scenario_robustness")


def evaluate_base_dose_smoke_test(
    config: MaisiTestingConfig,
    warped_mask_cache: WarpedStructureMaskCache | None = None,
) -> None:
    """Evaluate the clinical fraction-1 dose on all real and generated structures.

    The numerical dose is held fixed while structure masks come from the
    original fraction, observed follow-up fractions, and generated anatomies.
    This calculation does not require predicted doses. It writes detailed and
    summary CSV files under ``config.metrics_dir``.

    Parameters
    ----------
    config : MaisiTestingConfig
        Pipeline configuration containing reference dose paths, generated CT
        paths, structure labels, clinical metric settings, and output paths.

    warped_mask_cache : WarpedStructureMaskCache | None, optional
        Reusable generated-space masks keyed by generated CT path. If omitted,
        a new cache is created for this evaluation.
    """

    cache = warped_mask_cache if warped_mask_cache is not None else {}
    references = _collect_single_dose_per_patient(config.dose_root, planning_only=True)
    # Keep the clinical planning dose fixed while evaluating original, observed
    # follow-up, and generated-anatomy structure masks
    rows = _evaluate_dose_on_real_structures(
        references,
        references,
        config,
        "base_fraction_1",
        force_clinical=True,
    )
    rows.extend(_evaluate_dose_on_generated_structures(references, references, config, "base_fraction_1", cache))
    _write_smoke_test_outputs(rows, config)


def calculate_dose_evaluation_metrics(
    predicted: dict[str, Path],
    references: dict[str, Path],
    config: MaisiTestingConfig,
    warped_mask_cache: WarpedStructureMaskCache | None = None,
) -> None:
    """
    Calculate dose evaluation metrics for predicted-vs-reference doses.

    One predicted dose is paired with one clinical fraction-1 dose per patient.
    Dose files and structure label maps are loaded on CPU. Metric reductions
    run on `config.device`, so CUDA can accelerate masked dose errors, Dx/Vx,
    mean/max dose, and DVH binning.

    Outputs:
    - `dose_distribution_metrics.csv`: voxel, mean-dose, and max-dose rows.
    - `dose_clinical_metrics.csv`: Dx and Vx rows.
    - `dose_dvh.csv`: cumulative DVH rows for plotting or downstream analysis.
    - summary CSVs grouped by patient, structure label, and metric.

    Parameters
    ----------
    predicted : dict[str, Path]
        Single predicted dose path keyed by patient ID.

    references : dict[str, Path]
        Single clinical fraction-1 dose path keyed by patient ID.

    config : MaisiTestingConfig
        Pipeline configuration containing metric device, structure labels,
        dose metric settings, and output paths.

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
    # The cache may have been populated by structure metrics earlier in the
    # pipeline. Reusing it avoids repeating expensive deformable registration.
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
        f"{len(predicted)} predicted dose(s), "
        f"{len(references)} reference dose(s), "
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
            # Always report whole-volume metrics. Structure-specific entries
            # are appended only when a valid reference label map is available.
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

                # Voxel-wise errors require positional correspondence. Warped
                # and reference masks can select different voxel sets, while
                # aggregate clinical metrics remain meaningful for both sets.
                if torch.equal(pred_mask, ref_mask):
                    voxel_mae, voxel_rmse = _masked_error_metrics_from_values(
                        pred_structure_dose,
                        ref_structure_dose,
                    )
                else:
                    voxel_mae, voxel_rmse = float("nan"), float("nan")

                base = {
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
                            mean_dose(pred_dose, pred_mask),
                            mean_dose(ref_dose, ref_mask),
                            base,
                        ),
                        _metric_row(
                            "maximum_dose",
                            maximum_dose(pred_dose, pred_mask),
                            maximum_dose(ref_dose, ref_mask),
                            base,
                        ),
                    ]
                )

                for volume_percent in config.dose_dx_volume_percents:
                    clinical_rows.append(
                        _metric_row(
                            f"D{volume_percent:g}",
                            dose_at_volume(pred_dose, pred_mask, volume_percent),
                            dose_at_volume(ref_dose, ref_mask, volume_percent),
                            base,
                        )
                    )

                for dose_threshold in config.dose_vx_thresholds:
                    clinical_rows.append(
                        _metric_row(
                            f"V{dose_threshold:g}Gy",
                            volume_at_dose(pred_dose, pred_mask, dose_threshold),
                            volume_at_dose(ref_dose, ref_mask, dose_threshold),
                            base,
                        )
                    )

                pred_dvh = dose_volume_histogram(pred_dose, pred_mask, config.dose_dvh_bin_width)
                ref_dvh = dose_volume_histogram(ref_dose, ref_mask, config.dose_dvh_bin_width)
                # Outer alignment preserves bins present in only one curve;
                # missing cumulative volume is zero beyond that curve's range.
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

    for patient_id, pred_path in tqdm(predicted.items(), desc="Calculating dose metrics"):
        ref_path = references.get(patient_id)
        if ref_path is None:
            warnings.append(f"Skipping dose metrics for {patient_id}: no reference dose found")
            continue

        # Keep loader inputs patient-scoped. The shared generated-mask cache can
        # still reuse structure registration performed by an earlier workflow.
        reference_dose_cache: dict[Path, torch.Tensor] = {}
        structure_cache: dict[Path, torch.Tensor | None] = {}
        planning_ct_cache: dict[str, torch.Tensor] = {}
        planning_label_map_cache: dict[str, torch.Tensor] = {}
        generated_mask_cache: WarpedStructureMaskCache = shared_generated_mask_cache

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

        # Keep the whole-volume group (`structure_label=None`) in summaries
        summary = distribution_df.groupby(["patient_id", "structure_label", "metric"], dropna=False)[
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

        # Use the same grouping contract as distribution metrics so downstream
        # consumers can join both summary tables directly
        summary = clinical_df.groupby(["patient_id", "structure_label", "metric"], dropna=False)[
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

    Directory availability and configuration are validated once by the
    metrics orchestrator before this calculation function is called.

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

    predicted = _collect_single_dose_per_patient(config.predicted_dose_dir, planning_only=False)
    references = _collect_single_dose_per_patient(config.dose_root, planning_only=True)

    calculate_dose_evaluation_metrics(
        predicted=predicted,
        references=references,
        config=config,
        warped_mask_cache=warped_mask_cache,
    )
