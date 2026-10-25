import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from monai.metrics import compute_hausdorff_distance  # type: ignore[attr-defined]

from src.pipeline.config import MaisiTestingConfig
from src.pipeline.helpers.helpers import _is_planning_ct_path, _run_evaluation_tasks
from src.pipeline.helpers.helpers_structures import (
    WarpedStructureMaskCache,
    _as_binary_mask_pair,
    _calculate_structure_registration_result,
    _get_structure_label_transform,
    _label_to_binary_mask,
    _load_warped_structure_masks,
    _PatientContext,
    _PatientContextManager,
    _save_warped_structure_masks,
    _structure_path_for_ct,
    _StructureRegistrationResult,
    _warp_planning_mask_to_generated_ct,
)

logger = logging.getLogger(__name__)


def dice_coefficient(pred: torch.Tensor | np.ndarray, ref: torch.Tensor | np.ndarray) -> float:
    """
    Calculate Dice Similarity Coefficient (DSC) for two 3D masks.

    DSC measures spatial overlap between two binary structure masks. Non-zero
    values are treated as foreground. If both masks have no foreground voxels,
    the score is defined as 1.0 because the masks agree perfectly on
    foreground absence.

    Parameters
    ----------
    pred : torch.Tensor | np.ndarray
        Predicted, generated, or warped 3D structure mask.

    ref : torch.Tensor | np.ndarray
        Reference 3D structure mask.

    Returns
    -------
    dice : float
        Dice score in the range [0, 1], where 1 means perfect overlap.

    Raises
    ------
    ValueError
        If either mask has zero elements.
        If either mask is not 3-dimensional.
        If the masks have different shapes.
    """

    pred_t, ref_t = _as_binary_mask_pair(pred, ref)
    pred_sum = float(pred_t.sum().item())
    ref_sum = float(ref_t.sum().item())

    if pred_sum + ref_sum == 0.0:
        return 1.0

    return float((2.0 * (pred_t & ref_t).sum().item()) / (pred_sum + ref_sum))


def hausdorff_distance(
    pred: torch.Tensor | np.ndarray,
    ref: torch.Tensor | np.ndarray,
    spacing: tuple[float, float, float],
) -> float:
    """
    Calculate Hausdorff distance (HD) for a pair of 3D structure masks.

    Parameters
    ----------
    pred : torch.Tensor | np.ndarray
        Predicted, generated, or warped 3D structure mask.

    ref : torch.Tensor | np.ndarray
        Reference 3D structure mask.

    spacing : tuple[float, float, float]
        Physical voxel spacing for the mask axes.

    Returns
    -------
    HD : float
        The Hausdorff distance between the structures.

    Raises
    ------
    ValueError
        If either mask has zero elements.
        If either mask is not 3-dimensional.
        If the masks have different shapes.
    """

    pred_t, ref_t = _as_binary_mask_pair(pred, ref)

    # MONAI expects the first dimension to be batch and second to be class
    pred_t = pred_t.unsqueeze(0).unsqueeze(0)
    ref_t = ref_t.unsqueeze(0).unsqueeze(0)

    return float(compute_hausdorff_distance(pred_t, ref_t, spacing=spacing))


def calculate_structure_similarity_metrics(
    generated: dict[str, list[Path]],
    originals: dict[str, list[Path]],
    config: MaisiTestingConfig,
    warped_mask_cache: WarpedStructureMaskCache | None = None,
) -> WarpedStructureMaskCache:
    """
    Calculate generated-vs-real structure metrics.

    Generated CTs do not include model-predicted masks. For each generated CT,
    the planning CT is registered to the generated CT with a SimpleITK
    displacement-field transform, and planning structures are warped with
    nearest-neighbor interpolation. These generated-space masks are compared
    against the corresponding real fraction structure masks.

    Metrics:
    - Dice / DSC: higher means better structure overlap.
    - HD: lower means closer maximum boundary agreement.

    Outputs:
    - generated_vs_real_structure_metrics.csv
        Per-comparison, per-label structure metrics.
    - generated_vs_real_structure_summary.csv
        Patient- and label-level summary statistics.

    Parameters
    ----------
    generated : dict[str, list[Path]]
        Generated CT tensor paths grouped by patient ID.

    originals : dict[str, list[Path]]
        Original processed CT tensor paths grouped by patient ID. Each patient
        should contain the planning CT and one or more non-planning reference
        CTs.

    config : MaisiTestingConfig
        Pipeline configuration containing structure labels, spacing,
        registration settings, and output paths.

    warped_mask_cache : WarpedStructureMaskCache | None, optional
        Current run's index of generated CT paths to mask files on disk.
        Passing this index lets downstream dose metrics
        reuse the same generated-space target/OAR masks without repeating DVF
        registration or mask resampling.

    Returns
    -------
    warped_mask_cache : WarpedStructureMaskCache
        Paths to mask files written this run, keyed by generated CT path.
        Empty when no valid registrations are produced.

    Raises
    ------
    RuntimeError
        If structure metrics are configured to run tensor operations on CUDA
        but CUDA is not available.
    """

    if warped_mask_cache is None:
        warped_mask_cache = {}

    logger.info(
        "Calculating structure metrics: "
        f"{len(generated)} patient group(s), labels={config.structure_labels}, metric_device={config.device}"
    )

    rows: list[dict[str, Any]] = []
    info_messages: list[str] = []
    warnings: list[str] = []
    metric_device = torch.device(config.device)
    cpu_device = torch.device("cpu")
    if metric_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Structure metrics requested CUDA, but torch.cuda.is_available() is False")

    # Structure label maps use the same geometry as preprocessed CTs.
    # Build the label-preserving transform once for all structure files.
    structure_transform = _get_structure_label_transform(config)

    # Gather valid patient specs and task list using simple (patient_id, gen_path) tuples
    patient_specs: dict[str, dict[str, Any]] = {}
    tasks: list[tuple[str, Path]] = []

    for patient_id, gen_paths in generated.items():
        all_ref_paths = originals.get(patient_id, [])
        ref_paths = [path for path in all_ref_paths if not _is_planning_ct_path(path)]
        planning_paths = [path for path in all_ref_paths if _is_planning_ct_path(path)]

        if len(ref_paths) == 0 or len(planning_paths) == 0:
            warnings.append(f"Skipping structure metrics for {patient_id}: missing planning or non-planning CT")
            continue

        planning_ct_path = planning_paths[0]
        planning_structure_path = _structure_path_for_ct(planning_ct_path, config)

        if not planning_structure_path.exists():
            warnings.append(f"Skipping structure metrics for {patient_id}: missing {planning_structure_path}")
            continue

        valid_ref_paths: list[Path] = []
        for ref_path in ref_paths:
            ref_structure_path = _structure_path_for_ct(ref_path, config)
            if not ref_structure_path.exists():
                warnings.append(f"Skipping missing reference structure for {patient_id}: {ref_structure_path}")
                continue
            valid_ref_paths.append(ref_path)

        if len(valid_ref_paths) == 0:
            continue

        patient_specs[patient_id] = {
            "planning_ct_path": planning_ct_path,
            "planning_structure_path": planning_structure_path,
            "ref_paths": valid_ref_paths,
        }

        for gen_path in gen_paths:
            tasks.append((patient_id, gen_path))

    if len(tasks) == 0:
        for message in warnings:
            logger.warning(message)
        logger.warning("Skipping structure metrics: no valid structure comparisons were calculated")
        return warped_mask_cache

    ctx_mgr = _PatientContextManager(
        specs=patient_specs,
        config=config,
        structure_transform=structure_transform,
        cpu_device=cpu_device,
    )

    def prepare_registration(
        gen_path: Path,
        patient_id: str,
        planning_ct: torch.Tensor,
        config: MaisiTestingConfig,
    ) -> _StructureRegistrationResult:
        # Only completed writes are indexed; no disk lookup is needed here
        if gen_path in warped_mask_cache:
            return _StructureRegistrationResult(gen_path=gen_path, gen_ct=None, transform=None)
        return _calculate_structure_registration_result(gen_path, patient_id, planning_ct, config)

    def append_metrics_for_registration(
        result: _StructureRegistrationResult,
        current_ctx: _PatientContext,
    ) -> None:
        """
        Warp registered planning masks and append metric rows.

        Parameters
        ----------
        result : _StructureRegistrationResult
            Completed registration result for one generated CT.

        current_ctx : _PatientContext
            Loaded patient CPU context containing planning masks and reference maps.
        """

        if result.warning is not None:
            warnings.append(result.warning)
            return

        if result.info_message is not None:
            info_messages.append(result.info_message)

        cached_masks = _load_warped_structure_masks(config, result.gen_path, warped_mask_cache)
        has_cached_masks = cached_masks is not None
        if not has_cached_masks and (result.gen_ct is None or result.transform is None):
            warnings.append(
                f"Skipping structure metrics for {current_ctx.patient_id}:"
                + f" registration failed for {result.gen_path}"
            )
            return

        gen_ct = result.gen_ct
        transform = result.transform

        with torch.inference_mode():
            generated_masks = cached_masks if cached_masks is not None else {}
            for label in config.structure_labels:
                # Warp one planning label at a time.
                # Only the current predicted/reference masks are moved to metric_device,
                # so CUDA sees small boolean masks rather than full CT caches.
                pred_mask_cpu = generated_masks.get(label)
                if pred_mask_cpu is None:
                    if gen_ct is None or transform is None:
                        raise RuntimeError(
                            f"Cannot warp structure label {label} for {result.gen_path}: missing registration data"
                        )
                    pred_mask_cpu = _warp_planning_mask_to_generated_ct(
                        planning_mask=current_ctx.planning_masks[label],
                        generated_ct=gen_ct,
                        transform=transform,
                        config=config,
                    )
                    generated_masks[label] = pred_mask_cpu

                pred_mask = pred_mask_cpu.to(metric_device)

                for ref_path, ref_label_map in current_ctx.ref_label_maps.items():
                    ref_structure_path = _structure_path_for_ct(ref_path, config)
                    ref_mask = _label_to_binary_mask(ref_label_map, label).to(metric_device)
                    try:
                        dice = dice_coefficient(pred_mask, ref_mask)
                        hd = hausdorff_distance(pred_mask, ref_mask, spacing=config.spacing)
                    except ValueError as exc:
                        warnings.append(
                            f"Skipping invalid structure masks for {current_ctx.patient_id}, label {label}: {exc}"
                        )
                        del ref_mask
                        continue

                    rows.append(
                        {
                            "patient_id": current_ctx.patient_id,
                            "comparison_type": "generated_structure_vs_real_structure",
                            "generated_path": str(result.gen_path),
                            "reference_path": str(ref_path),
                            "planning_structure_path": str(current_ctx.planning_structure_path),
                            "reference_structure_path": str(ref_structure_path),
                            "structure_label": label,
                            "dice": dice,
                            "hd": hd,
                        }
                    )

            if not has_cached_masks:
                _save_warped_structure_masks(config, result.gen_path, generated_masks, warped_mask_cache)

    def prepare_task(task: tuple[str, Path]) -> tuple[_StructureRegistrationResult, _PatientContext]:
        patient_id, gen_path = task
        # Only the preparation worker owns the context manager. The current
        # metric callback holds its own context reference while the worker
        # loads the next patient, so at most two patient contexts are retained.
        ctx_mgr.retain_only({patient_id})
        context = ctx_mgr.get(patient_id)
        result = prepare_registration(gen_path, patient_id, context.planning_ct, config)
        return result, context

    def calculate_task_metrics(
        task: tuple[str, Path],
        prepared: tuple[_StructureRegistrationResult, _PatientContext],
    ) -> None:
        result, context = prepared
        append_metrics_for_registration(result, context)

    _run_evaluation_tasks(
        tasks,
        prepare_task,
        calculate_task_metrics,
        config.device,
        "Calculating generated vs real structure metrics",
    )

    for message in info_messages:
        logger.info(message)

    for message in warnings:
        logger.warning(message)

    if len(rows) == 0:
        logger.warning("Skipping structure metrics: no valid structure comparisons were calculated")
        return warped_mask_cache

    df = pd.DataFrame(rows)
    metrics_path = config.metrics_dir / "generated_vs_real_structure_metrics.csv"
    df.to_csv(metrics_path, index=False)
    logger.info("Saved structure metrics: %s", metrics_path)

    summary = df.groupby(["patient_id", "structure_label"])[["dice", "hd"]].agg(["mean", "std", "min", "max", "count"])
    summary_path = config.metrics_dir / "generated_vs_real_structure_summary.csv"
    summary.to_csv(summary_path)
    logger.info("Saved structure metric summary: %s", summary_path)

    return warped_mask_cache
