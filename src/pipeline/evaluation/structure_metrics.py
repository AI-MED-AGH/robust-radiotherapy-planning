import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.pipeline.config import MaisiTestingConfig
from src.pipeline.helpers.helpers import (
    _as_binary_mask_pair,
    _calculate_structure_registration_result,
    _get_structure_label_transform,
    _is_planning_ct_path,
    _label_to_binary_mask,
    _load_hu_tensor,
    _load_structure_label_map,
    _structure_path_for_ct,
    _StructureRegistrationResult,
    _surface_distances,
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


def hausdorff_and_hd95(
    pred: torch.Tensor | np.ndarray,
    ref: torch.Tensor | np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[float, float]:
    """
    Calculate HD and HD95 from a single surface-distance pass.

    HD is the maximum symmetric surface distance. HD95 is the 95th percentile
    of the same symmetric surface-distance distribution, making it less
    sensitive to isolated outlier voxels. Calculating both together avoids
    recomputing surfaces and pairwise nearest-neighbor distances.

    Foreground-empty mask convention:
    - Both masks have no foreground voxels -> ``(0.0, 0.0)``.
    - Only one mask has no foreground voxels -> ``(inf, inf)``.

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
    metrics : tuple[float, float]
        Pair containing symmetric Hausdorff distance and HD95 in physical
        units.

    Raises
    ------
    ValueError
        If either mask has zero elements.
        If either mask is not 3-dimensional.
        If the masks have different shapes.
    """

    # Surface extraction and nearest-neighbor distance search happen once
    distances = _surface_distances(pred, ref, spacing)

    if isinstance(distances, float):
        return distances, distances

    # Read HD and HD95 from the shared distance tensor
    return float(distances.max().item()), float(torch.quantile(distances, 0.95).item())


def calculate_structure_similarity_metrics(
    generated: dict[str, list[Path]],
    originals: dict[str, list[Path]],
    config: MaisiTestingConfig,
) -> None:
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
    - HD95: lower means closer robust boundary agreement.

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

    Raises
    ------
    RuntimeError
        If structure metrics are configured to run tensor operations on CUDA
        but CUDA is not available.
    """

    if not config.use_structure_metrics:
        logger.info("Skipping structure metrics: disabled by configuration")
        return

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

    for patient_id, gen_paths in tqdm(generated.items(), desc="Calculating generated vs real structure metrics"):
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

        # Registration and mask warping go through SimpleITK.
        # Keep CT volumes on CPU instead of caching patient CTs in VRAM.
        planning_ct = _load_hu_tensor(planning_ct_path, config=config, device=cpu_device)
        planning_label_map = _load_structure_label_map(planning_structure_path, structure_transform)
        planning_masks = {label: _label_to_binary_mask(planning_label_map, label) for label in config.structure_labels}

        ref_label_maps: dict[Path, torch.Tensor] = {}
        for ref_path in ref_paths:
            ref_structure_path = _structure_path_for_ct(ref_path, config)
            if not ref_structure_path.exists():
                warnings.append(f"Skipping missing reference structure for {patient_id}: {ref_structure_path}")
                continue

            ref_label_maps[ref_path] = _load_structure_label_map(ref_structure_path, structure_transform)

        if len(ref_label_maps) == 0:
            continue

        def append_metrics_for_registration(
            result: _StructureRegistrationResult,
            current_patient_id: str,
            current_planning_structure_path: Path,
            current_planning_masks: dict[int, torch.Tensor],
            current_ref_label_maps: dict[Path, torch.Tensor],
        ) -> None:
            """
            Warp registered planning masks and append metric rows.

            This consumes a completed CPU registration result. The generated
            CT and transform stay on CPU for SimpleITK mask warping, while
            only the current predicted and reference masks are moved to the
            configured metric device.

            Parameters
            ----------
            result : _StructureRegistrationResult
                Completed registration result for one generated CT.

            current_patient_id : str
                Patient identifier written to output rows and warnings.

            current_planning_structure_path : Path
                Path to the planning structure label map written to output
                rows.

            current_planning_masks : dict[int, torch.Tensor]
                Planning CT masks keyed by structure label.

            current_ref_label_maps : dict[Path, torch.Tensor]
                Reference structure label maps keyed by reference CT path.
            """

            if result.warning is not None:
                warnings.append(result.warning)
                return

            if result.info_message is not None:
                info_messages.append(result.info_message)

            if result.gen_ct is None or result.transform is None:
                warnings.append(
                    f"Skipping structure metrics for {current_patient_id}: registration failed for {result.gen_path}"
                )
                return

            gen_ct = result.gen_ct
            transform = result.transform

            with torch.inference_mode():
                for label in config.structure_labels:
                    # Warp one planning label at a time.
                    # Only the current predicted/reference masks are moved to metric_device,
                    # so CUDA sees small boolean masks rather than full CT caches.
                    pred_mask = _warp_planning_mask_to_generated_ct(
                        planning_mask=current_planning_masks[label],
                        generated_ct=gen_ct,
                        transform=transform,
                        config=config,
                    ).to(metric_device)

                    for ref_path, ref_label_map in current_ref_label_maps.items():
                        ref_structure_path = _structure_path_for_ct(ref_path, config)
                        ref_mask = _label_to_binary_mask(ref_label_map, label).to(metric_device)
                        if pred_mask.shape != ref_mask.shape:
                            warnings.append(
                                "Skipping structure shape mismatch for "
                                f"{current_patient_id}, label {label}: "
                                f"{tuple(pred_mask.shape)} vs {tuple(ref_mask.shape)}"
                            )
                            del ref_mask
                            continue

                        hd, hd95_value = hausdorff_and_hd95(pred_mask, ref_mask, spacing=config.spacing)
                        rows.append(
                            {
                                "patient_id": current_patient_id,
                                "comparison_type": "generated_structure_vs_real_structure",
                                "generated_path": str(result.gen_path),
                                "reference_path": str(ref_path),
                                "planning_structure_path": str(current_planning_structure_path),
                                "reference_structure_path": str(ref_structure_path),
                                "structure_label": label,
                                "dice": dice_coefficient(pred_mask, ref_mask),
                                "hd": hd,
                                "hd95": hd95_value,
                            }
                        )

        if metric_device.type == "cuda" and len(gen_paths) > 1:
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="structure-registration") as executor:
                next_future = executor.submit(
                    _calculate_structure_registration_result,
                    gen_paths[0],
                    patient_id,
                    planning_ct,
                    config,
                )

                for next_index in range(1, len(gen_paths) + 1):
                    result = next_future.result()
                    if next_index < len(gen_paths):
                        next_future = executor.submit(
                            _calculate_structure_registration_result,
                            gen_paths[next_index],
                            patient_id,
                            planning_ct,
                            config,
                        )

                    append_metrics_for_registration(
                        result=result,
                        current_patient_id=patient_id,
                        current_planning_structure_path=planning_structure_path,
                        current_planning_masks=planning_masks,
                        current_ref_label_maps=ref_label_maps,
                    )

        else:
            for gen_path in gen_paths:
                result = _calculate_structure_registration_result(
                    gen_path=gen_path,
                    patient_id=patient_id,
                    planning_ct=planning_ct,
                    config=config,
                )
                append_metrics_for_registration(
                    result=result,
                    current_patient_id=patient_id,
                    current_planning_structure_path=planning_structure_path,
                    current_planning_masks=planning_masks,
                    current_ref_label_maps=ref_label_maps,
                )

    for message in info_messages:
        logger.info(message)

    for message in warnings:
        logger.warning(message)

    if len(rows) == 0:
        logger.warning("Skipping structure metrics: no valid structure comparisons were calculated")
        return

    df = pd.DataFrame(rows)
    metrics_path = config.metrics_dir / "generated_vs_real_structure_metrics.csv"
    df.to_csv(metrics_path, index=False)
    logger.info("Saved structure metrics: %s", metrics_path)

    summary = df.groupby(["patient_id", "structure_label"])[["dice", "hd", "hd95"]].agg(
        ["mean", "std", "min", "max", "count"]
    )
    summary_path = config.metrics_dir / "generated_vs_real_structure_summary.csv"
    summary.to_csv(summary_path)
    logger.info("Saved structure metric summary: %s", summary_path)
