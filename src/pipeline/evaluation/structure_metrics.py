import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from tqdm import tqdm

from src.pipeline.config import MaisiTestingConfig
from src.pipeline.helpers.helpers import _is_planning_ct_path, _load_hu_tensor
from src.pipeline.helpers.helpers_structures import (
    WarpedStructureMaskCache,
    _as_binary_mask_pair,
    _calculate_structure_registration_result,
    _complete_warped_structure_masks,
    _get_structure_label_transform,
    _label_to_binary_mask,
    _load_structure_label_map,
    _load_warped_structure_masks,
    _PlanningRegistrationData,
    _prepare_planning_registration,
    _save_warped_structure_masks,
    _structure_path_for_ct,
    _StructureRegistrationResult,
    _surface_distances,
    _tensor_to_sitk_image,
    _warp_planning_labels_to_generated_ct,
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

    distances = _surface_distances(pred, ref, spacing)

    if isinstance(distances, float):
        return distances, distances

    return float(distances.max().item()), float(torch.quantile(distances, 0.95).item())


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
    - HD95: lower means closer robust boundary agreement.

    Outputs:
    - generated_vs_real_structure_metrics.csv
        Per-comparison, per-label structure metrics.
    - generated_vs_real_structure_summary.csv
        Patient- and label-level summary statistics.
    - warped structure mask cache files
        CPU `.pt` files under `config.warped_structure_cache_dir`, keyed by
        generated CT, for reuse by dose metrics in later independent runs.

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
        Optional CPU cache filled with planning masks warped into each
        generated CT space. Passing this cache lets downstream dose metrics
        reuse the same generated-space target/OAR masks without repeating DVF
        registration or mask resampling.

    Returns
    -------
    warped_mask_cache : WarpedStructureMaskCache
        CPU cache of warped generated-space masks keyed by generated CT path.
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
        planning_ct: torch.Tensor | None = None
        planning_registration: _PlanningRegistrationData | None = None

        ref_label_maps: dict[Path, torch.Tensor] = {}
        for ref_path in ref_paths:
            ref_structure_path = _structure_path_for_ct(ref_path, config)
            if not ref_structure_path.exists():
                warnings.append(f"Skipping missing reference structure for {patient_id}: {ref_structure_path}")
                continue

            ref_label_maps[ref_path] = _load_structure_label_map(ref_structure_path, structure_transform)

        if len(ref_label_maps) == 0:
            continue

        def load_registration(
            gen_path: Path,
            current_planning_ct_path: Path = planning_ct_path,
            current_patient_id: str = patient_id,
        ) -> _StructureRegistrationResult:
            """
            Resolve cached masks or calculate registration for one generated CT.

            Check run-local and persisted masks before loading CT data. On the
            first cache miss, prepare the patient's planning image and pyramid
            for reuse by subsequent generated CTs. This helper runs serially
            or in the single background registration worker.

            Parameters
            ----------
            gen_path : Path
                Generated CT tensor file to evaluate.

            current_planning_ct_path : Path, optional
                Planning CT source bound to the current patient.

            current_patient_id : str, optional
                Current patient identifier used in diagnostic messages.

            Returns
            -------
            result : _StructureRegistrationResult
                Complete cached masks, a generated CT and transform from a new
                registration, or a warning if the generated CT is skipped.
            """
            nonlocal planning_ct, planning_registration
            masks = warped_mask_cache.get(gen_path)
            if not _complete_warped_structure_masks(masks, config.structure_labels, tuple(config.target_image_size)):
                masks = _load_warped_structure_masks(config, gen_path, current_planning_ct_path)
            if masks is not None:
                return _StructureRegistrationResult(gen_path, None, None, masks=masks)
            if planning_ct is None:
                planning_ct = _load_hu_tensor(current_planning_ct_path, config=config, device=cpu_device)
                planning_registration = _prepare_planning_registration(planning_ct, config)
            return _calculate_structure_registration_result(
                gen_path, current_patient_id, planning_ct, config, planning_registration
            )

        def append_metrics_for_registration(
            result: _StructureRegistrationResult,
            current_patient_id: str,
            current_planning_structure_path: Path,
            current_ref_label_maps: dict[Path, torch.Tensor],
            current_planning_registration: _PlanningRegistrationData | None,
            current_planning_ct_path: Path = planning_ct_path,
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

            current_ref_label_maps : dict[Path, torch.Tensor]
                Reference structure label maps keyed by reference CT path.

            current_planning_registration : _PlanningRegistrationData | None
                Reusable planning images for uncached registrations, or
                ``None`` when cached masks make planning preparation unnecessary.
                The label image is initialized here on the first mask warp.

            current_planning_ct_path : Path, optional
                Planning CT source used to persist cache validation metadata.
            """

            if result.warning is not None:
                warnings.append(result.warning)
                return

            if result.info_message is not None:
                info_messages.append(result.info_message)

            generated_masks = result.masks
            with torch.inference_mode():
                if generated_masks is None:
                    if result.gen_ct is None or result.transform is None or current_planning_registration is None:
                        warnings.append(
                            f"Skipping structure metrics for {current_patient_id}: "
                            f"registration failed for {result.gen_path}"
                        )
                        return
                    if current_planning_registration.labels is None:
                        planning_label_map = _load_structure_label_map(
                            current_planning_structure_path, structure_transform
                        )
                        current_planning_registration.labels = _tensor_to_sitk_image(
                            planning_label_map, config, sitk.sitkInt16
                        )
                    generated_masks = _warp_planning_labels_to_generated_ct(
                        current_planning_registration.labels, result.gen_ct, result.transform, config
                    )
                    _save_warped_structure_masks(config, result.gen_path, generated_masks, current_planning_ct_path)
                warped_mask_cache[result.gen_path] = generated_masks
                for label in config.structure_labels:
                    # Only the current predicted/reference masks are moved to metric_device,
                    # so CUDA sees small boolean masks rather than full CT caches.
                    pred_mask_cpu = generated_masks[label]
                    pred_mask = pred_mask_cpu.to(metric_device)

                    for ref_path, ref_label_map in current_ref_label_maps.items():
                        ref_structure_path = _structure_path_for_ct(ref_path, config)
                        ref_mask = _label_to_binary_mask(ref_label_map, label).to(metric_device)
                        try:
                            dice = dice_coefficient(pred_mask, ref_mask)
                            hd, hd95_value = hausdorff_and_hd95(pred_mask, ref_mask, spacing=config.spacing)
                        except ValueError as exc:
                            warnings.append(
                                f"Skipping invalid structure masks for {current_patient_id}, label {label}: {exc}"
                            )
                            del ref_mask
                            continue

                        rows.append(
                            {
                                "patient_id": current_patient_id,
                                "comparison_type": "generated_structure_vs_real_structure",
                                "generated_path": str(result.gen_path),
                                "reference_path": str(ref_path),
                                "planning_structure_path": str(current_planning_structure_path),
                                "reference_structure_path": str(ref_structure_path),
                                "structure_label": label,
                                "dice": dice,
                                "hd": hd,
                                "hd95": hd95_value,
                            }
                        )

        if metric_device.type == "cuda" and len(gen_paths) > 1:
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="structure-registration") as executor:
                next_future = executor.submit(
                    load_registration,
                    gen_paths[0],
                )

                for next_index in range(1, len(gen_paths) + 1):
                    result = next_future.result()
                    if next_index < len(gen_paths):
                        next_future = executor.submit(
                            load_registration,
                            gen_paths[next_index],
                        )

                    append_metrics_for_registration(
                        result=result,
                        current_patient_id=patient_id,
                        current_planning_structure_path=planning_structure_path,
                        current_ref_label_maps=ref_label_maps,
                        current_planning_registration=planning_registration,
                    )

        else:
            for gen_path in gen_paths:
                result = load_registration(gen_path)
                append_metrics_for_registration(
                    result=result,
                    current_patient_id=patient_id,
                    current_planning_structure_path=planning_structure_path,
                    current_ref_label_maps=ref_label_maps,
                    current_planning_registration=planning_registration,
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

    summary = df.groupby(["patient_id", "structure_label"])[["dice", "hd", "hd95"]].agg(
        ["mean", "std", "min", "max", "count"]
    )
    summary_path = config.metrics_dir / "generated_vs_real_structure_summary.csv"
    summary.to_csv(summary_path)
    logger.info("Saved structure metric summary: %s", summary_path)

    return warped_mask_cache
