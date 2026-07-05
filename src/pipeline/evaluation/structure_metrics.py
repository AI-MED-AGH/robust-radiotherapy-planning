import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from monai.data.meta_tensor import MetaTensor
from monai.transforms import (  # type: ignore[attr-defined]
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    SpatialPadd,
)
from tqdm import tqdm

from src.pipeline.config import MaisiTestingConfig
from src.pipeline.helpers.helpers import (
    _as_binary_mask_pair,
    _is_planning_ct_path,
    _load_hu_tensor,
    _multiscale_demons,
    _nearest_distances,
    _sitk_image_to_tensor,
    _structure_path_for_ct,
    _surface_voxels,
    _tensor_to_sitk_image,
)

logger = logging.getLogger(__name__)


def get_structure_label_transform(config: MaisiTestingConfig) -> Compose:
    """
    Create the preprocessing transform for structure label maps.

    This mirrors the geometric part of CT preprocessing but deliberately skips
    intensity scaling. Structure masks are categorical labels, so interpolation
    or HU scaling would corrupt the label values. The transform is built once
    per structure-metric run and reused for all label maps.

    Parameters
    ----------
    config : MaisiTestingConfig
        Pipeline configuration containing target image size.

    Returns
    -------
    transform : Compose
        MONAI dictionary transform that loads a label-map NIfTI file and
        returns an integer tensor on the pipeline target grid.
    """

    return Compose(
        [
            LoadImaged(keys="label"),
            EnsureChannelFirstd(keys="label"),
            Orientationd(keys="label", axcodes="RAS", labels=None),
            SpatialPadd(
                keys="label",
                spatial_size=config.target_image_size,
                mode="constant",
                constant_values=0,
            ),
            CenterSpatialCropd(keys="label", roi_size=config.target_image_size),
            EnsureTyped(keys="label", dtype=torch.int16),
        ]
    )


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


def _surface_distances(
    pred: torch.Tensor | np.ndarray,
    ref: torch.Tensor | np.ndarray,
    spacing: tuple[float, float, float],
) -> torch.Tensor | float:
    """
    Calculate symmetric foreground-surface distances for two masks.

    Surface voxels are extracted from both masks, converted to physical
    coordinates using ``spacing``, and compared in both directions with
    batched nearest-neighbor distances. Foreground-empty masks follow the
    metric convention used by the structure evaluation: both foreground-empty
    masks return ``0.0`` and one foreground-empty mask returns positive
    infinity.

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
    distances : torch.Tensor | float
        Symmetric surface distances for masks with foreground voxels, or a
        scalar foreground-empty convention value.

    Raises
    ------
    ValueError
        If either mask has zero elements.
        If either mask is not 3-dimensional.
        If the masks have different shapes.
    """

    pred_t, ref_t = _as_binary_mask_pair(pred, ref)
    pred_empty = int(pred_t.sum().item()) == 0
    ref_empty = int(ref_t.sum().item()) == 0

    if pred_empty and ref_empty:
        return 0.0

    if pred_empty or ref_empty:
        return float("inf")

    pred_surface = _surface_voxels(pred_t)
    ref_surface = _surface_voxels(ref_t)

    # Work with surface coordinates rather than all foreground voxels. This is
    # both the Hausdorff definition and much smaller than dense mask distances.
    pred_points = pred_surface.nonzero().float()
    ref_points = ref_surface.nonzero().float()
    spacing_t = torch.as_tensor(spacing, dtype=torch.float32, device=pred_points.device)
    pred_points = pred_points * spacing_t
    ref_points = ref_points * spacing_t

    # Compute both directions so HD is symmetric. _nearest_distances batches
    # cdist internally to avoid one huge [N_surface, M_surface] allocation.
    distances = torch.cat(
        [
            _nearest_distances(pred_points, ref_points),
            _nearest_distances(ref_points, pred_points),
        ]
    )

    return distances


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

    # HD and HD95 share the same surface-distance tensor, avoiding a second
    # surface extraction and nearest-neighbor pass for every mask comparison.
    return float(distances.max().item()), float(torch.quantile(distances, 0.95).item())


def load_structure_label_map(path: Path, transform: Compose) -> torch.Tensor:
    """
    Load and preprocess a structure label map to the pipeline target grid.

    The transform mirrors the CT preprocessing geometry: load image, enforce
    channel-first format, reorient to RAS, pad/crop to the configured target
    image size, and return an integer label tensor.

    Parameters
    ----------
    path : Path
        Path to a structure label-map NIfTI file.

    transform : Compose
        Label-map preprocessing transform returned by
        ``get_structure_label_transform``.

    Returns
    -------
    label_map : torch.Tensor
        Integer 3D label map on CPU with shape matching
        ``config.target_image_size``.
    """

    transformed = transform({"label": str(path)})
    label_meta: MetaTensor = transformed["label"]
    label = label_meta.as_tensor().detach().cpu()[0]

    return label.round().to(torch.int16)


def label_to_binary_mask(label_map: torch.Tensor, label: int) -> torch.Tensor:
    """
    Extract a binary structure mask from a label map.

    Label 0 is treated as an aggregate foreground request and returns all
    non-zero labels. Positive labels return only that exact label value.

    Parameters
    ----------
    label_map : torch.Tensor
        3D integer structure label map.

    label : int
        Structure label to extract. Use 0 for any non-background structure.

    Returns
    -------
    mask : torch.Tensor
        Boolean 3D mask for the requested structure.
    """

    if label <= 0:
        return label_map != 0

    return label_map == label


def register_planning_ct_to_generated_ct(
    planning_ct: torch.Tensor,
    generated_ct: torch.Tensor,
    config: MaisiTestingConfig,
) -> Any:
    """
    Register a planning CT to a generated CT.

    The generated CT is used as the fixed image and the planning CT as the
    moving image. For SimpleITK resampling, the returned displacement-field
    transform maps generated/fixed-grid physical points back into
    planning/moving CT space. Reusing that transform lets each planning
    structure mask be sampled onto the generated CT grid.

    Parameters
    ----------
    planning_ct : torch.Tensor
        Planning CT in HU-like units.

    generated_ct : torch.Tensor
        Generated CT in HU-like units.

    config : MaisiTestingConfig
        Pipeline configuration containing spacing and demons-registration
        settings.

    Returns
    -------
    transform : Any
        SimpleITK displacement-field transform mapping generated/fixed-grid
        points into planning/moving CT space for resampling.
    """

    # SimpleITK registration runs on CPU. Callers keep CT tensors on CPU here
    # so we do not reserve VRAM for volumes that are immediately converted to
    # SimpleITK images.
    fixed = _tensor_to_sitk_image(generated_ct, config, sitk.sitkFloat32)
    moving = _tensor_to_sitk_image(planning_ct, config, sitk.sitkFloat32)

    demons_filter = sitk.DiffeomorphicDemonsRegistrationFilter()  # type: ignore[no-untyped-call]
    demons_filter.SetNumberOfIterations(config.structure_registration_iterations)  # type: ignore[no-untyped-call]
    demons_filter.SetSmoothDisplacementField(True)  # type: ignore[no-untyped-call]
    demons_filter.SetStandardDeviations(config.structure_registration_sigma)  # type: ignore[no-untyped-call]

    initial_transform = sitk.CenteredTransformInitializer(  # type: ignore[no-untyped-call]
        fixed,
        moving,
        sitk.Euler3DTransform(),  # type: ignore[no-untyped-call]
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    # The multiscale helper rasterizes the initial rigid transform into a DVF,
    # then refines that field from coarse to full resolution.
    return _multiscale_demons(
        registration_algorithm=demons_filter,
        fixed_image=fixed,
        moving_image=moving,
        initial_transform=initial_transform,
        shrink_factors=config.structure_registration_shrink_factors,
        smoothing_sigmas=config.structure_registration_smoothing_sigmas,
    )


def warp_planning_mask_to_generated_ct(
    planning_mask: torch.Tensor,
    generated_ct: torch.Tensor,
    transform: Any,
    config: MaisiTestingConfig,
) -> torch.Tensor:
    """
    Warp a planning structure mask into generated CT space.

    The transform maps each generated-grid output point back into planning
    mask space, which is the direction expected by ``sitk.Resample``.
    Nearest-neighbor interpolation is used to preserve binary mask labels.

    Parameters
    ----------
    planning_mask : torch.Tensor
        Boolean or binary 3D structure mask in planning CT space.

    generated_ct : torch.Tensor
        Generated CT tensor used as the resampling reference grid.

    transform : Any
        SimpleITK transform returned by
        ``register_planning_ct_to_generated_ct``. It maps generated/fixed-grid
        points into planning/moving mask space.

    config : MaisiTestingConfig
        Pipeline configuration containing spacing and image-grid settings.

    Returns
    -------
    warped_mask : torch.Tensor
        Boolean 3D mask in generated CT space.
    """

    # The generated CT is used only as the reference grid: size, spacing,
    # origin, and direction define where the warped planning mask is sampled.
    fixed = _tensor_to_sitk_image(generated_ct, config, sitk.sitkFloat32)
    moving_mask = _tensor_to_sitk_image(planning_mask.to(torch.uint8), config, sitk.sitkUInt8)

    resampler = sitk.ResampleImageFilter()  # type: ignore[no-untyped-call]
    resampler.SetReferenceImage(fixed)  # type: ignore[no-untyped-call]
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # type: ignore[no-untyped-call]
    resampler.SetDefaultPixelValue(0)  # type: ignore[no-untyped-call]
    resampler.SetTransform(transform)  # type: ignore[no-untyped-call]
    warped = resampler.Execute(moving_mask)  # type: ignore[no-untyped-call]

    return _sitk_image_to_tensor(warped) > 0


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
    metric_device = torch.device(config.device)
    cpu_device = torch.device("cpu")
    if metric_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Structure metrics requested CUDA, but torch.cuda.is_available() is False")

    # Structure label maps use the same geometry as preprocessed CTs, but their
    # label-preserving transform is not saved by prepare_test_data. Build it
    # once here and reuse it for every planning/reference structure file.
    structure_transform = get_structure_label_transform(config)

    for patient_id, gen_paths in tqdm(generated.items(), desc="Calculating generated vs real structure metrics"):
        all_ref_paths = originals.get(patient_id, [])
        ref_paths = [path for path in all_ref_paths if not _is_planning_ct_path(path)]
        planning_paths = [path for path in all_ref_paths if _is_planning_ct_path(path)]

        if len(ref_paths) == 0 or len(planning_paths) == 0:
            logger.warning("Skipping structure metrics for %s: missing planning or non-planning CT", patient_id)
            continue

        planning_ct_path = planning_paths[0]
        planning_structure_path = _structure_path_for_ct(planning_ct_path, config)

        if not planning_structure_path.exists():
            logger.warning("Skipping structure metrics for %s: missing %s", patient_id, planning_structure_path)
            continue

        # Registration and mask warping go through SimpleITK, so keep CT
        # volumes on CPU and avoid caching all patient CTs in CUDA memory.
        planning_ct = _load_hu_tensor(planning_ct_path, config=config, device=cpu_device)
        planning_label_map = load_structure_label_map(planning_structure_path, structure_transform)
        planning_masks = {label: label_to_binary_mask(planning_label_map, label) for label in config.structure_labels}

        ref_label_maps: dict[Path, torch.Tensor] = {}
        for ref_path in ref_paths:
            ref_structure_path = _structure_path_for_ct(ref_path, config)
            if not ref_structure_path.exists():
                logger.warning("Skipping missing reference structure for %s: %s", patient_id, ref_structure_path)
                continue

            ref_label_maps[ref_path] = load_structure_label_map(ref_structure_path, structure_transform)

        if len(ref_label_maps) == 0:
            continue

        for gen_path in gen_paths:
            gen_ct = _load_hu_tensor(gen_path, config=config, device=cpu_device)

            if gen_ct.shape != planning_ct.shape:
                logger.warning(
                    "Skipping structure registration for %s: %s %s vs planning %s",
                    patient_id,
                    gen_path.name,
                    tuple(gen_ct.shape),
                    tuple(planning_ct.shape),
                )
                del gen_ct
                continue

            with torch.inference_mode():
                logger.info("Generating structure DVF for %s: planning -> %s", patient_id, gen_path.name)
                transform = register_planning_ct_to_generated_ct(
                    planning_ct=planning_ct,
                    generated_ct=gen_ct,
                    config=config,
                )

                for label in config.structure_labels:
                    # Warp one planning label at a time. Only the current
                    # predicted/reference masks are moved to metric_device, so
                    # CUDA sees small boolean masks rather than full CT caches.
                    pred_mask = warp_planning_mask_to_generated_ct(
                        planning_mask=planning_masks[label],
                        generated_ct=gen_ct,
                        transform=transform,
                        config=config,
                    ).to(metric_device)

                    for ref_path, ref_label_map in ref_label_maps.items():
                        ref_structure_path = _structure_path_for_ct(ref_path, config)
                        ref_mask = label_to_binary_mask(ref_label_map, label).to(metric_device)
                        if pred_mask.shape != ref_mask.shape:
                            logger.warning(
                                "Skipping structure shape mismatch for %s, label %s: %s vs %s",
                                patient_id,
                                label,
                                tuple(pred_mask.shape),
                                tuple(ref_mask.shape),
                            )
                            del ref_mask
                            continue

                        hd, hd95_value = hausdorff_and_hd95(pred_mask, ref_mask, spacing=config.spacing)
                        rows.append(
                            {
                                "patient_id": patient_id,
                                "comparison_type": "generated_structure_vs_real_structure",
                                "generated_path": str(gen_path),
                                "reference_path": str(ref_path),
                                "planning_structure_path": str(planning_structure_path),
                                "reference_structure_path": str(ref_structure_path),
                                "structure_label": label,
                                "dice": dice_coefficient(pred_mask, ref_mask),
                                "hd": hd,
                                "hd95": hd95_value,
                            }
                        )

                        del ref_mask

                    del pred_mask

            del gen_ct, transform
            if metric_device.type == "cuda":
                torch.cuda.empty_cache()

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
