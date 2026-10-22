import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
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

from src.pipeline.config import MaisiTestingConfig
from src.pipeline.helpers.helpers import _extract_patient_id, _load_hu_tensor, _normalised_image_key
from src.pipeline.helpers.helpers_metrics import _as_metric_tensor, _resolve_metric_device

logger = logging.getLogger(__name__)

WarpedStructureMaskCache = dict[Path, dict[int, torch.Tensor]]

DemonsRegistrationAlgorithm = (
    sitk.DemonsRegistrationFilter
    | sitk.DiffeomorphicDemonsRegistrationFilter
    | sitk.FastSymmetricForcesDemonsRegistrationFilter
)


@dataclass
class _StructureRegistrationResult:
    """
    Store the CPU registration work product for one generated CT.

    A result contains either a generated CT plus the transform needed for
    structure warping, or a warning explaining why that generated CT should be
    skipped. It is used by the background registration queue so metric
    calculation can consume completed CPU work in generated-path order.

    Attributes
    ----------
    gen_path : Path
        Path to the generated CT tensor associated with this result.

    gen_ct : torch.Tensor | None
        Generated CT tensor loaded on CPU, or ``None`` when registration was
        skipped before a valid transform could be produced.

    transform : Any | None
        SimpleITK transform mapping generated/fixed-grid points into
        planning/moving CT space, or ``None`` when registration was skipped.

    info_message : str | None, optional
        Informational log message to emit after the result is consumed.

    warning : str | None, optional
        Warning message to emit when the generated CT cannot be evaluated.

    masks : dict[int, torch.Tensor] | None, optional
        Complete cached masks. A cache hit needs neither a CT nor a transform.
    """

    gen_path: Path
    gen_ct: torch.Tensor | None
    transform: Any | None
    info_message: str | None = None
    warning: str | None = None
    masks: dict[int, torch.Tensor] | None = None


@dataclass
class _PlanningRegistrationData:
    """
    Store reusable planning images for one patient's registrations and mask warps.

    Registration reads the planning CT and its pyramid without modifying them.
    The mask consumer initializes the label image once, allowing mask warping
    to overlap with the next background registration for the same patient.

    Attributes
    ----------
    image : sitk.Image
        Full-resolution planning CT with pipeline geometry and float32 pixels.

    levels : list[sitk.Image]
        Smoothed planning CT pyramid in configured shrink-factor order,
        excluding the full-resolution image.

    labels : sitk.Image | None, optional
        Full-resolution planning structure label image with signed 16-bit
        pixels, or ``None`` until the first successful registration needs masks.
    """

    image: sitk.Image
    levels: list[sitk.Image]
    labels: sitk.Image | None = None


def _prepare_planning_registration(planning_ct: torch.Tensor, config: MaisiTestingConfig) -> _PlanningRegistrationData:
    """
    Prepare the planning CT and its registration pyramid once per patient.

    Convert the CPU planning tensor to a SimpleITK image and construct each
    moving-image pyramid level with the configured smoothing and shrink factor.
    The result is reused across generated CTs with the same configuration.
    Structure labels are initialized separately when mask warping needs them.

    Parameters
    ----------
    planning_ct : torch.Tensor
        Planning CT in HU-like units with shape ``[H, W, D]``.

    config : MaisiTestingConfig
        Pipeline configuration containing voxel spacing, pyramid shrink factors,
        and per-level smoothing sigmas.

    Returns
    -------
    planning_registration : _PlanningRegistrationData
        Planning image and pyramid ready for reuse, with no label image loaded.
    """
    moving = _tensor_to_sitk_image(planning_ct, config, sitk.sitkFloat32)
    levels = [
        _smooth_and_resample(moving, factor, sigma)
        for factor, sigma in zip(
            config.structure_registration_shrink_factors,
            config.structure_registration_smoothing_sigmas,
            strict=True,
        )
    ]
    return _PlanningRegistrationData(moving, levels)


def _get_structure_label_transform(config: MaisiTestingConfig) -> Compose:
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


def _load_structure_label_map(path: Path, transform: Compose) -> torch.Tensor:
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
        ``_get_structure_label_transform``.

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


def _label_to_binary_mask(label_map: torch.Tensor, label: int) -> torch.Tensor:
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


def _as_binary_mask_pair(
    pred: torch.Tensor | np.ndarray,
    ref: torch.Tensor | np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert two mask-like inputs to validated boolean 3D tensors.

    This helper centralizes mask validation for DSC and Hausdorff metrics. Any
    non-zero value is treated as foreground, which supports binary masks saved
    as integer, float, or boolean arrays.

    Parameters
    ----------
    pred : torch.Tensor | np.ndarray
        Predicted, generated, or warped mask-like 3D array.

    ref : torch.Tensor | np.ndarray
        Reference mask-like 3D array.

    Returns
    -------
    pred_t : torch.Tensor
        Boolean tensor for the predicted mask.

    ref_t : torch.Tensor
        Boolean tensor for the reference mask.

    Raises
    ------
    ValueError
        If either mask has zero elements.
        If either mask is not 3-dimensional.
        If the masks have different shapes.
    """

    device = _resolve_metric_device(pred, ref)
    pred_t = _as_metric_tensor(pred, device=device) > 0
    ref_t = _as_metric_tensor(ref, device=device) > 0

    if pred_t.numel() == 0 or ref_t.numel() == 0:
        raise ValueError("Masks cannot have zero elements")

    if pred_t.ndim != 3 or ref_t.ndim != 3:
        raise ValueError(f"Masks must be 3D. Got {tuple(pred_t.shape)} and {tuple(ref_t.shape)}")

    if pred_t.shape != ref_t.shape:
        raise ValueError(f"Masks must have the same shape. Got {tuple(pred_t.shape)} and {tuple(ref_t.shape)}")

    return pred_t, ref_t


def _surface_voxels(mask: torch.Tensor) -> torch.Tensor:
    """
    Identify foreground surface voxels in a 3D binary mask.

    A foreground voxel is considered part of the surface when its 3x3x3
    neighborhood is not completely filled with foreground. Padding makes
    foreground voxels on the image boundary count as surface voxels.

    Parameters
    ----------
    mask : torch.Tensor
        Boolean 3D foreground mask.

    Returns
    -------
    surface : torch.Tensor
        Boolean 3D mask containing only foreground surface voxels.
    """

    volume = mask.float().unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32, device=mask.device)
    neighbor_count = F.conv3d(volume, kernel, padding=1).squeeze(0).squeeze(0)

    return mask & (neighbor_count < 27.0)


def _nearest_distances(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    batch_size: int = 512,
) -> torch.Tensor:
    """
    Calculate nearest-neighbor distances from source points to target points.

    Distances are computed in batches to avoid materializing one very large
    pairwise distance matrix for large structure surfaces.

    Parameters
    ----------
    source_points : torch.Tensor
        Tensor of source coordinates with shape ``[N, 3]``.

    target_points : torch.Tensor
        Tensor of target coordinates with shape ``[M, 3]``.

    batch_size : int
        Number of source points processed per distance batch.

    Returns
    -------
    distances : torch.Tensor
        One nearest-target distance for each source point, shape ``[N]``.
    """

    chunks = []
    for start in range(0, source_points.shape[0], batch_size):
        chunk = source_points[start : start + batch_size]
        chunks.append(torch.cdist(chunk, target_points).min(dim=1).values)

    return torch.cat(chunks)


def _structure_path_for_ct(ct_path: Path, config: MaisiTestingConfig) -> Path:
    """
    Resolve the original structure label-map path for a processed CT tensor.

    Processed CT tensors are saved as ``.pt`` files, while structure label maps
    remain in the original ``STRUCTURES`` tree as ``.nii.gz`` files. This helper
    maps the processed tensor filename back to the matching original structure
    filename using the patient ID and fraction name.

    Parameters
    ----------
    ct_path : Path
        Processed CT tensor path, for example
        ``Patient_01_fraction_2_.pt``.

    config : MaisiTestingConfig
        Pipeline configuration containing ``structures_root``.

    Returns
    -------
    structure_path : Path
        Expected path to the corresponding structure label-map NIfTI file.
    """

    patient_id = _extract_patient_id(ct_path)
    structure_name = ct_path.name.replace(".pt", ".nii.gz")
    return config.structures_root / patient_id / structure_name


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

    # Hausdorff distance is defined on mask boundaries
    pred_surface = _surface_voxels(pred_t)
    ref_surface = _surface_voxels(ref_t)

    # Surface coordinates are much smaller than dense foreground coordinates
    pred_points = pred_surface.nonzero().float()
    ref_points = ref_surface.nonzero().float()
    spacing_t = torch.as_tensor(spacing, dtype=torch.float32, device=pred_points.device)
    pred_points = pred_points * spacing_t
    ref_points = ref_points * spacing_t

    # Use both directions so HD is symmetric
    distances = torch.cat(
        [
            _nearest_distances(pred_points, ref_points),
            _nearest_distances(ref_points, pred_points),
        ]
    )

    return distances


def _warped_structure_mask_cache_path(config: MaisiTestingConfig, generated_ct_path: Path) -> Path:
    """
    Resolve the persisted warped-mask cache path for one generated CT.

    Parameters
    ----------
    config : MaisiTestingConfig
        Pipeline configuration containing `warped_structure_cache_dir`.

    generated_ct_path : Path
        Generated CT tensor path whose generated-space masks are cached.

    Returns
    -------
    path : Path
        Cache file path for warped masks keyed by structure label.
    """

    patient_id = _extract_patient_id(generated_ct_path)
    return config.warped_structure_cache_dir / patient_id / f"{_normalised_image_key(generated_ct_path)}.pt"


def _warped_structure_cache_metadata(
    config: MaisiTestingConfig, generated_ct_path: Path, planning_ct_path: Path
) -> dict[str, Any]:
    """
    Identify source files and settings for persisted warped structure masks.

    Use resolved paths, file sizes, and modification times to identify the
    generated CT, planning CT, and planning structures without loading their
    contents. Include geometry, intensity conversion, labels, and registration
    settings so incompatible persisted masks can be rejected.

    Parameters
    ----------
    config : MaisiTestingConfig
        Pipeline configuration containing structure paths, image geometry,
        intensity limits, requested labels, and registration settings.

    generated_ct_path : Path
        Generated CT tensor file whose anatomy defines the warped-mask grid.

    planning_ct_path : Path
        Planning CT tensor file used to locate the source structure label map.

    Returns
    -------
    metadata : dict[str, Any]
        Cache format version, source-file identities, and settings required to
        validate persisted masks.

    Raises
    ------
    OSError
        If a source file cannot be accessed to read its filesystem metadata.
    """
    sources = [generated_ct_path, planning_ct_path, _structure_path_for_ct(planning_ct_path, config)]
    return {
        "version": 1,
        "sources": [(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) for path in sources],
        "spacing": tuple(config.spacing),
        "target_image_size": tuple(config.target_image_size),
        "hu_range": (config.data_min, config.data_max),
        "labels": tuple(config.structure_labels),
        "iterations": config.structure_registration_iterations,
        "sigma": config.structure_registration_sigma,
        "shrink_factors": tuple(config.structure_registration_shrink_factors),
        "smoothing_sigmas": tuple(config.structure_registration_smoothing_sigmas),
    }


def _complete_warped_structure_masks(
    masks: dict[int, torch.Tensor] | None, labels: list[int], shape: tuple[int, ...]
) -> bool:
    """
    Check that a run-local mask cache contains every requested output mask.

    Require tensor values with the expected output shape before bypassing
    registration. These dictionaries belong to one evaluation run with fixed
    inputs and settings; persisted entries additionally require source and
    configuration metadata validation.

    Parameters
    ----------
    masks : dict[int, torch.Tensor] | None
        Cached masks keyed by structure label, or ``None`` for a cache miss.

    labels : list[int]
        Structure labels that must all be present in the cache.

    shape : tuple[int, ...]
        Expected spatial shape of every requested mask.

    Returns
    -------
    complete : bool
        Whether the cache contains a tensor with the expected shape for every
        requested label.
    """
    return masks is not None and all(
        isinstance(masks.get(label), torch.Tensor) and tuple(masks[label].shape) == shape for label in labels
    )


def _save_warped_structure_masks(
    config: MaisiTestingConfig,
    generated_ct_path: Path,
    masks: dict[int, torch.Tensor],
    planning_ct_path: Path,
) -> None:
    """
    Persist generated-space warped structure masks for reuse by dose metrics.

    Store CPU boolean masks together with source-file and configuration
    metadata. Later evaluations can validate the cache before reusing it.

    Parameters
    ----------
    config : MaisiTestingConfig
        Pipeline configuration containing `warped_structure_cache_dir`.

    generated_ct_path : Path
        Generated CT tensor path used as the cache key.

    masks : dict[int, torch.Tensor]
        CPU or GPU boolean masks keyed by structure label.

    planning_ct_path : Path
        Planning CT source used to validate the cache's input files.

    Returns
    -------
    None
        Masks and validation metadata are written to the configured cache file.
    """

    cache_path = _warped_structure_mask_cache_path(config, generated_ct_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": _warped_structure_cache_metadata(config, generated_ct_path, planning_ct_path),
            "masks": {label: mask.detach().cpu().bool() for label, mask in masks.items()},
        },
        cache_path,
    )


def _load_warped_structure_masks(
    config: MaisiTestingConfig,
    generated_ct_path: Path,
    planning_ct_path: Path,
) -> dict[int, torch.Tensor] | None:
    """
    Load persisted generated-space warped structure masks when available.

    Validate source-file identities, configuration, requested labels, and mask
    shapes before returning CPU boolean tensors. Missing, unreadable, legacy,
    incomplete, or incompatible cache files are treated as cache misses.

    Parameters
    ----------
    config : MaisiTestingConfig
        Pipeline configuration containing `warped_structure_cache_dir`.

    generated_ct_path : Path
        Generated CT tensor path used as the cache key.

    planning_ct_path : Path
        Planning CT source whose image and structure file must match the cache.

    Returns
    -------
    masks : dict[int, torch.Tensor] | None
        CPU boolean masks keyed by structure label, or `None` when no complete
        cache exists for the configured structure labels.
    """

    cache_path = _warped_structure_mask_cache_path(config, generated_ct_path)
    if not cache_path.exists():
        return None

    try:
        loaded = torch.load(cache_path, weights_only=True, map_location="cpu")
        metadata = _warped_structure_cache_metadata(config, generated_ct_path, planning_ct_path)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError):
        logger.warning("Ignoring unreadable warped-structure cache: %s", cache_path)
        return None
    if not isinstance(loaded, dict) or loaded.get("metadata") != metadata:
        logger.warning("Ignoring invalid warped-structure cache: %s", cache_path)
        return None

    loaded = loaded.get("masks")
    if not isinstance(loaded, dict):
        return None

    masks: dict[int, torch.Tensor] = {}
    for label in config.structure_labels:
        value = loaded.get(label)
        if value is None:
            value = loaded.get(str(label))

        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(config.target_image_size):
            logger.warning("Ignoring incomplete warped-structure cache: %s", cache_path)
            return None

        masks[label] = value.detach().cpu().bool()

    return masks


def _tensor_to_sitk_image(tensor: torch.Tensor, config: MaisiTestingConfig, pixel_id: int) -> sitk.Image:
    """
    Convert a pipeline tensor to a SimpleITK image.

    Pipeline tensors use shape ``[H, W, D]``. Transposing with ``(2, 1, 0)``
    produces an array ordered as ``[D, W, H]``. SimpleITK interprets array
    axes as z/y/x, so image x/y/z correspond to tensor H/W/D. Spacing is
    assigned unchanged and must describe voxel spacing along those axes.
    Assigning spacing does not resample voxel data.

    Parameters
    ----------
    tensor : torch.Tensor
        Pipeline image or mask tensor with shape ``[H, W, D]``.

    config : MaisiTestingConfig
        Pipeline configuration containing voxel spacing.

    pixel_id : int
        Supported SimpleITK pixel type: ``sitk.sitkFloat32`` for CT images,
        ``sitk.sitkUInt8`` for binary masks, or ``sitk.sitkInt16`` for label maps.

    Returns
    -------
    image : sitk.Image
        SimpleITK image with spacing and identity physical metadata assigned.
    """

    arr = tensor.detach().cpu().numpy()
    arr = np.transpose(arr, (2, 1, 0))
    dtype = {sitk.sitkFloat32: np.float32, sitk.sitkUInt8: np.uint8, sitk.sitkInt16: np.int16}[pixel_id]
    image = sitk.GetImageFromArray(arr.astype(dtype, copy=False))
    image.SetSpacing(config.spacing)  # type: ignore[no-untyped-call]
    image.SetOrigin((0.0, 0.0, 0.0))  # type: ignore[no-untyped-call]
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))  # type: ignore[no-untyped-call]
    return image


def _sitk_image_to_tensor(image: sitk.Image) -> torch.Tensor:
    """
    Convert a SimpleITK image to a pipeline tensor with shape ``[H, W, D]``.

    SimpleITK array extraction returns axes in z/y/x order. Transposing with
    ``(2, 1, 0)`` produces x/y/z order, reversing the axis conversion in
    ``_tensor_to_sitk_image`` and restoring the original pipeline tensor axes.

    Parameters
    ----------
    image : sitk.Image
        SimpleITK image.

    Returns
    -------
    tensor : torch.Tensor
        Tensor with axes in image x/y/z order, or shape ``[H, W, D]`` for an
        image created by ``_tensor_to_sitk_image``.
    """

    arr = sitk.GetArrayFromImage(image)
    arr = np.transpose(arr, (2, 1, 0))
    return torch.as_tensor(arr)


def _smooth_and_resample(image: sitk.Image, shrink_factor: float, smoothing_sigma: float) -> sitk.Image:
    """
    Smooth and downsample an image for multiscale demons registration.

    The image is smoothed in physical units before downsampling, matching the
    multiscale registration strategy used by the DDF workflow. The output
    preserves the original origin and direction while adjusting spacing for the
    lower-resolution grid.

    Parameters
    ----------
    image : sitk.Image
        SimpleITK image to smooth and resample.

    shrink_factor : float
        Factor by which each image dimension is reduced.

    smoothing_sigma : float
        Gaussian smoothing sigma in physical units.

    Returns
    -------
    image : sitk.Image
        Smoothed and resampled SimpleITK image.
    """

    smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_sigma)  # type: ignore[no-untyped-call]

    original_spacing = image.GetSpacing()  # type: ignore[no-untyped-call]
    original_size = image.GetSize()  # type: ignore[no-untyped-call]
    new_size = [max(2, int(sz / shrink_factor + 0.5)) for sz in original_size]
    new_spacing = [
        ((original_sz - 1) * original_spc) / (new_sz - 1)
        for original_sz, original_spc, new_sz in zip(original_size, original_spacing, new_size, strict=True)
    ]

    return sitk.Resample(
        smoothed_image,
        new_size,
        sitk.Transform(),  # type: ignore[no-untyped-call]
        sitk.sitkLinear,
        image.GetOrigin(),  # type: ignore[no-untyped-call]
        new_spacing,
        image.GetDirection(),  # type: ignore[no-untyped-call]
        0.0,
        image.GetPixelID(),  # type: ignore[no-untyped-call]
    )


def _multiscale_demons(
    registration_algorithm: DemonsRegistrationAlgorithm,
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    initial_transform: sitk.Transform,
    shrink_factors: list[float],
    smoothing_sigmas: list[float],
    moving_levels: list[sitk.Image] | None = None,
) -> sitk.DisplacementFieldTransform:
    """
    Run demons registration from coarse to full resolution.

    This helper initializes a displacement field on the coarsest requested
    level, then refines it toward the original image grid one level at a time.
    Fixed pyramid levels are created on demand. Moving levels can be supplied
    from a patient-scoped cache to avoid rebuilding them for every generated CT.

    Parameters
    ----------
    registration_algorithm : DemonsRegistrationAlgorithm
        SimpleITK demons registration filter with an ``Execute`` method.

    fixed_image : sitk.Image
        SimpleITK fixed image defining the output spatial domain.

    moving_image : sitk.Image
        SimpleITK moving image that is registered into fixed-image space.

    initial_transform : sitk.Transform
        SimpleITK transform used to initialize the displacement field.

    shrink_factors : list[float]
        Pyramid shrink factors, ordered from coarse to fine in configuration
        style. The original resolution is handled implicitly.

    smoothing_sigmas : list[float]
        Gaussian smoothing sigmas for each shrink factor.

    moving_levels : list[sitk.Image] | None, optional
        Precomputed moving images in shrink-factor order, excluding full resolution.

    Returns
    -------
    transform : sitk.DisplacementFieldTransform
        SimpleITK displacement-field transform mapping fixed-image points to
        moving-image points for resampling into fixed-image space.
    """

    pyramid_levels = list(zip(shrink_factors, smoothing_sigmas, strict=True))
    if len(pyramid_levels) > 0:
        first_shrink_factor, first_smoothing_sigma = pyramid_levels[0]
        fixed_level = _smooth_and_resample(fixed_image, first_shrink_factor, first_smoothing_sigma)
        moving_level = (
            moving_levels[0]
            if moving_levels is not None
            else _smooth_and_resample(moving_image, first_shrink_factor, first_smoothing_sigma)
        )
        remaining_levels = pyramid_levels[1:]
    else:
        fixed_level = fixed_image
        moving_level = moving_image
        remaining_levels = []

    # Demons filters require displacement fields with vector float64 pixels
    displacement_field = sitk.TransformToDisplacementField(  # type: ignore[no-untyped-call]
        initial_transform,
        sitk.sitkVectorFloat64,
        fixed_level.GetSize(),  # type: ignore[no-untyped-call]
        fixed_level.GetOrigin(),  # type: ignore[no-untyped-call]
        fixed_level.GetSpacing(),  # type: ignore[no-untyped-call]
        fixed_level.GetDirection(),  # type: ignore[no-untyped-call]
    )

    # Start registration on the coarsest grid so large deformations are found
    # before the full-resolution details
    displacement_field = registration_algorithm.Execute(  # type: ignore[no-untyped-call]
        fixed_level,
        moving_level,
        displacement_field,
    )

    for level_index, (shrink_factor, smoothing_sigma) in enumerate(remaining_levels, start=1):
        fixed_level = _smooth_and_resample(fixed_image, shrink_factor, smoothing_sigma)
        moving_level = (
            moving_levels[level_index]
            if moving_levels is not None
            else _smooth_and_resample(moving_image, shrink_factor, smoothing_sigma)
        )

        # Carry the current DVF estimate onto the next finer fixed-image grid
        displacement_field = sitk.Resample(displacement_field, fixed_level)
        displacement_field = registration_algorithm.Execute(  # type: ignore[no-untyped-call]
            fixed_level,
            moving_level,
            displacement_field,
        )

    # Finish with one refinement on the original fixed-image grid
    displacement_field = sitk.Resample(displacement_field, fixed_image)
    displacement_field = registration_algorithm.Execute(  # type: ignore[no-untyped-call]
        fixed_image,
        moving_image,
        displacement_field,
    )

    return sitk.DisplacementFieldTransform(displacement_field)  # type: ignore[no-untyped-call]


def _register_planning_ct_to_generated_ct(
    planning_ct: torch.Tensor,
    generated_ct: torch.Tensor,
    config: MaisiTestingConfig,
    planning_registration: _PlanningRegistrationData | None = None,
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

    planning_registration : _PlanningRegistrationData | None, optional
        Patient-scoped planning image and pyramid for the same configuration.

    Returns
    -------
    transform : Any
        SimpleITK displacement-field transform mapping generated/fixed-grid
        points into planning/moving CT space for resampling.
    """

    # SimpleITK registration runs on CPU.
    # These conversions detach tensors from any accelerator residency.
    fixed = _tensor_to_sitk_image(generated_ct, config, sitk.sitkFloat32)
    moving = (
        planning_registration.image
        if planning_registration is not None
        else _tensor_to_sitk_image(planning_ct, config, sitk.sitkFloat32)
    )

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
    # then refines that field from coarse to full resolution
    return _multiscale_demons(
        registration_algorithm=demons_filter,
        fixed_image=fixed,
        moving_image=moving,
        initial_transform=initial_transform,
        shrink_factors=config.structure_registration_shrink_factors,
        smoothing_sigmas=config.structure_registration_smoothing_sigmas,
        moving_levels=planning_registration.levels if planning_registration is not None else None,
    )


def _warp_planning_mask_to_generated_ct(
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
        SimpleITK transform mapping generated/fixed-grid points into
        planning/moving mask space.

    config : MaisiTestingConfig
        Pipeline configuration containing spacing and image-grid settings.

    Returns
    -------
    warped_mask : torch.Tensor
        Boolean 3D mask in generated CT space.
    """

    # The generated CT is used only as the reference grid: size, spacing,
    # origin, and direction define where the warped planning mask is sampled.
    # Pipeline images have zero origin and identity direction, matching the
    # resampler defaults. No generated CT pixels need to be copied.
    moving_mask = _tensor_to_sitk_image(planning_mask.to(torch.uint8), config, sitk.sitkUInt8)

    resampler = sitk.ResampleImageFilter()  # type: ignore[no-untyped-call]
    resampler.SetSize(tuple(generated_ct.shape))  # type: ignore[no-untyped-call]
    resampler.SetOutputSpacing(config.spacing)  # type: ignore[no-untyped-call]
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # type: ignore[no-untyped-call]
    resampler.SetDefaultPixelValue(0)  # type: ignore[no-untyped-call]
    resampler.SetTransform(transform)  # type: ignore[no-untyped-call]
    warped = resampler.Execute(moving_mask)  # type: ignore[no-untyped-call]

    return _sitk_image_to_tensor(warped) > 0


def _warp_planning_labels_to_generated_ct(
    planning_label_map: torch.Tensor | sitk.Image,
    generated_ct: torch.Tensor,
    transform: Any,
    config: MaisiTestingConfig,
) -> dict[int, torch.Tensor]:
    """
    Warp planning labels into generated CT space and extract binary masks.

    Nearest-neighbor sampling commutes with label extraction, including the
    aggregate foreground label (<= 0). Signed 16-bit pixels preserve the
    preprocessed label map's values without the UInt8 truncation of labels.
    A preconverted SimpleITK image can be reused across generated CTs.

    Parameters
    ----------
    planning_label_map : torch.Tensor | sitk.Image
        Planning structure labels as an integer tensor with shape
        ``[H, W, D]`` or a preconverted SimpleITK image with pipeline geometry.

    generated_ct : torch.Tensor
        Generated CT tensor whose spatial shape defines the output grid.
        Its voxel intensities are not read during mask resampling.

    transform : Any
        SimpleITK transform mapping generated-grid physical points back into
        planning label-map space.

    config : MaisiTestingConfig
        Pipeline configuration containing voxel spacing and requested labels.
        Output origin is zero and direction is identity, as in CT conversion.

    Returns
    -------
    masks : dict[int, torch.Tensor]
        CPU boolean masks on the generated CT grid, keyed by requested label.
        Labels less than or equal to zero select all non-background structures.
    """
    moving = (
        planning_label_map
        if isinstance(planning_label_map, sitk.Image)
        else _tensor_to_sitk_image(planning_label_map, config, sitk.sitkInt16)
    )
    resampler = sitk.ResampleImageFilter()  # type: ignore[no-untyped-call]
    resampler.SetSize(tuple(generated_ct.shape))  # type: ignore[no-untyped-call]
    resampler.SetOutputSpacing(config.spacing)  # type: ignore[no-untyped-call]
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # type: ignore[no-untyped-call]
    resampler.SetDefaultPixelValue(0)  # type: ignore[no-untyped-call]
    resampler.SetTransform(transform)  # type: ignore[no-untyped-call]
    warped_labels = _sitk_image_to_tensor(resampler.Execute(moving))  # type: ignore[no-untyped-call]
    return {label: _label_to_binary_mask(warped_labels, label) for label in config.structure_labels}


def _calculate_structure_registration_result(
    gen_path: Path,
    patient_id: str,
    planning_ct: torch.Tensor,
    config: MaisiTestingConfig,
    planning_registration: _PlanningRegistrationData | None = None,
) -> _StructureRegistrationResult:
    """
    Load one generated CT and calculate its planning-to-generated transform.

    This helper is intentionally CPU-only so it can run in a background thread
    while CUDA evaluates metrics for the previous generated CT. Shape
    mismatches are returned as warnings instead of being raised, which lets the
    caller preserve the existing skip-and-continue behavior.

    Parameters
    ----------
    gen_path : Path
        Path to the generated CT tensor that should be registered.

    patient_id : str
        Patient identifier used only for diagnostic messages.

    planning_ct : torch.Tensor
        Planning CT tensor in HU-like units on CPU.

    config : MaisiTestingConfig
        Pipeline configuration containing CT intensity range, spacing, and
        demons-registration settings.

    planning_registration : _PlanningRegistrationData | None, optional
        Reusable planning image and pyramid built with the same configuration.

    Returns
    -------
    result : _StructureRegistrationResult
        Registration result containing the generated CT and transform when
        registration succeeds, or a warning when the generated CT is skipped.
    """

    cpu_device = torch.device("cpu")
    gen_ct = _load_hu_tensor(gen_path, config=config, device=cpu_device)

    if gen_ct.shape != planning_ct.shape:
        warning = (
            "Skipping structure registration for "
            f"{patient_id}: {gen_path.name} {tuple(gen_ct.shape)} vs planning {tuple(planning_ct.shape)}"
        )
        return _StructureRegistrationResult(
            gen_path=gen_path,
            gen_ct=None,
            transform=None,
            warning=warning,
        )

    transform = _register_planning_ct_to_generated_ct(
        planning_ct=planning_ct,
        generated_ct=gen_ct,
        config=config,
        planning_registration=planning_registration,
    )

    return _StructureRegistrationResult(
        gen_path=gen_path,
        gen_ct=gen_ct,
        transform=transform,
        info_message=f"Generated structure DVF for {patient_id}: planning -> {gen_path.name}",
    )
