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
    _cache_patient_cts,
    _is_planning_ct_path,
    _multiscale_demons,
    _nearest_distances,
    _sitk_image_to_tensor,
    _structure_path_for_ct,
    _surface_voxels,
    _tensor_to_sitk_image,
)


def dice_coefficient(pred: torch.Tensor | np.ndarray, ref: torch.Tensor | np.ndarray) -> float:
    """
    Calculate Dice Similarity Coefficient (DSC) for two 3D masks.

    DSC measures spatial overlap between two binary structure masks. Non-zero
    values are treated as foreground. If both masks are empty, the score is
    defined as 1.0 because the masks agree perfectly on foreground absence.

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
        If either mask is empty.
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
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    percentile: float | None = None,
) -> float:
    """
    Calculate symmetric Hausdorff distance for two 3D masks.

    The distance is computed between foreground surface voxels in physical
    units using the provided voxel spacing. When ``percentile`` is provided,
    this function returns the corresponding percentile Hausdorff distance, such
    as HD95 for ``percentile=95``.

    Empty-mask convention:
    - Both masks empty -> 0.0.
    - Only one mask empty -> positive infinity.

    Parameters
    ----------
    pred : torch.Tensor | np.ndarray
        Predicted, generated, or warped 3D structure mask.

    ref : torch.Tensor | np.ndarray
        Reference 3D structure mask.

    spacing : tuple[float, float, float]
        Physical voxel spacing for the mask axes.

    percentile : float | None
        Optional percentile in the interval (0, 100]. If None, the maximum
        surface distance is returned.

    Returns
    -------
    hd : float
        Symmetric Hausdorff distance in physical units.

    Raises
    ------
    ValueError
        If either mask is empty.
        If either mask is not 3-dimensional.
        If the masks have different shapes.
        If ``percentile`` is outside the interval (0, 100].
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

    pred_points = pred_surface.nonzero().float()
    ref_points = ref_surface.nonzero().float()
    spacing_t = torch.as_tensor(spacing, dtype=torch.float32, device=pred_points.device)
    pred_points = pred_points * spacing_t
    ref_points = ref_points * spacing_t

    distances = torch.cat(
        [
            _nearest_distances(pred_points, ref_points),
            _nearest_distances(ref_points, pred_points),
        ]
    )

    if percentile is None:
        return float(distances.max().item())

    if percentile <= 0.0 or percentile > 100.0:
        raise ValueError(f"`percentile` must be in (0, 100]. Got {percentile}")

    return float(torch.quantile(distances, percentile / 100.0).item())


def hd95(pred: torch.Tensor | np.ndarray, ref: torch.Tensor | np.ndarray, spacing: tuple[float, float, float]) -> float:
    """
    Calculate the 95th percentile Hausdorff distance (HD95).

    HD95 is less sensitive to isolated outlier voxels than the maximum
    Hausdorff distance while still measuring boundary disagreement.

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
    hd95 : float
        Symmetric 95th percentile Hausdorff distance in physical units.
    """

    return hausdorff_distance(pred, ref, spacing=spacing, percentile=95.0)


def load_structure_label_map(path: Path, config: MaisiTestingConfig) -> torch.Tensor:
    """
    Load and preprocess a structure label map to the pipeline target grid.

    The transform mirrors the CT preprocessing geometry: load image, enforce
    channel-first format, reorient to RAS, pad/crop to the configured target
    image size, and return an integer label tensor.

    Parameters
    ----------
    path : Path
        Path to a structure label-map NIfTI file.

    config : MaisiTestingConfig
        Pipeline configuration containing target image size and transform
        settings.

    Returns
    -------
    label_map : torch.Tensor
        Integer 3D label map on CPU with shape matching
        ``config.target_image_size``.
    """

    transform = Compose(
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
    moving image. The returned SimpleITK transform maps planning CT space into
    generated CT space, so it can be reused to warp every selected planning
    structure label for the same generated CT.

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
        SimpleITK displacement-field transform from planning CT space to
        generated CT space.
    """

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

    Nearest-neighbor interpolation is used to preserve the binary mask labels.

    Parameters
    ----------
    planning_mask : torch.Tensor
        Boolean or binary 3D structure mask in planning CT space.

    generated_ct : torch.Tensor
        Generated CT tensor used as the resampling reference grid.

    transform : Any
        SimpleITK transform returned by
        ``register_planning_ct_to_generated_ct``.

    config : MaisiTestingConfig
        Pipeline configuration containing spacing and image-grid settings.

    Returns
    -------
    warped_mask : torch.Tensor
        Boolean 3D mask in generated CT space.
    """

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
        return

    rows: list[dict[str, Any]] = []
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Structure metrics requested CUDA, but torch.cuda.is_available() is False")

    print(f"Structure metric tensors use {device}; SimpleITK displacement registration and mask resampling run on CPU.")

    for patient_id, gen_paths in tqdm(generated.items(), desc="Calculating generated vs real structure metrics."):
        all_ref_paths = originals.get(patient_id, [])
        ref_paths = [path for path in all_ref_paths if not _is_planning_ct_path(path)]
        planning_paths = [path for path in all_ref_paths if _is_planning_ct_path(path)]

        if len(ref_paths) == 0 or len(planning_paths) == 0:
            print(f"Skipping structure metrics for {patient_id}: missing planning or non-planning CT")
            continue

        planning_ct_path = planning_paths[0]
        planning_structure_path = _structure_path_for_ct(planning_ct_path, config)

        if not planning_structure_path.exists():
            print(f"Skipping structure metrics for {patient_id}: missing {planning_structure_path}")
            continue

        patient_tensors = _cache_patient_cts([*gen_paths, *ref_paths, planning_ct_path], config=config, device=device)
        planning_ct = patient_tensors[planning_ct_path]
        planning_label_map = load_structure_label_map(planning_structure_path, config)

        generated_masks: dict[tuple[Path, int], torch.Tensor] = {}

        for gen_path in gen_paths:
            gen_ct = patient_tensors[gen_path]

            if gen_ct.shape != planning_ct.shape:
                print(
                    f"Skipping structure registration for {patient_id}: "
                    f"{gen_path.name} {tuple(gen_ct.shape)} vs planning {tuple(planning_ct.shape)}"
                )
                continue

            transform = register_planning_ct_to_generated_ct(
                planning_ct=planning_ct,
                generated_ct=gen_ct,
                config=config,
            )

            for label in config.structure_labels:
                planning_mask = label_to_binary_mask(planning_label_map, label)
                generated_masks[(gen_path, label)] = warp_planning_mask_to_generated_ct(
                    planning_mask=planning_mask,
                    generated_ct=gen_ct,
                    transform=transform,
                    config=config,
                ).to(device)

        for gen_path in gen_paths:
            for ref_path in ref_paths:
                ref_structure_path = _structure_path_for_ct(ref_path, config)
                if not ref_structure_path.exists():
                    print(f"Skipping missing reference structure for {patient_id}: {ref_structure_path}")
                    continue

                ref_label_map = load_structure_label_map(ref_structure_path, config)

                for label in config.structure_labels:
                    pred_mask = generated_masks.get((gen_path, label))
                    if pred_mask is None:
                        continue

                    ref_mask = label_to_binary_mask(ref_label_map, label).to(device)
                    if pred_mask.shape != ref_mask.shape:
                        print(
                            f"Skipping structure shape mismatch for {patient_id}, label {label}: "
                            f"{tuple(pred_mask.shape)} vs {tuple(ref_mask.shape)}"
                        )
                        continue

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
                            "hd": hausdorff_distance(pred_mask, ref_mask, spacing=config.spacing),
                            "hd95": hd95(pred_mask, ref_mask, spacing=config.spacing),
                        }
                    )

    if len(rows) == 0:
        print("Skipping structure metrics: no valid structure comparisons were calculated")
        return

    df = pd.DataFrame(rows)
    df.to_csv(config.metrics_dir / "generated_vs_real_structure_metrics.csv", index=False)

    summary = df.groupby(["patient_id", "structure_label"])[["dice", "hd", "hd95"]].agg(
        ["mean", "std", "min", "max", "count"]
    )
    summary.to_csv(config.metrics_dir / "generated_vs_real_structure_summary.csv")
