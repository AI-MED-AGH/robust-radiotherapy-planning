import glob
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
from src.pipeline.evaluation.structure_metrics import (
    WarpedStructureMaskCache,
    load_warped_structure_masks,
    save_warped_structure_masks,
)
from src.pipeline.helpers.helpers import (
    _as_metric_tensor,
    _calculate_structure_registration_result,
    _extract_patient_id,
    _get_structure_label_transform,
    _is_planning_ct_path,
    _label_to_binary_mask,
    _load_hu_tensor,
    _load_structure_label_map,
    _load_tensor,
    _structure_path_for_ct,
    _warp_planning_mask_to_generated_ct,
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


def _normalised_image_key(path: Path) -> str:
    """
    Return a filename key independent of common medical-image extensions.

    Parameters
    ----------
    path : Path
        Dose or structure file path.

    Returns
    -------
    key : str
        Filename without `.nii.gz`, `.nii`, or `.pt` suffix.
    """

    name = path.name
    for suffix in (".nii.gz", ".nii", ".pt"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _structure_path_for_dose(dose_path: Path, config: MaisiTestingConfig) -> Path:
    """
    Resolve the structure label map corresponding to a dose file.

    Parameters
    ----------
    dose_path : Path
        Dose file whose patient and fraction identify the structure label map.

    config : MaisiTestingConfig
        Pipeline configuration containing `structures_root`.

    Returns
    -------
    structure_path : Path
        Expected NIfTI label-map path for the same patient/fraction.
    """

    patient_id = _extract_patient_id(dose_path)
    return config.structures_root / patient_id / f"{_normalised_image_key(dose_path)}.nii.gz"


def _get_dose_transform(config: MaisiTestingConfig) -> Compose:
    """
    Create the geometric preprocessing transform for NIfTI dose maps.

    The transform mirrors the geometric part of CT preprocessing and structure
    preprocessing so dose, CT, and label tensors share the configured target
    grid. Dose values are kept in physical units and are not intensity-scaled.

    Parameters
    ----------
    config : MaisiTestingConfig
        Pipeline configuration containing `target_image_size`.

    Returns
    -------
    transform : Compose
        MONAI dictionary transform that returns a float dose tensor.
    """

    return Compose(
        [
            LoadImaged(keys="dose"),
            EnsureChannelFirstd(keys="dose"),
            Orientationd(keys="dose", axcodes="RAS", labels=None),
            SpatialPadd(
                keys="dose",
                spatial_size=config.target_image_size,
                mode="constant",
                constant_values=0,
            ),
            CenterSpatialCropd(keys="dose", roi_size=config.target_image_size),
            EnsureTyped(keys="dose", dtype=torch.float32),
        ]
    )


def _load_nifti_dose(path: Path, transform: Compose) -> torch.Tensor:
    """
    Load and geometrically align one NIfTI dose map.

    Parameters
    ----------
    path : Path
        NIfTI dose file.

    transform : Compose
        Dose preprocessing transform returned by `_get_dose_transform`.

    Returns
    -------
    dose : torch.Tensor
        CPU float tensor with shape matching the pipeline target grid.
    """

    transformed = transform({"dose": str(path)})
    dose_meta: MetaTensor = transformed["dose"]
    return dose_meta.as_tensor().detach().cpu()[0].float()


def load_dose_distribution(path: Path, transform: Compose) -> torch.Tensor:
    """
    Load a 3D dose distribution from either a NIfTI image or a saved tensor.

    Dose values are assumed to be stored in physical dose units, typically Gy.
    Unlike CT loading, this function does not normalize or convert intensity
    ranges.

    Parameters
    ----------
    path : Path
        Dose file path. Supported suffixes are `.nii`, `.nii.gz`, and `.pt`.

    transform : Compose
        MONAI transform used for NIfTI dose files. Tensor files are loaded
        directly and do not use this transform.

    Returns
    -------
    dose : torch.Tensor
        CPU float 3D dose tensor.

    Raises
    ------
    ValueError
        If the file suffix is unsupported, the tensor is not 3D, the tensor is
        empty, or dose values contain NaN/inf.
    """

    if path.name.endswith(".pt"):
        dose = _load_tensor(path).float()
    elif path.name.endswith((".nii", ".nii.gz")):
        dose = _load_nifti_dose(path, transform)
    else:
        raise ValueError(f"Unsupported dose file extension: {path}")

    if dose.numel() == 0:
        raise ValueError(f"Dose distribution is empty: {path}")

    if dose.ndim != 3:
        raise ValueError(f"Dose distribution must be 3D. Got {tuple(dose.shape)} for {path}")

    if not torch.isfinite(dose).all():
        raise ValueError(f"Dose distribution contains NaN or infinite values: {path}")

    return dose


def collect_dose_distributions(dose_dir: Path) -> dict[str, list[Path]]:
    """
    Collect dose distribution files grouped by patient ID.

    Supported files are `.nii`, `.nii.gz`, and `.pt`, allowing the same
    evaluator to compare raw converted RTDOSE files and model-produced tensors.

    Parameters
    ----------
    dose_dir : Path
        Directory searched recursively for dose files.

    Returns
    -------
    doses : dict[str, list[Path]]
        Dose paths grouped by patient ID.

    Raises
    ------
    FileNotFoundError
        If `dose_dir` does not exist.

    ValueError
        If no supported dose files are found.
    """

    if not dose_dir.exists():
        raise FileNotFoundError(f"Dose directory does not exist: {dose_dir}")

    paths: list[Path] = []
    for pattern in ("*.nii", "*.nii.gz", "*.pt"):
        paths.extend(Path(path) for path in glob.glob(str(dose_dir / "**" / pattern), recursive=True))

    paths = sorted(set(paths))
    if len(paths) == 0:
        raise ValueError(f"No dose files found in: {dose_dir}")

    doses: dict[str, list[Path]] = {}
    for path in paths:
        doses.setdefault(_extract_patient_id(path), []).append(path)

    return doses


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

    dose_t = _as_metric_tensor(dose)
    mask_t = torch.as_tensor(mask, device=dose_t.device).bool()

    if dose_t.shape != mask_t.shape:
        raise ValueError(f"Dose and mask must have the same shape. Got {tuple(dose_t.shape)} and {tuple(mask_t.shape)}")

    structure_dose = dose_t[mask_t]
    if structure_dose.numel() == 0:
        return pd.DataFrame(columns=["dose_gy", "volume_percent"])

    max_dose = float(structure_dose.max().item())
    bins = torch.arange(0.0, max_dose + bin_width, bin_width, dtype=torch.float32, device=dose_t.device)
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

    dose_t = _as_metric_tensor(dose)
    mask_t = torch.as_tensor(mask, device=dose_t.device).bool()
    structure_dose = dose_t[mask_t]

    if structure_dose.numel() == 0:
        return float("nan")

    quantile = torch.as_tensor((100.0 - volume_percent) / 100.0, dtype=structure_dose.dtype, device=dose_t.device)
    return float(torch.quantile(structure_dose, quantile).item())


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

    dose_t = _as_metric_tensor(dose)
    mask_t = torch.as_tensor(mask, device=dose_t.device).bool()
    structure_dose = dose_t[mask_t]

    if structure_dose.numel() == 0:
        return float("nan")

    return float((structure_dose >= dose_threshold).float().mean().item() * 100.0)


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

    dose_t = _as_metric_tensor(dose)
    mask_t = torch.as_tensor(mask, device=dose_t.device).bool()
    structure_dose = dose_t[mask_t]

    if structure_dose.numel() == 0:
        return float("nan")

    return float(structure_dose.mean().item())


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

    dose_t = _as_metric_tensor(dose)
    mask_t = torch.as_tensor(mask, device=dose_t.device).bool()
    structure_dose = dose_t[mask_t]

    if structure_dose.numel() == 0:
        return float("nan")

    return float(structure_dose.max().item())


def _masked_error_metrics(
    pred_dose: torch.Tensor,
    ref_dose: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, float]:
    """
    Calculate masked voxel MAE and RMSE between two dose distributions.

    Parameters
    ----------
    pred_dose : torch.Tensor
        Predicted dose tensor already on the metric device.

    ref_dose : torch.Tensor
        Reference dose tensor already on the metric device.

    mask : torch.Tensor
        Boolean mask already on the metric device.

    Returns
    -------
    metrics : tuple[float, float]
        Pair containing masked voxel MAE and RMSE. Empty masks return
        `(nan, nan)`.
    """

    mask_t = mask.bool()
    pred_values = pred_dose[mask_t]
    ref_values = ref_dose[mask_t]

    if pred_values.numel() == 0:
        return float("nan"), float("nan")

    error = pred_values - ref_values
    mae = float(torch.mean(torch.abs(error)).item())
    rmse = float(torch.sqrt(torch.mean(error.square())).item())
    return mae, rmse


def _metric_row(
    metric_name: str,
    pred_value: float,
    ref_value: float,
    base: dict[str, Any],
) -> dict[str, Any]:
    """
    Build one predicted/reference metric comparison row.

    Parameters
    ----------
    metric_name : str
        Name of the metric represented by the row.

    pred_value : float
        Metric value measured on the predicted dose.

    ref_value : float
        Metric value measured on the reference dose.

    base : dict[str, Any]
        Shared row metadata such as patient ID and structure label.

    Returns
    -------
    row : dict[str, Any]
        CSV-ready metric row containing values and signed/absolute
        differences.
    """

    return {
        **base,
        "metric": metric_name,
        "predicted_value": pred_value,
        "reference_value": ref_value,
        "difference": pred_value - ref_value,
        "absolute_difference": abs(pred_value - ref_value),
    }


def _reference_lookup(paths: list[Path]) -> dict[str, Path]:
    """
    Build a dose lookup keyed by extension-independent filename.

    Parameters
    ----------
    paths : list[Path]
        Reference dose paths for one patient.

    Returns
    -------
    lookup : dict[str, Path]
        Mapping from normalized dose filename to path.
    """

    return {_normalised_image_key(path): path for path in paths}


def _generated_ct_lookup(config: MaisiTestingConfig) -> dict[str, Path]:
    """
    Build a lookup for generated CT tensors keyed like predicted dose files.

    Parameters
    ----------
    config : MaisiTestingConfig
        Pipeline configuration containing `generated_ct_dir`.

    Returns
    -------
    lookup : dict[str, Path]
        Mapping from normalized generated CT filename to path. An empty lookup
        is returned when the generated CT directory is not available.
    """

    if not config.generated_ct_dir.exists():
        return {}

    return {
        _normalised_image_key(Path(path)): Path(path)
        for path in glob.glob(str(config.generated_ct_dir / "**" / "*.pt"), recursive=True)
    }


def _planning_ct_path_for_patient(patient_id: str, config: MaisiTestingConfig) -> Path | None:
    """
    Resolve the processed planning CT tensor for a patient.

    Parameters
    ----------
    patient_id : str
        Patient identifier.

    config : MaisiTestingConfig
        Pipeline configuration containing `processed_ct_dir`.

    Returns
    -------
    path : Path | None
        Planning CT tensor path, or `None` when it cannot be found.
    """

    candidates = sorted(
        Path(path)
        for path in glob.glob(str(config.processed_ct_dir / patient_id / "*.pt"), recursive=True)
        if _is_planning_ct_path(path)
    )

    if len(candidates) == 0:
        candidates = sorted(
            Path(path)
            for path in glob.glob(str(config.processed_ct_dir / "**" / "*.pt"), recursive=True)
            if _extract_patient_id(path) == patient_id and _is_planning_ct_path(path)
        )

    if len(candidates) == 0:
        return None

    return candidates[0]


def _warp_planning_masks_for_predicted_dose(
    pred_path: Path,
    pred_dose: torch.Tensor,
    patient_id: str,
    generated_ct_lookup: dict[str, Path],
    structure_transform: Compose,
    planning_ct_cache: dict[str, torch.Tensor],
    planning_label_map_cache: dict[str, torch.Tensor],
    generated_mask_cache: WarpedStructureMaskCache,
    config: MaisiTestingConfig,
) -> tuple[dict[int, torch.Tensor] | None, str, str | None, str | None]:
    """
    Warp planning structures into generated-dose space when possible.

    Predicted doses produced from generated CTs should be evaluated with masks
    on that generated anatomy. This helper resolves the matching generated CT,
    registers the planning CT to it, and warps planning structure masks through
    the resulting DVF. If the generated CT or planning inputs cannot be
    resolved, callers can fall back to same-grid reference masks.

    Parameters
    ----------
    pred_path : Path
        Predicted dose path.

    pred_dose : torch.Tensor
        Predicted dose tensor on CPU.

    patient_id : str
        Patient identifier.

    generated_ct_lookup : dict[str, Path]
        Generated CT paths keyed by normalized filename.

    structure_transform : Compose
        Preprocessing transform for structure label maps.

    planning_ct_cache : dict[str, torch.Tensor]
        CPU cache of processed planning CT tensors keyed by patient ID.

    planning_label_map_cache : dict[str, torch.Tensor]
        CPU cache of planning structure label maps keyed by patient ID.

    generated_mask_cache : WarpedStructureMaskCache
        CPU cache of warped planning masks keyed by generated CT path.

    config : MaisiTestingConfig
        Pipeline configuration containing CT, structure, and registration
        settings.

    Returns
    -------
    result : tuple[dict[int, torch.Tensor] | None, str, str | None, str | None]
        Warped masks keyed by structure label, a source description, an
        optional info message, and an optional warning. `None` masks mean no
        generated-space structure could be produced.
    """

    if not config.use_dose_structure_warping:
        return None, "reference_structure_same_grid", None, None

    gen_ct_path = generated_ct_lookup.get(_normalised_image_key(pred_path))
    if gen_ct_path is None:
        return (
            None,
            "reference_structure_same_grid",
            None,
            f"Using reference-space dose masks for {pred_path}: no matching generated CT found",
        )

    cached_masks = generated_mask_cache.get(gen_ct_path)
    if cached_masks is not None:
        return cached_masks, "warped_planning_structure_to_generated_ct", None, None

    persisted_masks = load_warped_structure_masks(config, gen_ct_path)
    if persisted_masks is not None:
        generated_mask_cache[gen_ct_path] = persisted_masks
        return persisted_masks, "persisted_warped_planning_structure_to_generated_ct", None, None

    planning_ct_path = _planning_ct_path_for_patient(patient_id, config)
    if planning_ct_path is None:
        return (
            None,
            "reference_structure_same_grid",
            None,
            f"Using reference-space dose masks for {pred_path}: no processed planning CT found for {patient_id}",
        )

    planning_structure_path = _structure_path_for_ct(planning_ct_path, config)
    if not planning_structure_path.exists():
        return (
            None,
            "reference_structure_same_grid",
            None,
            f"Using reference-space dose masks for {pred_path}: missing {planning_structure_path}",
        )

    cpu_device = torch.device("cpu")
    planning_ct = planning_ct_cache.get(patient_id)
    if planning_ct is None:
        planning_ct = _load_hu_tensor(planning_ct_path, config=config, device=cpu_device)
        planning_ct_cache[patient_id] = planning_ct

    planning_label_map = planning_label_map_cache.get(patient_id)
    if planning_label_map is None:
        planning_label_map = _load_structure_label_map(planning_structure_path, structure_transform)
        planning_label_map_cache[patient_id] = planning_label_map

    result = _calculate_structure_registration_result(
        gen_path=gen_ct_path,
        patient_id=patient_id,
        planning_ct=planning_ct,
        config=config,
    )

    if result.warning is not None:
        return None, "reference_structure_same_grid", None, result.warning

    if result.gen_ct is None or result.transform is None:
        return (
            None,
            "reference_structure_same_grid",
            None,
            f"Using reference-space dose masks for {pred_path}: generated CT registration failed",
        )

    if result.gen_ct.shape != pred_dose.shape:
        return (
            None,
            "reference_structure_same_grid",
            None,
            "Using reference-space dose masks for "
            f"{pred_path}: generated CT {tuple(result.gen_ct.shape)} vs dose {tuple(pred_dose.shape)}",
        )

    warped_masks: dict[int, torch.Tensor] = {}
    for label in config.structure_labels:
        # Warp one planning label at a time with nearest-neighbor sampling so
        # target/OAR identities stay categorical after DVF resampling.
        warped_masks[label] = _warp_planning_mask_to_generated_ct(
            planning_mask=_label_to_binary_mask(planning_label_map, label),
            generated_ct=result.gen_ct,
            transform=result.transform,
            config=config,
        )

    generated_mask_cache[gen_ct_path] = warped_masks
    save_warped_structure_masks(config, gen_ct_path, warped_masks)

    return warped_masks, "warped_planning_structure_to_generated_ct", result.info_message, None


def _load_dose_comparison_data(
    pred_path: Path,
    ref_path: Path,
    patient_id: str,
    dose_transform: Compose,
    structure_transform: Compose,
    generated_ct_lookup: dict[str, Path],
    reference_dose_cache: dict[Path, torch.Tensor],
    structure_cache: dict[Path, torch.Tensor | None],
    planning_ct_cache: dict[str, torch.Tensor],
    planning_label_map_cache: dict[str, torch.Tensor],
    generated_mask_cache: WarpedStructureMaskCache,
    config: MaisiTestingConfig,
) -> _DoseComparisonData:
    """
    Load CPU data for one dose comparison and reuse patient-level caches.

    Predicted dose files are unique per comparison and are loaded on demand.
    Reference dose and reference structure files can be reused across multiple
    predictions for the same patient, so they are cached on CPU. Returning a
    structured result instead of raising for expected data issues lets the
    outer evaluator preserve the skip-and-continue behavior used elsewhere in
    the pipeline.

    Parameters
    ----------
    pred_path : Path
        Predicted dose distribution path.

    ref_path : Path
        Matching reference dose distribution path.

    patient_id : str
        Patient identifier used in diagnostic messages.

    dose_transform : Compose
        Preprocessing transform for NIfTI dose maps.

    structure_transform : Compose
        Preprocessing transform for structure label maps.

    generated_ct_lookup : dict[str, Path]
        Generated CT tensor paths keyed by normalized filename. Used to warp
        planning structures into generated-dose space.

    reference_dose_cache : dict[Path, torch.Tensor]
        CPU cache of reference dose tensors for the current patient.

    structure_cache : dict[Path, torch.Tensor | None]
        CPU cache of reference structure label maps for the current patient.
        Missing or shape-mismatched structures are cached as `None`.

    planning_ct_cache : dict[str, torch.Tensor]
        CPU cache of planning CT tensors keyed by patient ID.

    planning_label_map_cache : dict[str, torch.Tensor]
        CPU cache of planning structure label maps keyed by patient ID.

    generated_mask_cache : WarpedStructureMaskCache
        CPU cache of DVF-warped generated-space masks keyed by generated CT.

    config : MaisiTestingConfig
        Pipeline configuration containing paths and preprocessing settings.

    Returns
    -------
    result : _DoseComparisonData
        CPU-loaded comparison data, or a result containing a warning when the
        comparison should be skipped.
    """

    pred_dose = load_dose_distribution(pred_path, dose_transform)
    ref_dose = reference_dose_cache.get(ref_path)
    if ref_dose is None:
        ref_dose = load_dose_distribution(ref_path, dose_transform)
        reference_dose_cache[ref_path] = ref_dose

    if pred_dose.shape != ref_dose.shape:
        return _DoseComparisonData(
            pred_path=pred_path,
            ref_path=ref_path,
            pred_dose=None,
            ref_dose=None,
            ref_label_map=None,
            pred_masks=None,
            pred_structure_source="none",
            ref_structure_source="none",
            warning=(
                "Skipping dose shape mismatch for "
                f"{patient_id}: {pred_path.name} {tuple(pred_dose.shape)} vs "
                f"{ref_path.name} {tuple(ref_dose.shape)}"
            ),
        )

    if ref_path not in structure_cache:
        structure_path = _structure_path_for_dose(ref_path, config)
        if not structure_path.exists():
            structure_cache[ref_path] = None
            structure_warning = f"Skipping structure-specific dose metrics for {patient_id}: missing {structure_path}"
        else:
            label_map = _load_structure_label_map(structure_path, structure_transform)
            if label_map.shape != pred_dose.shape:
                structure_cache[ref_path] = None
                structure_warning = (
                    "Skipping structure-specific dose metrics for "
                    f"{patient_id}: dose {tuple(pred_dose.shape)} vs structures {tuple(label_map.shape)}"
                )
            else:
                structure_cache[ref_path] = label_map
                structure_warning = None
    else:
        structure_warning = None

    pred_masks, pred_structure_source, pred_mask_info, pred_mask_warning = _warp_planning_masks_for_predicted_dose(
        pred_path=pred_path,
        pred_dose=pred_dose,
        patient_id=patient_id,
        generated_ct_lookup=generated_ct_lookup,
        structure_transform=structure_transform,
        planning_ct_cache=planning_ct_cache,
        planning_label_map_cache=planning_label_map_cache,
        generated_mask_cache=generated_mask_cache,
        config=config,
    )

    return _DoseComparisonData(
        pred_path=pred_path,
        ref_path=ref_path,
        pred_dose=pred_dose,
        ref_dose=ref_dose,
        ref_label_map=structure_cache[ref_path],
        pred_masks=pred_masks,
        pred_structure_source=pred_structure_source,
        ref_structure_source="reference_structure",
        info_message=pred_mask_info,
        nonfatal_warning=structure_warning or pred_mask_warning,
    )


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
                if torch.equal(pred_mask, ref_mask):
                    voxel_mae, voxel_rmse = _masked_error_metrics(pred_dose, ref_dose, pred_mask)
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
                        _metric_row("mean_dose", mean_dose(pred_dose, pred_mask), mean_dose(ref_dose, ref_mask), base),
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
        predicted = collect_dose_distributions(config.predicted_dose_dir)
        references = collect_dose_distributions(config.reference_dose_dir)
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
