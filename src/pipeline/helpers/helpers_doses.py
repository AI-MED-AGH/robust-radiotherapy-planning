import glob
import logging
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

from src.pipeline.config import MaisiTestingConfig
from src.pipeline.helpers.helpers import (
    _extract_patient_id,
    _is_planning_ct_path,
    _load_hu_tensor,
    _load_tensor,
)
from src.pipeline.helpers.helpers_structures import (
    WarpedStructureMaskCache,
    _calculate_structure_registration_result,
    _label_to_binary_mask,
    _load_structure_label_map,
    _load_warped_structure_masks,
    _save_warped_structure_masks,
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

    # Dose and structure files share the patient/scenario stem even when their
    # extensions differ, so the normalized dose name is also the label-map key
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

    # Apply only geometric operations: unlike CT preprocessing, dose values
    # must remain in physical units (normally Gy) for clinical interpretation
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
    # Drop MONAI metadata and the singleton channel before caching on CPU
    return dose_meta.as_tensor().detach().cpu()[0].float()


def _masked_dose_values(dose: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Select dose values belonging to a binary structure mask.

    The dose is converted to a detached floating-point tensor without changing
    the device of an existing tensor. The mask is converted to Boolean on that
    same device, after which Boolean indexing returns the selected values as a
    one-dimensional tensor.

    Parameters
    ----------
    dose : torch.Tensor | np.ndarray
        Dose distribution in physical dose units, normally Gy.

    mask : torch.Tensor | np.ndarray
        Binary mask whose shape must exactly match ``dose``. Nonzero values are
        treated as foreground.

    Returns
    -------
    values : torch.Tensor
        Detached one-dimensional floating-point tensor containing the dose at
        every foreground mask voxel. An empty mask returns an empty tensor.

    Raises
    ------
    ValueError
        If ``dose`` and ``mask`` have different shapes.
    """

    dose_t = torch.as_tensor(dose).detach().float()
    mask_t = torch.as_tensor(mask, device=dose_t.device).bool()
    if dose_t.shape != mask_t.shape:
        raise ValueError(f"Dose and mask must have the same shape. Got {tuple(dose_t.shape)} and {tuple(mask_t.shape)}")
    return dose_t[mask_t]


def _masked_error_metrics_from_values(pred_values: torch.Tensor, ref_values: torch.Tensor) -> tuple[float, float]:
    """
    Calculate MAE and RMSE from already-masked predicted/reference values.

    Parameters
    ----------
    pred_values : torch.Tensor
        Predicted dose values selected by the comparison mask.

    ref_values : torch.Tensor
        Reference dose values selected by the same comparison mask.

    Returns
    -------
    metrics : tuple[float, float]
        Pair containing masked voxel MAE and RMSE. Empty inputs return
        `(nan, nan)`.
    """

    if pred_values.numel() == 0:
        return float("nan"), float("nan")

    # Callers only use this helper when both vectors come from the same mask,
    # preserving one-to-one spatial correspondence between selected voxels
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

    # Predicted dose and generated CT files use the same scenario stem, which
    # later connects the dose grid to its generated anatomy
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


def _load_dose_distribution(path: Path, transform: Compose) -> torch.Tensor:
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

    # Tensor outputs are already on the evaluation grid; NIfTI inputs need the
    # shared geometric transform used to align dose and structure volumes
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


def _collect_single_dose_per_patient(dose_dir: Path, *, planning_only: bool) -> dict[str, Path]:
    """Collect one unambiguous dose distribution for every patient.

    The directory is searched recursively for supported dose files (`.nii`,
    `.nii.gz`, and `.pt`). Each retained path is assigned to a patient using
    :func:`_extract_patient_id`. The helper enforces a one-dose-per-patient
    contract and returns individual paths rather than lists of scenario paths.

    When ``planning_only`` is true, files without the ``fraction_1_`` planning
    marker are ignored. This is intended for clinical planning-dose discovery:
    follow-up fractions may coexist in the same directory but do not make the
    planning dose ambiguous. When it is false, every supported file is treated
    as a candidate, so a second file for the same patient is an error.

    Parameters
    ----------
    dose_dir : Path
        Root directory searched recursively for dose distributions.

    planning_only : bool
        If true, retain only paths identified by
        :func:`_is_planning_ct_path`; otherwise, retain every supported dose
        file.

    Returns
    -------
    doses : dict[str, Path]
        Mapping from patient ID to its single retained dose path. Both patient
        discovery and path ordering are deterministic because paths are
        processed in sorted order.

    Raises
    ------
    FileNotFoundError
        If ``dose_dir`` does not exist.

    ValueError
        If the recursive search finds no relevant dose files, or if more than
        one retained file resolves to the same patient ID.
    """

    if not dose_dir.exists():
        raise FileNotFoundError(f"Dose directory does not exist: {dose_dir}")

    doses: dict[str, Path] = {}
    for path in sorted(dose_dir.rglob("*")):
        if not path.is_file() or not path.name.endswith((".nii", ".nii.gz", ".pt")):
            continue
        if planning_only and not _is_planning_ct_path(path):
            continue

        patient_id = _extract_patient_id(path)
        previous = doses.get(patient_id)
        if previous is not None:
            dose_kind = "fraction-1 clinical" if planning_only else "candidate"
            raise ValueError(f"Multiple {dose_kind} doses found for {patient_id}: {previous} and {path}")
        doses[patient_id] = path

    if not doses:
        dose_kind = "fraction-1 clinical" if planning_only else "candidate"
        raise ValueError(f"No {dose_kind} dose files found in: {dose_dir}")

    return doses


def _contains_dose_files(directory: Path) -> bool:
    """Check whether a directory contains a supported dose distribution.

    Parameters
    ----------
    directory : Path
        Root directory searched recursively for NIfTI or PyTorch dose files.

    Returns
    -------
    contains_dose : bool
        ``True`` when at least one ``.nii``, ``.nii.gz``, or ``.pt`` file is
        present; otherwise, ``False``. A missing directory returns ``False``.
    """

    return directory.exists() and any(
        path.is_file() and path.name.endswith((".nii", ".nii.gz", ".pt")) for path in directory.rglob("*")
    )


def _planning_masks_for_dose(
    dose_path: Path,
    structure_transform: Compose,
    config: MaisiTestingConfig,
) -> dict[int, torch.Tensor] | None:
    """Load configured fraction-1 structure masks for a dose file.

    Parameters
    ----------
    dose_path : Path
        Clinical dose path used to resolve the matching structure label map.

    structure_transform : Compose
        Label-preserving preprocessing transform applied to the structure map.

    config : MaisiTestingConfig
        Configuration containing the structures root and evaluated labels.

    Returns
    -------
    masks : dict[int, torch.Tensor] | None
        Binary masks keyed by configured label, or ``None`` when the matching
        structure file is unavailable.
    """

    structure_path = _structure_path_for_dose(dose_path, config)
    if not structure_path.exists():
        logger.warning("Skipping planning-structure dose metrics: missing %s", structure_path)
        return None
    label_map = _load_structure_label_map(structure_path, structure_transform)
    # Split the categorical map into independent boolean masks so downstream
    # calculations can iterate only over labels requested by configuration
    return {label: _label_to_binary_mask(label_map, label) for label in config.structure_labels}


def _write_scenario_outputs(rows: list[dict[str, Any]], config: MaisiTestingConfig, stem: str) -> None:
    """Write detailed scenario metrics and their across-scenario summary.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Long-form scenario metric records.

    config : MaisiTestingConfig
        Configuration containing the metrics output directory.

    stem : str
        Filename stem used for the detailed and summary CSV files.

    Returns
    -------
    None
        CSV files are written under ``config.metrics_dir``. Empty inputs are
        logged and produce no files.

    Raises
    ------
    ValueError
        If non-empty input rows omit columns required for the scenario
        summary.
    """

    if not rows:
        logger.warning("Skipping %s: no valid dose/anatomy scenarios", stem)
        return

    frame = pd.DataFrame(rows)
    group_columns = ["dose_source", "patient_id", "scenario_type", "structure_label", "metric"]
    required_columns = [*group_columns, "value"]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Cannot write {stem} scenario outputs: missing required columns {missing_columns}")

    frame.to_csv(config.metrics_dir / f"{stem}.csv", index=False)
    summary = (
        frame.groupby(group_columns, dropna=False)["value"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    )
    summary.to_csv(config.metrics_dir / f"{stem}_summary.csv", index=False)


def _write_smoke_test_outputs(rows: list[dict[str, Any]], config: MaisiTestingConfig) -> None:
    """Write base-dose comparisons against original-anatomy metric values.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Long-form original- and generated-anatomy metric records.

    config : MaisiTestingConfig
        Configuration containing the metrics output directory.

    Returns
    -------
    None
        Detailed and summary CSV files are written under
        ``config.metrics_dir``. Relative differences are undefined when the
        original value is zero.

    Raises
    ------
    KeyError
        If non-empty input rows omit a column required for comparison.

    pandas.errors.MergeError
        If more than one original-anatomy row exists for the same patient,
        structure label, and metric.
    """

    if not rows:
        logger.warning("Skipping base dose smoke test: no valid dose/anatomy scenarios")
        return

    frame = pd.DataFrame(rows)
    key_columns = ["patient_id", "structure_label", "metric"]
    original = frame.loc[frame["scenario_type"] == "original_anatomy", key_columns + ["value"]].rename(
        columns={"value": "original_anatomy_value"}
    )
    comparison = frame.merge(original, on=key_columns, how="left", validate="many_to_one")
    comparison["difference_from_original"] = comparison["value"] - comparison["original_anatomy_value"]
    comparison["absolute_difference_from_original"] = comparison["difference_from_original"].abs()
    denominator = comparison["original_anatomy_value"].abs().replace(0.0, np.nan)
    comparison["relative_difference_percent"] = 100.0 * comparison["difference_from_original"] / denominator
    comparison.to_csv(config.metrics_dir / "base_dose_smoke_test.csv", index=False)

    alternatives = comparison.loc[comparison["scenario_type"] != "original_anatomy"]
    if alternatives.empty:
        logger.warning("Base dose smoke test contains no alternative-anatomy comparisons")
        return
    summary = (
        alternatives.groupby([*key_columns, "scenario_type"], dropna=False)
        .agg(
            original_anatomy_value=("original_anatomy_value", "first"),
            alternative_mean=("value", "mean"),
            alternative_std=("value", "std"),
            alternative_min=("value", "min"),
            alternative_max=("value", "max"),
            mean_absolute_difference=("absolute_difference_from_original", "mean"),
            maximum_absolute_difference=("absolute_difference_from_original", "max"),
            mean_absolute_relative_difference_percent=(
                "relative_difference_percent",
                lambda values: values.abs().mean(),
            ),
            scenario_count=("value", "count"),
        )
        .reset_index()
    )
    summary.to_csv(config.metrics_dir / "base_dose_smoke_test_summary.csv", index=False)


def _select_dose_for_anatomy(
    patient_dose: Path | None,
    clinical_path: Path | None,
    dose_source: str,
    force_clinical: bool = False,
) -> tuple[Path | None, str]:
    """Select a patient's single candidate dose or clinical fallback."""

    if force_clinical:
        return clinical_path, "clinical_fraction_1"

    if patient_dose is not None:
        return patient_dose, f"{dose_source}_patient_dose"

    if clinical_path is not None:
        return clinical_path, "clinical_fraction_1_fallback"

    return None, "none"


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
        # Returning no predicted masks tells the comparison caller to use the
        # reference masks on the shared grid instead
        return None, "reference_structure_same_grid", None, None

    gen_ct_path = generated_ct_lookup.get(_normalised_image_key(pred_path))
    if gen_ct_path is None:
        return (
            None,
            "reference_structure_same_grid",
            None,
            f"Using reference-space dose masks for {pred_path}: no matching generated CT found",
        )

    # Check the current run's cache before touching disk or running deformable
    # registration, which is the most expensive path through this helper
    cached_masks = generated_mask_cache.get(gen_ct_path)
    if cached_masks is not None:
        return cached_masks, "warped_planning_structure_to_generated_ct", None, None

    persisted_masks = _load_warped_structure_masks(config, gen_ct_path)
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

    # Registration estimates the planning-to-generated deformation shared by
    # every configured structure label for this generated CT.
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

    # Persist successful warps so both later dose comparisons and future runs
    # can bypass registration for this generated anatomy
    generated_mask_cache[gen_ct_path] = warped_masks
    _save_warped_structure_masks(config, gen_ct_path, warped_masks)

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

    # Predicted doses are unique per pair, whereas a reference is commonly
    # reused by several generated scenarios and therefore benefits from caching
    pred_dose = _load_dose_distribution(pred_path, dose_transform)
    ref_dose = reference_dose_cache.get(ref_path)
    if ref_dose is None:
        ref_dose = _load_dose_distribution(ref_path, dose_transform)
        reference_dose_cache[ref_path] = ref_dose

    # Overall voxel metrics require identical grids; represent this expected
    # data issue in the result so the outer patient loop can continue.
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

    # Cache `None` as well as valid label maps to avoid repeating failed file
    # lookups or shape checks for every prediction sharing this reference.
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

    # Resolve anatomy-specific masks independently of the reference label map:
    # missing warps are nonfatal because callers can use same-grid ref masks
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

    nonfatal_warnings = [message for message in (structure_warning, pred_mask_warning) if message is not None]

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
        nonfatal_warning="\n".join(nonfatal_warnings) if len(nonfatal_warnings) > 0 else None,
    )
