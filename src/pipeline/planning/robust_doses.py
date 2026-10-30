import copy
import hashlib
import json
import logging
import re
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch

from src.pipeline.config import MaisiTestingConfig
from src.pipeline.helpers.helpers import _extract_patient_id, _is_planning_ct_path, _load_hu_tensor, _load_tensor
from src.pipeline.helpers.helpers_structures import (
    _register_planning_ct_to_generated_ct,
    _warp_planning_mask_to_generated_ct,
)
from src.pipeline.planning.geometry import (
    labels_on_grid,
    load_ct,
    make_nifti,
    mask_image,
    pipeline_grid,
    resample_to_grid,
)
from src.pipeline.planning.settings import OpenTPSSettings

logger = logging.getLogger(__name__)


def _run_opentps_worker(request_path: Path) -> dict[str, Any]:
    """
    Run one patient's OpenTPS request in a separate Python process.

    The worker uses the current Python executable and the repository root as
    its working directory. Process isolation keeps OpenTPS's numerical
    dependencies separate from PyTorch loaded by the parent pipeline.
    Standard output and errors are written to a sibling ``opentps.log``.

    Parameters
    ----------
    request_path : Path
        JSON request identifying the patient's settings, anatomies and outputs.
        The worker writes ``result.json`` in the same directory.

    Returns
    -------
    dict[str, Any]
        Worker result containing ``plans`` and ``doses`` metadata. Dose arrays
        remain in the NIfTI files referenced by these records.

    Raises
    ------
    RuntimeError
        If the worker exits with a nonzero status. The error includes the log
        path for diagnosing the failure.

    OSError
        If the log or result file cannot be accessed, or the process cannot
        be started.

    json.JSONDecodeError
        If the worker result is not valid JSON.
    """

    log_path = request_path.with_name("opentps.log")
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [sys.executable, "-m", "src.pipeline.planning.worker", str(request_path.resolve())],
            cwd=Path(__file__).resolve().parents[3],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"OpenTPS worker failed (exit {result.returncode}); see {log_path.resolve()}")
    value: dict[str, Any] = json.loads(request_path.with_name("result.json").read_text(encoding="utf-8"))
    return value


@dataclass
class _PatientInputs:
    """
    Store validated input paths and scenario selections for one patient.

    Attributes
    ----------
    patient_id : str
        Patient identifier extracted from prepared CT paths.

    planning_tensor : Path
        Prepared planning CT tensor used for registration to generated CTs.

    planning_ct : Path
        Original planning CT NIfTI supplying the full anatomy and geometry.

    real_cts : list[tuple[Path, Path]]
        Original CT and structure-label-map pairs for the prepared fractions,
        including the planning fraction. All receive dose recalculation.

    generated : dict[int, Path]
        Available generated CT tensor paths keyed by variant index.

    selected : dict[str, list[int]]
        Generated variant indices for each named scenario set. The worker
        adds the planning anatomy to every set during optimization.
    """

    patient_id: str
    planning_tensor: Path
    planning_ct: Path
    real_cts: list[tuple[Path, Path]]
    generated: dict[int, Path]
    selected: dict[str, list[int]]


def _nifti(root: Path, patient: str, stem: str) -> Path:
    """
    Find exactly one compressed or uncompressed NIfTI for a patient fraction.

    Parameters
    ----------
    root : Path
        Root directory containing patient subdirectories.

    patient : str
        Patient subdirectory name.

    stem : str
        Fraction filename without the ``.nii`` or ``.nii.gz`` extension.

    Returns
    -------
    Path
        Existing file at ``root / patient`` with the requested stem.

    Raises
    ------
    ValueError
        If neither extension exists or both extensions exist.
    """
    
    matches = [path for suffix in (".nii.gz", ".nii") if (path := root / patient / f"{stem}{suffix}").is_file()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one NIfTI for {patient}/{stem} in {root}; found {len(matches)}")
    return matches[0]


def collect_planning_inputs(config: MaisiTestingConfig, settings: OpenTPSSettings) -> list[_PatientInputs]:
    """
    Collect planning inputs for the patients and fractions in the prepared split.

    Prepared CT tensors determine which original CTs and structure maps are
    included. Generated tensors are matched by ``<patient>_gen_<index>.pt``;
    an ``"all"`` scenario selection expands to all available variant indices.
    This function resolves paths without loading image or tensor contents.

    Parameters
    ----------
    config : MaisiTestingConfig
        Pipeline configuration supplying prepared, generated, original CT and
        structure directories.

    settings : OpenTPSSettings
        Planning settings defining the named generated-anatomy scenario sets.

    Returns
    -------
    list[_PatientInputs]
        Input records containing each patient's planning CT, real fractions,
        available generated tensors and resolved scenario selections.

    Raises
    ------
    ValueError
        If no prepared CTs exist, a patient lacks exactly one planning CT or
        at least one follow-up CT, generated indices are duplicated or missing,
        an ``"all"`` selection is empty, or an original CT or structure map
        does not have exactly one matching NIfTI file.
    """

    patients: dict[str, list[Path]] = {}
    for path in sorted(config.processed_ct_dir.rglob("*.pt")):
        patients.setdefault(_extract_patient_id(path), []).append(path)
    if not patients:
        raise ValueError(f"No prepared CTs in {config.processed_ct_dir}; run 'prepare' first")
    generated: dict[str, dict[int, Path]] = {}
    for path in sorted(config.generated_ct_dir.rglob("*.pt")):
        match = re.fullmatch(r"(.+)_gen_(\d+)\.pt", path.name)
        if match:
            patient, index = match[1], int(match[2])
            variants = generated.setdefault(patient, {})
            if index in variants:
                raise ValueError(f"Duplicate generated variant {index} for {patient}")
            variants[index] = path
    inputs = []
    for patient, paths in patients.items():
        planning = [path for path in paths if _is_planning_ct_path(path)]
        if len(planning) != 1 or not any(not _is_planning_ct_path(path) for path in paths):
            raise ValueError(f"{patient} needs exactly one planning CT and at least one real follow-up CT")
        variants = generated.get(patient, {})
        selected = {}
        for name, selection in settings.scenario_sets.items():
            indices = sorted(variants) if selection == "all" else selection
            if selection == "all" and not indices:
                raise ValueError(f"No generated anatomy scenarios for {patient}")
            missing = set(indices) - variants.keys()
            if missing:
                raise ValueError(f"Missing generated variants {sorted(missing)} for {patient}, scenario set {name}")
            selected[name] = list(indices)
        real_cts = []
        for path in paths:
            ct = _nifti(config.ct_root, patient, path.stem)
            structures = _nifti(config.structures_root, patient, path.stem)
            real_cts.append((ct, structures))
        planning_ct = _nifti(config.ct_root, patient, planning[0].stem)
        inputs.append(_PatientInputs(patient, planning[0], planning_ct, real_cts, variants, selected))
    return inputs


def _generated_anatomy(
    path: Path,
    planning: Any,
    planning_masks: dict[int, Any],
    planning_tensor: torch.Tensor,
    grid: Any,
    config: MaisiTestingConfig,
) -> tuple[Any, dict[int, Any]]:
    """
    Build a full CT and deformed structure masks for one generated anatomy.

    The generated HU patch is resampled onto the full planning CT grid using
    linear interpolation. Planning CT values outside the patch coverage are
    retained to preserve anatomy along beam paths. Planning masks are
    registered to the generated patch, resampled with nearest-neighbour
    interpolation, and combined with the original masks outside coverage.
    Registration is recomputed for this call without using evaluation caches.

    Parameters
    ----------
    path : Path
        Generated CT tensor file containing HU values on ``grid``.

    planning : nibabel.spatialimages.SpatialImage
        Full planning CT supplying the output grid and surrounding anatomy.

    planning_masks : dict[int, nibabel.spatialimages.SpatialImage]
        Binary optimization masks keyed by label, on the planning CT grid.

    planning_tensor : torch.Tensor
        Prepared planning CT in HU on the generated patch grid, used as the
        source image for registration.

    grid : nibabel.spatialimages.SpatialImage
        Geometry of the pipeline's padded and centre-cropped tensor.

    config : MaisiTestingConfig
        HU bounds and registration settings. A shallow copy receives the
        patch grid's voxel spacing; the supplied configuration is unchanged.

    Returns
    -------
    image : nibabel.Nifti1Image
        Generated anatomy embedded in the full planning CT grid.

    masks : dict[int, nibabel.Nifti1Image]
        Corresponding structure masks on the same full grid.

    Raises
    ------
    ValueError
        If the generated tensor has an incorrect shape, nonfinite or
        out-of-range HU values, or a warped mask has an incorrect shape.
    """

    generated = _load_tensor(path)
    if tuple(generated.shape) != grid.shape or not torch.isfinite(generated).all():
        raise ValueError(f"Generated CT must contain finite HU values on the planning tensor grid: {path}")
    if generated.min() < config.data_min or generated.max() > config.data_max:
        raise ValueError(f"Generated CT values are outside the configured HU range: {path}")
    # Retain uncropped planning anatomy along the beam path outside the model's
    # field of view. Only the generated patch replaces the corresponding CT.
    patch = make_nifti(generated.numpy(), grid.affine)
    coverage = np.asarray(resample_to_grid(mask_image(np.ones(grid.shape), grid), planning, order=0).dataobj) > 0
    warped_patch = np.asarray(resample_to_grid(patch, planning, order=1).dataobj)
    image = make_nifti(np.where(coverage, warped_patch, np.asarray(planning.dataobj)), planning.affine)

    # Existing evaluation caches are keyed only by filename, so they can be
    # stale after changing the model. Recompute masks for this planning run.
    registration_config = copy.copy(config)
    voxel_spacing = grid.header.get_zooms()
    registration_config.spacing = (float(voxel_spacing[0]), float(voxel_spacing[1]), float(voxel_spacing[2]))
    transform = _register_planning_ct_to_generated_ct(planning_tensor, generated, registration_config)
    warped = {}
    for label, mask in planning_masks.items():
        patch_mask = np.asarray(resample_to_grid(mask, grid, order=0).dataobj).copy()
        warped[label] = _warp_planning_mask_to_generated_ct(
            torch.from_numpy(patch_mask), generated, transform, registration_config
        )
    masks = {}
    for label, values in warped.items():
        if tuple(values.shape) != grid.shape:
            raise ValueError(f"Warped structure {label} has incorrect grid for {path}")
        patch_mask = mask_image(values.numpy(), grid)
        full = np.asarray(resample_to_grid(patch_mask, planning, order=0).dataobj) > 0
        masks[label] = mask_image(np.where(coverage, full, np.asarray(planning_masks[label].dataobj)), planning)
    return image, masks


def generate_robust_doses(config: MaisiTestingConfig) -> Path:
    """
    Optimize robust plans and recalculate their doses on generated and real CTs.

    For each patient, prepare the planning anatomy and selected generated
    anatomies with their structure masks, then dispatch a file-based request
    to an isolated OpenTPS worker. Each named scenario set produces one plan
    optimized on the planning CT and that set's generated CTs. The fixed plan
    is MCsquare-recalculated on the selected generated CTs and every prepared
    real fraction, including the planning fraction. Real follow-up scans never
    contribute to plan optimization.

    Settings, anatomy files, requests, logs, plans and doses are stored in a
    unique run directory. The manifest records settings, generated input
    SHA-256 hashes, plan metadata and dose paths.
    A completed manifest is replaced atomically only after every plan and dose
    succeeds. Older successful runs remain usable if a subsequent run fails.
    Partial files from a failed run remain in its run directory.

    Parameters
    ----------
    config : MaisiTestingConfig
        Pipeline configuration providing the OpenTPS settings path, input
        directories, tensor geometry, HU bounds and robust-dose output root.

    Returns
    -------
    Path
        Path to the completed ``manifest.json`` in ``config.robust_dose_dir``.

    Raises
    ------
    ValueError
        If OpenTPS configuration is missing or invalid, required patient
        inputs are invalid, an optimization structure is empty, or tensor
        shapes or generated HU values are invalid.

    RuntimeError
        If an OpenTPS worker fails during planning or dose recalculation.

    OSError
        If settings, inputs or outputs cannot be accessed or written.
    """
    
    if config.opentps_config_path is None:
        raise ValueError("Provide --opentps-config with machine, prescription and scenario settings")
    settings = OpenTPSSettings.load(config.opentps_config_path)
    inputs = collect_planning_inputs(config, settings)
    run_id = uuid.uuid4().hex
    run_dir = config.robust_dose_dir / f"run-{run_id}"
    run_dir.mkdir(parents=True)
    settings_snapshot = (run_dir / "settings.json").resolve()
    settings_snapshot.write_text(json.dumps(asdict(settings), indent=2, default=str), encoding="utf-8")
    records: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    sources: dict[str, str] = {}
    required_labels = sorted({settings.target_label, *settings.oar_limits_gy})
    for patient in inputs:
        logger.info("OpenTPS: generating plans for %s using %s", patient.patient_id, settings.model_name)
        planning = load_ct(patient.planning_ct)
        structure_path = next(structure for ct, structure in patient.real_cts if ct == patient.planning_ct)
        label_map = labels_on_grid(structure_path, planning)
        planning_masks = {label: mask_image(label_map == label, planning) for label in required_labels}
        for label, mask in planning_masks.items():
            if not np.any(np.asarray(mask.dataobj)):
                raise ValueError(f"Required optimization structure {label} is empty for {patient.patient_id}")
        grid = pipeline_grid(planning, config.target_image_size)
        tensor = _load_hu_tensor(patient.planning_tensor, config, torch.device("cpu"))
        if tuple(tensor.shape) != grid.shape:
            raise ValueError(f"Prepared planning CT shape differs from configured target size for {patient.patient_id}")
        anatomies: dict[int, tuple[Any, dict[int, Any]]] = {0: (planning, planning_masks)}
        for index in sorted({index for indices in patient.selected.values() for index in indices}):
            source = patient.generated[index]
            with source.open("rb") as stream:
                sources[str(source.resolve())] = hashlib.file_digest(stream, "sha256").hexdigest()
            anatomies[index] = _generated_anatomy(source, planning, planning_masks, tensor, grid, config)
        directory = (run_dir / patient.patient_id).resolve()
        serialized = {}
        for index, (image, masks) in anatomies.items():
            anatomy_dir = directory / "anatomies" / str(index)
            anatomy_dir.mkdir(parents=True)
            ct_output = anatomy_dir / "ct.nii.gz"
            nib.save(image, ct_output)
            mask_paths = {}
            for label, mask in masks.items():
                mask_output = anatomy_dir / f"label_{label}.nii.gz"
                nib.save(mask, mask_output)
                mask_paths[str(label)] = str(mask_output)
            serialized[str(index)] = {"ct_path": str(ct_output), "masks": mask_paths}
        request = {
            "patient_id": patient.patient_id,
            "patient_dir": str(directory),
            "output_root": str(config.robust_dose_dir.resolve()),
            "settings_path": str(settings_snapshot),
            "scenario_sets": patient.selected,
            "anatomies": serialized,
            "real_cts": [
                {
                    "ct_path": str(ct.resolve()),
                    "structure_path": str(structure.resolve()),
                    "scenario_type": "original_anatomy" if _is_planning_ct_path(ct) else "real_fraction_anatomy",
                }
                for ct, structure in patient.real_cts
            ],
        }
        request_path = directory / "request.json"
        request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
        result = _run_opentps_worker(request_path)
        plans.extend(result["plans"])
        records.extend(result["doses"])
    manifest = {
        "schema_version": 1,
        "engine": "opentps-core==3.0.1/MCsquare",
        "run_id": run_id,
        "model_name": settings.model_name,
        "generated_ct_dir": str(config.generated_ct_dir.resolve()),
        "rflow_weight_path": str(config.rflow_weight_path.resolve()),
        "settings": asdict(settings),
        "generated_ct_sha256": sources,
        "plans": plans,
        "doses": records,
    }
    temporary = config.robust_dose_dir / f"manifest-{run_id}.json"
    temporary.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    output_manifest = config.robust_dose_dir / "manifest.json"
    temporary.replace(output_manifest)
    logger.info("OpenTPS: saved %d recalculated doses to %s", len(records), output_manifest)
    return output_manifest
