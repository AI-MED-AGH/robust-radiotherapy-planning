import json
import logging
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from src.pipeline.planning.geometry import load_ct, mask_image, opentps_image
from src.pipeline.planning.opentps_backend import OpenTPSBackend
from src.pipeline.planning.settings import OpenTPSSettings

logger = logging.getLogger(__name__)


def _load_anatomies(entries: dict[str, Any]) -> dict[int, tuple[Any, dict[int, Any]]]:
    """
    Load the scenario CTs and structure masks listed in a worker request.

    CTs are validated and reoriented to canonical RAS+ by ``load_ct``. The
    masks are loaded as stored and must already match the canonical CT grid.

    Parameters
    ----------
    entries : dict[str, Any]
        The request's ``anatomies`` mapping. Each scenario index maps to an
        entry with a ``ct_path`` and a ``masks`` mapping from structure labels
        to NIfTI paths. JSON represents both indices and labels as strings.

    Returns
    -------
    dict[int, tuple[Any, dict[int, Any]]]
        Pairs of CT images and label-to-mask mappings keyed by integer
        scenario index. Index zero denotes the planning CT.

    Raises
    ------
    KeyError
        If an anatomy entry is missing ``ct_path`` or ``masks``.
    ValueError
        If an index or label cannot be parsed as an integer or a CT fails
        the value or grid checks in ``load_ct``.
    OSError, nibabel.filebasedimages.ImageFileError
        If a CT or mask cannot be opened as an image.
    """
    anatomies = {}
    for index, entry in entries.items():
        image = load_ct(Path(entry["ct_path"]))
        masks = {int(label): nib.load(path) for label, path in entry["masks"].items()}
        anatomies[int(index)] = (image, masks)
    return anatomies


def run_request(request_path: Path) -> Path:
    """
    Run one patient's OpenTPS planning and dose-recalculation request.

    This worker deliberately avoids importing PyTorch and ``pipeline.config``.
    On Windows, loading PyTorch and OpenTPS's MKL/OpenMP dependencies in one
    process can abort the interpreter, so the parent process passes data using
    NIfTI files and JSON request/result files.

    The request JSON identifies the patient output directory, OpenTPS settings,
    planning and generated anatomies with their masks, selected scenario sets,
    and real fraction CTs. For each scenario set, this function optimizes a plan
    on the planning and selected generated anatomies, then calculates doses on
    generated and real CTs. It writes a sibling ``result.json`` containing plan
    and dose records.

    Parameters
    ----------
    request_path : Path
        Path to the JSON request written by the pipeline process.

    Returns
    -------
    Path
        Path to the generated ``result.json`` file.

    Raises
    ------
    OSError
        If the request, settings, CTs, masks or machine data cannot be read,
        or simulation, plan, dose or result files cannot be written.

    json.JSONDecodeError
        If the request or settings file is not valid JSON.

    UnicodeDecodeError
        If the request or settings file cannot be decoded as UTF-8.

    KeyError
        If a required request/settings field, anatomy index or structure
        label is missing.

    TypeError, AttributeError
        If request or settings containers or field values have incompatible types.

    nibabel.filebasedimages.ImageFileError
        If a CT or mask file is empty or its image format cannot be recognized.

    nibabel.spatialimages.HeaderDataError
        If an input or output image header is invalid.

    nibabel.orientations.OrientationError
        If image axes cannot be reoriented.

    numpy.linalg.LinAlgError
        If an image affine decomposition fails.

    ValueError
        If settings, an input image, target, plan or calculated dose is invalid,
        or scenario/label indices cannot be converted to integers.

    RuntimeError
        If plan optimization fails or the exported plan file is absent.

    Notes
    -----
    OpenTPS and its native dependencies are imported before this function is
    called; import failures occur at module import time. Errors from OpenTPS
    calibration readers and MCsquare are propagated without wrapping.
    """

    request = json.loads(request_path.read_text(encoding="utf-8"))
    settings = OpenTPSSettings.load(Path(request["settings_path"]))
    directory = Path(request["patient_dir"])
    root = Path(request["output_root"])

    backend = OpenTPSBackend(settings, directory / "simulation")
    anatomies = _load_anatomies(request["anatomies"])
    planning = anatomies[0][0]

    # Place shared spots to cover the target across anatomies on the common voxel grid
    union = np.logical_or.reduce(
        [np.asarray(masks[settings.target_label].dataobj) > 0 for _, masks in anatomies.values()]
    )
    target = opentps_image(mask_image(union, planning), backend.ROIMask, name="target_union")
    base = backend.build_plan(opentps_image(planning, backend.CTImage, name=request["patient_id"]), target)

    # Reuse beamlets and structure objectives when an anatomy appears in multiple sets
    fidelities = {
        index: backend.scenario(image, masks, base, f"{request['patient_id']}_scenario_{index}")
        for index, (image, masks) in anatomies.items()
    }
    plans: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for name, indices in request["scenario_sets"].items():
        output_dir = directory / name
        output_dir.mkdir(parents=True)

        # Include the planning anatomy first so it supplies the initial target dose
        optimized, optimization = backend.optimize(base, [fidelities[index] for index in [0, *indices]])
        backend.save_plan(optimized, output_dir / "plan.dcm")

        generated_doses = []
        for index in indices:
            output = output_dir / f"{request['patient_id']}_gen_{index}.nii.gz"
            backend.calculate_dose(anatomies[index][0], optimized, output)
            generated_doses.append({"variant_index": index, "dose_path": str(output.relative_to(root))})

        plans.append(
            {
                "patient_id": request["patient_id"],
                "scenario_set": name,
                "variant_indices": indices,
                "plan_path": str((output_dir / "plan.dcm").relative_to(root)),
                "optimization": optimization,
                "generated_doses": generated_doses,
            }
        )

        for entry in request["real_cts"]:
            ct_path = Path(entry["ct_path"])
            image = load_ct(ct_path)
            # Remove both parts of .nii.gz to obtain the real fraction's identifier
            stem = ct_path.name.removesuffix(".gz").removesuffix(".nii")
            output = output_dir / f"{stem}.nii.gz"
            logger.info("Recalculating %s on %s", name, stem)
            backend.calculate_dose(image, optimized, output)
            records.append(
                {
                    **entry,
                    "patient_id": request["patient_id"],
                    "scenario_set": name,
                    "scenario_id": stem,
                    "dose_path": str(output.relative_to(root)),
                }
            )

    result = request_path.with_name("result.json")
    result.write_text(json.dumps({"plans": plans, "doses": records}, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    # The parent launches this module with the request JSON path as its first argument
    run_request(Path(sys.argv[1]))
