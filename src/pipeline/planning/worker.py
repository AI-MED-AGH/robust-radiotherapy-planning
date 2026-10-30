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
    FileNotFoundError
        If a referenced settings, CT, mask, or other input file is missing.

    ValueError
        If an input image, target, plan, or calculated dose is invalid.

    RuntimeError
        If OpenTPS is unavailable or plan/dose calculation fails.
    """

    # The parent process passes file paths and planning choices through JSON
    request = json.loads(request_path.read_text(encoding="utf-8"))
    settings = OpenTPSSettings.load(Path(request["settings_path"]))
    directory = Path(request["patient_dir"])
    root = Path(request["output_root"])
    # Reuse one calculator sequentially for this patient's scenarios and final doses
    backend = OpenTPSBackend(settings, directory / "simulation")
    anatomies = {}
    for index, entry in request["anatomies"].items():
        image = load_ct(Path(entry["ct_path"]))
        # JSON keys are strings; restore integer structure labels and anatomy indices
        masks: dict[int, Any] = {int(label): nib.load(path) for label, path in entry["masks"].items()}
        anatomies[int(index)] = (image, masks)
    # Index zero is the planning CT; positive indices identify generated anatomies
    planning = anatomies[0][0]
    # Cover every loaded target position when placing the shared proton spots
    # Masks must already share a voxel grid: this combines arrays without registration
    union = np.logical_or.reduce(
        [np.asarray(masks[settings.target_label].dataobj) > 0 for _, masks in anatomies.values()]
    )
    # Give the union the planning CT's geometry, then convert it to an OpenTPS mask
    target = opentps_image(mask_image(union, planning), backend.ROIMask, name="target_union")
    # All scenario sets start from this same spot geometry, designed on the planning CT
    base = backend.build_plan(opentps_image(planning, backend.CTImage, name=request["patient_id"]), target)
    # Calculate each anatomy's beamlets and structure objectives once for reuse across sets
    fidelities = {
        index: backend.scenario(image, masks, base, f"{request['patient_id']}_scenario_{index}")
        for index, (image, masks) in anatomies.items()
    }
    # Keep plan/generated-dose metadata separate from dose records for real fraction CTs
    plans: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for name, indices in request["scenario_sets"].items():
        # Each named selection of generated anatomies produces its own optimized plan
        output_dir = directory / name
        output_dir.mkdir(parents=True)
        # Always include the planning anatomy first, followed by this set's generated anatomies
        # Optimize one shared MU vector across them, keeping the base spot geometry fixed
        optimized, optimization = backend.optimize(base, [fidelities[index] for index in [0, *indices]])
        backend.save_plan(optimized, output_dir / "plan.dcm")
        generated_doses = []
        for index in indices:
            output = output_dir / f"{request['patient_id']}_gen_{index}.nii.gz"
            # Recalculate the fixed optimized plan on each selected generated CT
            backend.calculate_dose(anatomies[index][0], optimized, output)
            # Store paths relative to the output root so the manifest can move with its files
            generated_doses.append({"variant_index": index, "dose_path": str(output.relative_to(root))})
        # Record this plan, the solver report, and its generated-anatomy dose files
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
        # Evaluate the same plan on real anatomies without optimizing it again
        for entry in request["real_cts"]:
            ct_path = Path(entry["ct_path"])
            image = load_ct(ct_path)
            # Remove both parts of .nii.gz to obtain the real fraction's identifier
            stem = ct_path.name.removesuffix(".gz").removesuffix(".nii")
            output = output_dir / f"{stem}.nii.gz"
            logger.info("Recalculating %s on %s", name, stem)
            backend.calculate_dose(image, optimized, output)
            # Preserve the input CT metadata and link this dose to its scenario set
            records.append(
                {
                    **entry,
                    "patient_id": request["patient_id"],
                    "scenario_set": name,
                    "scenario_id": stem,
                    "dose_path": str(output.relative_to(root)),
                }
            )
    # Return a manifest to the parent process; dose arrays remain in their NIfTI files
    result = request_path.with_name("result.json")
    result.write_text(json.dumps({"plans": plans, "doses": records}, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    # The parent launches this module with the request JSON path as its first argument
    run_request(Path(sys.argv[1]))
