import json
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from src.pipeline.config import MaisiTestingConfig
from src.pipeline.evaluation.dose_metrics import _clinical_metric_values, dose_volume_histogram
from src.pipeline.planning.geometry import canonical_image, labels_on_grid, load_ct, make_nifti, resample_to_grid


def evaluate_robust_doses(config: MaisiTestingConfig) -> None:
    """
    Evaluate recalculated OpenTPS plans on real CTs and compare scenario sets.

    Calculate structure-wise mean/max dose, Dx, Vx and DVHs on full real CT
    grids, then summarize fractions and compare plans on matching scans.
    Pairwise differences always mean candidate minus reference.

    Parameters
    ----------
    config : MaisiTestingConfig
        Pipeline configuration supplying the robust-dose manifest directory,
        output metrics directory, structure labels, Dx volume percentages,
        Vx dose thresholds and DVH dose-bin width.

    Raises
    ------
    ValueError
        If the manifest has an unsupported schema or empty plan/dose records,
        plan coverage differs, dose records are duplicated, scenario sets use
        inconsistent scan sources or unmatched comparison rows, a dose path
        leaves the output root, or a dose is invalid or does not cover its CT.
        Invalid metric settings also propagate from the metric helpers.

    OSError
        If the manifest, input images or metric outputs cannot be accessed.

    json.JSONDecodeError
        If the manifest is not valid JSON.

    KeyError
        If a required manifest field is missing.
    """

    manifest_path = config.robust_dose_dir / "manifest.json"
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1 or not manifest.get("doses") or not manifest.get("plans"):
        raise ValueError(f"Invalid or empty OpenTPS dose manifest: {manifest_path}")
    rows: list[dict[str, Any]] = []
    dvhs: list[pd.DataFrame] = []
    seen: set[tuple[str, str, str]] = set()
    actual_sources: dict[tuple[str, str], tuple[str, str]] = {}
    expected_plans = {(plan["patient_id"], plan["scenario_set"]) for plan in manifest["plans"]}
    received_plans = {(record["patient_id"], record["scenario_set"]) for record in manifest["doses"]}
    if expected_plans != received_plans:
        raise ValueError("Recalculated doses do not cover every plan in the OpenTPS manifest")
    for record in manifest["doses"]:
        identity = (record["patient_id"], record["scenario_set"], record["scenario_id"])
        if identity in seen:
            raise ValueError(f"Duplicate recalculated dose in manifest: {identity}")
        seen.add(identity)
        scan_key = (record["patient_id"], record["scenario_id"])
        source_pair = (record["ct_path"], record["structure_path"])
        if scan_key in actual_sources and actual_sources[scan_key] != source_pair:
            raise ValueError(f"Scenario sets do not refer to identical real scans for {scan_key}")
        actual_sources[scan_key] = source_pair
        dose_path = (config.robust_dose_dir / record["dose_path"]).resolve()
        if not dose_path.is_relative_to(config.robust_dose_dir.resolve()):
            raise ValueError(f"Dose manifest path leaves the configured dose directory: {dose_path}")
        ct = load_ct(Path(record["ct_path"]))
        image = canonical_image(nib.load(str(dose_path)))
        native = image.get_fdata(dtype=np.float32)
        if native.ndim != 3 or not np.isfinite(native).all() or (native < 0).any() or not (native > 0).any():
            raise ValueError(f"Invalid recalculated dose: {dose_path}")
        # A dose must cover the full scan. Zero-padding a misaligned grid would
        # otherwise produce apparently valid but misleading target metrics.
        same_grid = image.shape == ct.shape and np.allclose(image.affine, ct.affine, atol=1e-3)
        if not same_grid:
            coverage_image = make_nifti(np.ones(image.shape, dtype=np.uint8), image.affine)
            coverage = np.asarray(resample_to_grid(coverage_image, ct, order=0).dataobj)
            if not coverage.all():
                raise ValueError(f"Recalculated dose does not cover the real CT grid: {dose_path}")
        aligned = image if same_grid else resample_to_grid(image, ct, order=1)
        dose = torch.from_numpy(np.asarray(aligned.dataobj, dtype=np.float32).copy())
        labels = labels_on_grid(Path(record["structure_path"]), ct)
        metadata = {
            **record,
            "model_name": manifest["model_name"],
            "run_id": manifest["run_id"],
            "dose_source": f"opentps:{record['scenario_set']}",
            "dose_path": str(dose_path),
        }
        for label in config.structure_labels:
            mask = torch.from_numpy(labels != 0 if label == 0 else labels == label)
            for metric, value in _clinical_metric_values(dose, mask, config).items():
                rows.append({**metadata, "structure_label": label, "metric": metric, "value": value})
            dvh = dose_volume_histogram(dose, mask, config.dose_dvh_bin_width)
            if not dvh.empty:
                dvhs.append(dvh.assign(**metadata, structure_label=label))
    frame = pd.DataFrame(rows)
    keys = ["model_name", "patient_id", "scenario_set", "scenario_type", "structure_label", "metric"]
    summary = frame.groupby(keys, dropna=False)["value"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    comparisons = []
    pairing = ["model_name", "patient_id", "scenario_id", "scenario_type", "structure_label", "metric"]
    sets = sorted(frame["scenario_set"].unique(), key=lambda name: (name != "nominal", name))
    for reference, candidate in combinations(sets, 2):
        left = frame.loc[frame["scenario_set"] == reference, [*pairing, "value"]]
        right = frame.loc[frame["scenario_set"] == candidate, [*pairing, "value"]]
        compared = right.merge(
            left, on=pairing, how="outer", suffixes=("_candidate", "_reference"), validate="one_to_one", indicator=True
        )
        if not (compared["_merge"] == "both").all():
            raise ValueError(f"Scenario sets {candidate} and {reference} do not cover identical real scans")
        compared = compared.drop(columns="_merge")
        compared["candidate_set"] = candidate
        compared["reference_set"] = reference
        compared["delta"] = compared["value_candidate"] - compared["value_reference"]
        comparisons.append(compared)
    # Calculate everything before publishing any metric file
    config.metrics_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".robust-metrics-", dir=config.metrics_dir) as temporary:
        directory = Path(temporary)
        frame.to_csv(directory / "opentps_dose_metrics.csv", index=False)
        summary.to_csv(directory / "opentps_dose_summary.csv", index=False)
        dvh_frame = pd.concat(dvhs, ignore_index=True) if dvhs else pd.DataFrame(columns=["dose_gy", "volume_percent"])
        dvh_frame.to_csv(directory / "opentps_dose_dvh.csv", index=False)
        comparison = (
            pd.concat(comparisons, ignore_index=True)
            if comparisons
            else pd.DataFrame(
                columns=[*pairing, "value_candidate", "value_reference", "candidate_set", "reference_set", "delta"]
            )
        )
        comparison.to_csv(directory / "opentps_dose_comparison.csv", index=False)
        for path in directory.iterdir():
            path.replace(config.metrics_dir / path.name)
