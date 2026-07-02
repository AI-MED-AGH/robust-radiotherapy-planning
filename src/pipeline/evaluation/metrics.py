import glob
import itertools
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.pipeline.config import MaisiTestingConfig
from src.pipeline.evaluation.image_metrics import (
    LPIPSModel,
    calculate_similarity_metrics,
    lpips_3d,
    mae_3d,
    psnr_3d,
    sob_3d,
    ssim_3d,
)
from src.pipeline.evaluation.structure_metrics import calculate_structure_similarity_metrics
from src.pipeline.helpers.helpers import (
    _cache_patient_cts,
    _extract_patient_id,
)


def collect_original_cts(original_ct_dir: Path) -> dict[str, list[Path]]:
    """
    Collect available original/reference CT tensors grouped by patient ID.

    This function searches recursively for `.pt` files inside
    `original_ct_dir`, extracts the patient ID from each file path, and groups
    all available original/reference CT tensors by patient.

    It works for both cases:
    - only planning CTs are available
    - multiple real test fractions are available per patient

    Parameters
    ----------
    original_ct_dir : Path
        Directory containing original/reference CT tensors saved as `.pt`
        files.

    Returns
    -------
    originals : dict[str, list[Path]]
        Dictionary mapping each patient ID to a list of CT tensor paths.

        Example:
        {
            "Patient_1": [
                Path(".../Patient_1_fraction_1_.pt"),
                Path(".../Patient_1_fraction_2_.pt"),
            ],
            "Patient_2": [
                Path(".../Patient_2_fraction_1_.pt"),
            ],
        }

    Raises
    ------
    FileNotFoundError
        If `original_ct_dir` does not exist.

    ValueError
        If no `.pt` files are found in `original_ct_dir`.
    """

    if not original_ct_dir.exists():
        raise FileNotFoundError(f"Original CT directory does not exist: {original_ct_dir}")

    paths = sorted(
        glob.glob(
            str(original_ct_dir / "**" / "*.pt"),
            recursive=True,
        )
    )

    if len(paths) == 0:
        raise ValueError(f"No original/reference CT `.pt` files found in: {original_ct_dir}")

    originals: dict[str, list[Path]] = {}

    for path in paths:
        patient_id = _extract_patient_id(path)
        originals.setdefault(patient_id, []).append(Path(path))

    return originals


def collect_generated_cts(generated_ct_dir: Path) -> dict[str, list[Path]]:
    """
    Collect generated CT variants grouped by patient ID.

    This function searches recursively for generated `.pt` CT files inside
    `generated_ct_dir`, extracts the patient ID from each path, and groups
    generated variants by patient.

    Missing patients or missing variants are allowed. The function only
    collects files that actually exist.

    Expected structure:
        generated_ct_dir/
            Patient_1/
                Patient_1_gen_1.pt
                Patient_1_gen_2.pt
            Patient_2/
                Patient_2_gen_1.pt

    Parameters
    ----------
    generated_ct_dir : Path
        Directory containing generated CT tensors saved as `.pt` files.

    Returns
    -------
    generated : dict[str, list[Path]]
        Dictionary mapping each patient ID to a list of generated CT paths.

        Example:
        {
            "Patient_1": [
                Path(".../Patient_1_gen_1.pt"),
                Path(".../Patient_1_gen_2.pt"),
            ],
            "Patient_2": [
                Path(".../Patient_2_gen_1.pt"),
            ],
        }

    Raises
    ------
    FileNotFoundError
        If `generated_ct_dir` does not exist.

    ValueError
        If no generated CT `.pt` files are found in `generated_ct_dir`.
    """

    if not generated_ct_dir.exists():
        raise FileNotFoundError(f"Generated CT directory does not exist: {generated_ct_dir}")

    paths = sorted(
        glob.glob(
            str(generated_ct_dir / "**" / "*.pt"),
            recursive=True,
        )
    )

    if len(paths) == 0:
        raise ValueError(f"No generated CT `.pt` files found in: {generated_ct_dir}")

    generated: dict[str, list[Path]] = {}

    for path in paths:
        patient_id = _extract_patient_id(path)
        generated.setdefault(patient_id, []).append(Path(path))

    return generated


def calculate_pairwise_variety_metrics(
    ct_groups: dict[str, list[Path]],
    config: MaisiTestingConfig,
    output_filename: str,
) -> None:
    """
    Calculate pairwise variety metrics within groups of CT images.

    This function compares all possible generated image pairs within each
    patient group.

    Metrics:
    - MAE: lower means more similar voxel intensities
    - SSIM: higher means more similar structure
    - PSNR: higher means lower squared voxel-wise error
    - SOB: lower means more similar edge structure
    - LPIPS: higher usually means more visual/perceptual difference

    Parameters
    ----------
    ct_groups : dict[str, list[Path]]
        Dictionary mapping patient IDs to CT tensor paths.

    config : MaisiTestingConfig
        Configuration object containing evaluation settings.

    output_filename : str
        Name of the per-comparison output CSV file.

    Raises
    ------
    ValueError
        If `ct_groups` is empty.
        If `output_filename` does not end with ".csv".
        If `max_lpips_slices` is less than 1.
        If the inputs are invalid. If no valid pairwise comparisons can be
        calculated, the metric file is skipped.
    """

    if len(ct_groups) == 0:
        raise ValueError("`ct_groups` cannot be empty")

    device = torch.device(config.device)
    lpips_model = _build_lpips_model(config)
    rows = []

    for patient_id, paths in tqdm(ct_groups.items(), desc="Generated pairwise variety metrics"):
        if len(paths) < 2:
            print(f"Skipping pairwise generated variety metrics for {patient_id}: only {len(paths)} image(s).")
            continue

        patient_tensors = _cache_patient_cts(paths, config=config, device=device)

        for path_a, path_b in itertools.combinations(paths, 2):
            arr_a = patient_tensors[path_a]
            arr_b = patient_tensors[path_b]
            if arr_a.shape != arr_b.shape:
                print(
                    f"Skipping shape mismatch for {patient_id}: "
                    f"{path_a.name} {tuple(arr_a.shape)} vs "
                    f"{path_b.name} {tuple(arr_b.shape)}"
                )
                continue

            row = {
                "patient_id": patient_id,
                "path_a": str(path_a),
                "path_b": str(path_b),
                "mae": mae_3d(arr_a, arr_b),
                "ssim": ssim_3d(arr_a, arr_b, config.data_min, config.data_max),
                "psnr": psnr_3d(arr_a, arr_b, config.data_min, config.data_max),
                "sob": sob_3d(arr_a, arr_b),
                "lpips": np.nan,
            }

            if config.use_lpips and lpips_model is not None:
                row["lpips"] = lpips_3d(
                    pred=arr_a,
                    ref=arr_b,
                    lpips_model=lpips_model,
                    data_min=config.data_min,
                    data_max=config.data_max,
                    max_slices=config.max_lpips_slices,
                )

            rows.append(row)

    if len(rows) == 0:
        print("Skipping generated pairwise variety metrics: no valid pairwise comparisons were calculated")
        return

    df = pd.DataFrame(rows)
    df.to_csv(config.metrics_dir / output_filename, index=False)

    summary = df.groupby("patient_id")[["mae", "ssim", "psnr", "sob", "lpips"]].agg(
        ["mean", "std", "min", "max", "count"]
    )
    summary_name = output_filename.replace(".csv", "_summary.csv")
    summary.to_csv(config.metrics_dir / summary_name)


def evaluate_generated_cts(
    config: MaisiTestingConfig,
) -> None:
    """
    Run full CT generation evaluation.

    This function produces up to two metric files:

    1. generated_vs_real_metrics.csv
       Generated CTs compared to original/reference CTs.

    2. generated_pairwise_variety_metrics.csv
       Generated CT variants compared with each other.

    The goal is to check:
    - whether generated CTs are similar to real CTs
    - whether generated variants are diverse relative to each other

    Parameters
    ----------
    config : MaisiTestingConfig
        Configuration object containing evaluation settings.

    use_lpips : bool
        Whether to calculate LPIPS.

    Raises
    ------
    FileNotFoundError
        If generated or original CT directories do not exist.

    ValueError
        If no valid generated-vs-real comparisons can be calculated.
    """

    generated = collect_generated_cts(config.generated_ct_dir)
    originals = collect_original_cts(config.processed_ct_dir)

    calculate_similarity_metrics(
        generated=generated,
        originals=originals,
        config=config,
        lpips_model=_build_lpips_model(config),
    )

    calculate_structure_similarity_metrics(
        generated=generated,
        originals=originals,
        config=config,
    )

    calculate_pairwise_variety_metrics(
        ct_groups=generated,
        config=config,
        output_filename="generated_pairwise_variety_metrics.csv",
    )


def _build_lpips_model(config: MaisiTestingConfig) -> LPIPSModel | None:
    if not config.use_lpips:
        return None

    try:
        import lpips  # type: ignore[import-untyped]
    except ImportError:
        print("LPIPS package not installed. Skipping LPIPS")
        config.use_lpips = False
        return None

    model = lpips.LPIPS(net=config.lpips_net).to(config.device)
    model.eval()
    return cast(LPIPSModel, model)
