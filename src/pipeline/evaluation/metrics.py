import glob
from pathlib import Path

from src.pipeline.config import MaisiTestingConfig
from src.pipeline.evaluation.image_metrics import (
    calculate_pairwise_variety_metrics,
    calculate_similarity_metrics,
)
from src.pipeline.evaluation.structure_metrics import calculate_structure_similarity_metrics
from src.pipeline.helpers.helpers import (
    _build_lpips_model,
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
    )
