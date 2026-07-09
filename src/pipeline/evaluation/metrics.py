import glob
import logging
from pathlib import Path

from src.pipeline.config import MaisiTestingConfig
from src.pipeline.evaluation.dose_metrics import evaluate_predicted_doses
from src.pipeline.evaluation.image_metrics import (
    calculate_pairwise_variety_metrics,
    calculate_similarity_metrics,
)
from src.pipeline.evaluation.structure_metrics import calculate_structure_similarity_metrics
from src.pipeline.helpers.cleanup import clear_directory_contents
from src.pipeline.helpers.helpers import (
    _build_lpips_model,
    _clean_pipeline_memory,
    _extract_patient_id,
)

logger = logging.getLogger(__name__)


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

    The metrics output directory is cleared before new CSV files are written.
    Generated CTs and processed reference CTs are left untouched.

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

    OSError
        If the metrics output directory cannot be cleared.
    """

    clear_directory_contents(config.metrics_dir)

    generated = collect_generated_cts(config.generated_ct_dir)
    originals = collect_original_cts(config.processed_ct_dir)
    lpips_model = _build_lpips_model(config)

    logger.info(
        "Starting evaluation: "
        f"{sum(len(paths) for paths in generated.values())} generated CT(s), "
        f"{sum(len(paths) for paths in originals.values())} original CT(s), "
        f"metrics_dir={config.metrics_dir}"
    )

    logger.info("Calculating generated-vs-real image metrics")
    calculate_similarity_metrics(
        generated=generated,
        originals=originals,
        config=config,
        lpips_model=lpips_model,
    )
    _clean_pipeline_memory()

    warped_mask_cache = {}
    if config.use_structure_metrics:
        logger.info("Calculating generated-vs-real structure metrics")
        warped_mask_cache = calculate_structure_similarity_metrics(
            generated=generated,
            originals=originals,
            config=config,
        )
        _clean_pipeline_memory()
    else:
        logger.info("Skipping generated-vs-real structure metrics: disabled by configuration")

    logger.info("Calculating generated pairwise variety metrics")
    calculate_pairwise_variety_metrics(
        ct_groups=generated,
        config=config,
        lpips_model=lpips_model,
    )
    _clean_pipeline_memory()

    if config.use_dose_metrics:
        logger.info("Calculating predicted-vs-reference dose metrics")
        evaluate_predicted_doses(config, warped_mask_cache=warped_mask_cache)
        _clean_pipeline_memory()
    else:
        logger.info("Skipping predicted-vs-reference dose metrics: disabled by configuration")

    logger.info("Finished evaluation")
