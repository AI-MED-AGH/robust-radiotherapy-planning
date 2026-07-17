import glob
import logging
from pathlib import Path

from src.pipeline.config import MaisiTestingConfig
from src.pipeline.evaluation.dose_metrics import (
    evaluate_base_dose_smoke_test,
    evaluate_original_anatomy_comparison,
    evaluate_predicted_doses,
    evaluate_scenario_robustness,
)
from src.pipeline.evaluation.image_metrics import (
    calculate_pairwise_variety_metrics,
    calculate_similarity_metrics,
)
from src.pipeline.evaluation.structure_metrics import calculate_structure_similarity_metrics
from src.pipeline.helpers.cleanup import clear_directory_contents
from src.pipeline.helpers.helpers import _clean_pipeline_memory, _extract_patient_id
from src.pipeline.helpers.helpers_doses import _contains_dose_files
from src.pipeline.helpers.helpers_metrics import _build_lpips_model
from src.pipeline.helpers.helpers_structures import WarpedStructureMaskCache

logger = logging.getLogger(__name__)


def evaluate_doses(
    config: MaisiTestingConfig,
    warped_mask_cache: WarpedStructureMaskCache | None = None,
) -> None:
    """Validate dose inputs and dispatch the configured evaluation workflows.

    The function always checks whether dose evaluation is enabled and whether
    clinical dose files are available. It can then run the base-dose smoke
    test, predicted-vs-reference metrics, scenario robustness evaluation, and
    original-anatomy comparison according to ``config``. Missing predicted
    doses skip only workflows that require predictions.

    Parameters
    ----------
    config : MaisiTestingConfig
        Pipeline configuration containing workflow flags, dose directories,
        metric settings, structure labels, and output paths.

    warped_mask_cache : WarpedStructureMaskCache | None, optional
        Generated-space structure masks produced during structure evaluation.
        Reusing this cache avoids repeating planning-to-generated registration.
    """

    if not config.use_dose_metrics:
        logger.info("Skipping dose metrics: disabled by configuration")
        return

    # Every dose workflow needs the clinical dose set, including the smoke
    # test that intentionally runs without model predictions
    if not _contains_dose_files(config.reference_dose_dir):
        logger.warning("Skipping dose metrics: no clinical dose files found in %s", config.reference_dose_dir)
        return

    if config.use_base_dose_smoke_test:
        logger.info("Calculating fraction-1 base-dose smoke-test metrics")
        evaluate_base_dose_smoke_test(config, warped_mask_cache=warped_mask_cache)
    else:
        logger.info("Skipping base-dose smoke test: disabled by configuration")

    has_predicted_doses = _contains_dose_files(config.predicted_dose_dir)
    if not has_predicted_doses:
        logger.info(
            "No predicted doses found in %s; scenario robustness will use the clinical fallback if enabled",
            config.predicted_dose_dir,
        )
    else:
        logger.info("Calculating predicted-vs-reference dose metrics")
        evaluate_predicted_doses(config, warped_mask_cache=warped_mask_cache)

    if config.use_scenario_robustness:
        logger.info("Calculating scenario robustness metrics")
        evaluate_scenario_robustness(config, warped_mask_cache=warped_mask_cache)
    else:
        logger.info("Skipping scenario robustness metrics: disabled by configuration")

    if config.use_original_anatomy_comparison and has_predicted_doses:
        logger.info("Calculating original-anatomy dose comparison")
        evaluate_original_anatomy_comparison(config)
    elif config.use_original_anatomy_comparison:
        logger.info("Skipping original-anatomy dose comparison: no predicted doses found")
    else:
        logger.info("Skipping original-anatomy dose comparison: disabled by configuration")


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

    # setdefault keeps discovery tolerant of sparse and uneven patient sets
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

    # Preserve every generated variant; later metric stages decide which
    # patient/scenario pairs have a corresponding reference
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

    # Collect once and pass the same grouping to all image/structure stages so
    # they evaluate an identical snapshot of the available files
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
    # Release metric tensors before registration and structure processing,
    # which can also have substantial CPU and GPU memory footprints
    _clean_pipeline_memory()

    # Structure evaluation returns generated-space masks keyed by CT path.
    # Dose evaluation consumes the same cache later when enabled.
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

    # Dose orchestration remains last because it can reuse warped masks while
    # independently deciding which optional dose workflows are configured
    evaluate_doses(config, warped_mask_cache=warped_mask_cache)
    _clean_pipeline_memory()

    logger.info("Finished evaluation")
