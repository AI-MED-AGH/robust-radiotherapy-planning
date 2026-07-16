import argparse
from pathlib import Path
from typing import Literal

from src.pipeline.config import MaisiTestingConfig
from src.pipeline.data.prepare_test_data import prepare_test_data
from src.pipeline.evaluation.metrics import evaluate_doses, evaluate_generated_cts
from src.pipeline.inference.encode_latents import encode_latents
from src.pipeline.inference.run_generation import generate_ct_variants

PipelineStage = Literal["prepare", "encode", "generate", "evaluate", "dose-evaluate", "all"]


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the MAISI testing pipeline.

    The CLI supports running individual pipeline stages or the complete
    pipeline.

    Available stages:
    - prepare: preprocess test CTs
    - encode: encode processed planning CTs into latent space
    - generate: generate CT variants from latent conditions
    - evaluate: calculate evaluation metrics
    - all: run all stages in order

    Returns
    -------
    args : argparse.Namespace
        Parsed command-line arguments containing:
        - stage
        - device
        - cts_per_patient
        - steps
        - evaluation_split
        - validation_fold
        - no_validate_paths
        - no_lpips
        - no_dose_metrics
        - no_dose_structure_warping
        - generated_ct_dir
        - predicted_dose_dir
        - reference_dose_dir
        - warped_structure_cache_dir
        - processed_ct_dir
        - metrics_dir
    """

    parser = argparse.ArgumentParser(description="Run MAISI-based CT generation testing pipeline.")

    parser.add_argument(
        "stage",
        choices=["prepare", "encode", "generate", "evaluate", "dose-evaluate", "all"],
        help=("Pipeline stage to run: 'prepare', 'encode', 'generate', 'evaluate', 'dose-evaluate', or 'all'"),
    )

    parser.add_argument(
        "--cts-per-patient",
        type=int,
        default=1,
        help="Number of generated CT variants per patient",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of rectified flow sampling steps",
    )

    parser.add_argument(
        "--evaluation-split",
        choices=["test", "val"],
        default="test",
        help="Dataset split to run the pipeline on",
    )

    parser.add_argument(
        "--validation-fold",
        type=int,
        default=0,
        help="Cross-validation fold to use when --evaluation-split=val",
    )

    parser.add_argument(
        "--no-validate-paths",
        action="store_true",
        help="Disable path validation during config initialization",
    )

    parser.add_argument(
        "--no-lpips",
        action="store_true",
        help="Disable LPIPS metric calculation during evaluation",
    )

    parser.add_argument(
        "--no-structure-metrics",
        action="store_true",
        help="Disable structure metric calculation during evaluation",
    )

    parser.add_argument(
        "--no-dose-metrics",
        action="store_true",
        help="Disable predicted-vs-reference dose metric calculation during evaluation",
    )

    parser.add_argument(
        "--no-dose-structure-warping",
        action="store_true",
        help="Disable DVF-based planning structure warping for generated-dose structure metrics",
    )

    parser.add_argument(
        "--scenario-robustness",
        action="store_true",
        help="Enable evaluation of candidate doses across generated-anatomy scenarios",
    )

    parser.add_argument(
        "--original-anatomy-comparison",
        action="store_true",
        help="Enable robust-vs-clinical dose comparison on fraction-1 structures",
    )

    parser.add_argument(
        "--no-base-dose-smoke-test",
        action="store_true",
        help="Disable the fraction-1 base-dose smoke test",
    )

    parser.add_argument(
        "--structure-labels",
        type=int,
        nargs="+",
        default=None,
        help="Structure labels to evaluate. Defaults to 1 2 3 4 5",
    )

    parser.add_argument(
        "--generated-ct-dir",
        type=Path,
        default=None,
        help="Optional override for generated CT directory",
    )

    parser.add_argument(
        "--predicted-dose-dir",
        type=Path,
        default=None,
        help="Optional override for predicted dose directory",
    )

    parser.add_argument(
        "--reference-dose-dir",
        type=Path,
        default=None,
        help="Optional override for reference/ground-truth dose directory",
    )

    parser.add_argument(
        "--warped-structure-cache-dir",
        type=Path,
        default=None,
        help="Optional override for persisted warped structure mask cache directory",
    )

    parser.add_argument(
        "--dose-model-name",
        type=str,
        default=None,
        help="Model or baseline name written to dose metric outputs",
    )

    parser.add_argument(
        "--dose-dvh-bin-width",
        type=float,
        default=None,
        help="Dose bin width in Gy for cumulative DVH output",
    )

    parser.add_argument(
        "--dose-dx-volume-percents",
        type=float,
        nargs="+",
        default=None,
        help="Volume percentages for Dx metrics, for example 2 50 95 98",
    )

    parser.add_argument(
        "--dose-vx-thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Dose thresholds in Gy for Vx metrics, for example 5 10 20 30",
    )

    parser.add_argument(
        "--processed-ct-dir",
        type=Path,
        default=None,
        help="Optional override for processed/original CT directory",
    )

    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=None,
        help="Optional override for metrics output directory",
    )

    parser.add_argument(
        "--structures-root",
        type=Path,
        default=None,
        help="Optional override for original structure label-map directory",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> MaisiTestingConfig:
    """
    Build a MAISI testing configuration from parsed CLI arguments.

    This function converts command-line arguments into a `MaisiTestingConfig`
    object. It sets the main runtime options directly during initialization
    and then applies optional path overrides.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments returned by `parse_args()`.

    Returns
    -------
    config : MaisiTestingConfig
        Configuration object used by the MAISI testing pipeline.

    Raises
    ------
    AttributeError
        If required CLI arguments are missing from `args`.
    """

    config_kwargs = {
        "validate_paths": not args.no_validate_paths,
        "cts_per_patient": args.cts_per_patient,
        "steps": args.steps,
        "evaluation_split": args.evaluation_split,
        "validation_fold": args.validation_fold,
        "use_lpips": not args.no_lpips,
        "use_structure_metrics": not args.no_structure_metrics,
        "use_dose_metrics": not args.no_dose_metrics,
        "use_dose_structure_warping": not args.no_dose_structure_warping,
        "use_base_dose_smoke_test": not args.no_base_dose_smoke_test,
        "use_scenario_robustness": args.scenario_robustness,
        "use_original_anatomy_comparison": args.original_anatomy_comparison,
    }

    if args.structure_labels is not None:
        config_kwargs["structure_labels"] = args.structure_labels

    if args.generated_ct_dir is not None:
        config_kwargs["generated_ct_dir"] = args.generated_ct_dir

    if args.predicted_dose_dir is not None:
        config_kwargs["predicted_dose_dir"] = args.predicted_dose_dir

    if args.reference_dose_dir is not None:
        config_kwargs["reference_dose_dir"] = args.reference_dose_dir

    if args.warped_structure_cache_dir is not None:
        config_kwargs["warped_structure_cache_dir"] = args.warped_structure_cache_dir

    if args.dose_model_name is not None:
        config_kwargs["dose_model_name"] = args.dose_model_name

    if args.dose_dvh_bin_width is not None:
        config_kwargs["dose_dvh_bin_width"] = args.dose_dvh_bin_width

    if args.dose_dx_volume_percents is not None:
        config_kwargs["dose_dx_volume_percents"] = args.dose_dx_volume_percents

    if args.dose_vx_thresholds is not None:
        config_kwargs["dose_vx_thresholds"] = args.dose_vx_thresholds

    if args.processed_ct_dir is not None:
        config_kwargs["processed_ct_dir"] = args.processed_ct_dir

    if args.metrics_dir is not None:
        config_kwargs["metrics_dir"] = args.metrics_dir

    if args.structures_root is not None:
        config_kwargs["structures_root"] = args.structures_root

    config = MaisiTestingConfig(**config_kwargs)

    return config


def run_stage(
    stage: PipelineStage,
    config: MaisiTestingConfig,
) -> None:
    """
    Run one selected MAISI testing pipeline stage.

    Each stage entry point clears only the output directory it owns before
    saving new results. This keeps single-stage runs usable: inputs produced by
    earlier stages are preserved, while stale outputs for the selected stage are
    removed by the function that writes them.

    Supported stages:
    - prepare: preprocess test CTs
    - encode: encode processed planning CTs into latent space
    - generate: generate CT variants from latent conditions
    - evaluate: calculate generated-vs-real and variety metrics
    - all: run prepare, encode, generate, and evaluate in sequence

    Parameters
    ----------
    stage : PipelineStage
        Pipeline stage to run.

    config : MaisiTestingConfig
        Pipeline configuration object.

    Raises
    ------
    ValueError
        If `stage` is not supported.
    """

    if stage == "prepare":
        prepare_test_data(config)

    elif stage == "encode":
        encode_latents(config)

    elif stage == "generate":
        generate_ct_variants(config)

    elif stage == "evaluate":
        evaluate_generated_cts(config)

    elif stage == "dose-evaluate":
        evaluate_doses(config)

    elif stage == "all":
        prepare_test_data(config)
        encode_latents(config)
        generate_ct_variants(config)
        evaluate_generated_cts(config)

    else:
        raise ValueError(f"Unsupported stage: {stage}")


def main() -> None:
    """
    Run the MAISI testing pipeline CLI.

    This function:
    - parses command-line arguments
    - builds the pipeline configuration
    - runs the selected pipeline stage

    It is used as the entry point when calling:

        python -m src.pipeline.cli <stage>
    """

    args = parse_args()
    config = build_config(args)

    run_stage(
        stage=args.stage,
        config=config,
    )


if __name__ == "__main__":
    main()
