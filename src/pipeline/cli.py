import argparse
from pathlib import Path
from typing import Literal

from src.pipeline.config import MaisiTestingConfig
from src.pipeline.data.prepare_test_data import prepare_test_data
from src.pipeline.evaluation.metrics import evaluate_generated_cts
from src.pipeline.inference.encode_latents import encode_latents
from src.pipeline.inference.run_generation import generate_ct_variants


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the MAISI testing pipeline.

    The CLI supports running individual pipeline stages or the complete
    pipeline.

    Available stages:
    - prepare: preprocess test planning CTs
    - encode: encode processed CTs into latent space
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
        - no_validate_paths
        - no_lpips
        - generated_ct_dir
        - processed_ct_dir
        - metrics_dir
    """

    parser = argparse.ArgumentParser(description="Run MAISI-based CT generation testing pipeline.")

    parser.add_argument(
        "stage",
        choices=["prepare", "encode", "generate", "evaluate", "all"],
        help=("Pipeline stage to run: 'prepare', 'encode', 'generate', 'evaluate', or 'all'"),
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
        "--generated-ct-dir",
        type=Path,
        default=None,
        help="Optional override for generated CT directory",
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
        "use_lpips": not args.no_lpips,
    }

    if args.generated_ct_dir is not None:
        config_kwargs["generated_ct_dir"] = args.generated_ct_dir

    if args.processed_ct_dir is not None:
        config_kwargs["processed_ct_dir"] = args.processed_ct_dir

    if args.metrics_dir is not None:
        config_kwargs["metrics_dir"] = args.metrics_dir

    config = MaisiTestingConfig(**config_kwargs)

    return config


def run_stage(
    stage: Literal["prepare", "encode", "generate", "evaluate", "all"],
    config: MaisiTestingConfig,
) -> None:
    """
    Run one selected MAISI testing pipeline stage.

    Supported stages:
    - prepare: preprocess test planning CTs
    - encode: encode processed CTs into latent space
    - generate: generate CT variants from latent conditions
    - evaluate: calculate generated-vs-real and variety metrics
    - all: run prepare, encode, generate, and evaluate in sequence

    Parameters
    ----------
    stage : Literal["prepare", "encode", "generate", "evaluate", "all"]
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
