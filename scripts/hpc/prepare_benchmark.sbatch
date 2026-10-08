#!/bin/bash
#SBATCH --job-name=prepare-benchmark
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=prepare-benchmark-%j.out

# Submit from the repository root:
#   sbatch --account=YOUR_ACCOUNT scripts/hpc/prepare_benchmark.sbatch
# Optional overrides (export before submitting): BENCHMARK_PATIENT,
# BENCHMARK_FRACTION, BENCHMARK_LABEL, BENCHMARK_STEPS, BENCHMARK_DATA_ROOT,
# BENCHMARK_WEIGHTS_DIR, BENCHMARK_PIPELINE_ROOT, BENCHMARK_PAIR_DIR.
# Pipeline outputs use a separate directory for each job by default.
# Reusing a pipeline root replaces its latent, generated, mask and metric outputs.
# The exported pair directory is overwritten when reused.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"

export BENCHMARK_PATIENT="${BENCHMARK_PATIENT:-Patient_01}"
export BENCHMARK_FRACTION="${BENCHMARK_FRACTION:-2}"
export BENCHMARK_LABEL="${BENCHMARK_LABEL:-1}"
export BENCHMARK_STEPS="${BENCHMARK_STEPS:-30}"
export BENCHMARK_DATA_ROOT="${BENCHMARK_DATA_ROOT:-$SLURM_SUBMIT_DIR/src/data_full}"
export BENCHMARK_WEIGHTS_DIR="${BENCHMARK_WEIGHTS_DIR:-$SLURM_SUBMIT_DIR/src/pipeline/weights}"
export BENCHMARK_PIPELINE_ROOT="${BENCHMARK_PIPELINE_ROOT:-$SLURM_SUBMIT_DIR/RESULTS/BENCHMARK_PIPELINE/$SLURM_JOB_ID}"
export BENCHMARK_PAIR_DIR="${BENCHMARK_PAIR_DIR:-$SLURM_SUBMIT_DIR/RESULTS/benchmarks/${BENCHMARK_PATIENT}_pair}"

uv --version
if ! uv sync --locked --extra gpu --no-dev --offline; then
    echo "Environment setup failed. On a JUPITER login node, run: uv sync --locked --extra gpu --no-dev" >&2
    exit 1
fi
python_bin="$(uv run --offline --no-sync python -c 'import sys; print(sys.executable)')"

srun "$python_bin" - <<'PY'
import os
from pathlib import Path

import torch

from src.pipeline.config import MaisiTestingConfig
from src.pipeline.data.prepare_test_data import process_and_save_cts
from src.pipeline.evaluation.metrics import evaluate_generated_cts
from src.pipeline.inference.encode_latents import encode_latents
from src.pipeline.inference.run_generation import generate_ct_variants

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable; pair preparation requires a GPU allocation")

patient = os.environ["BENCHMARK_PATIENT"]
fraction = int(os.environ["BENCHMARK_FRACTION"])
label = int(os.environ["BENCHMARK_LABEL"])
if fraction < 2 or label < 1:
    raise ValueError("Reference fraction must be at least 2 and structure label must be positive")

torch.manual_seed(42)
config = MaisiTestingConfig(
    stage="all",
    device="cuda",
    data_root=Path(os.environ["BENCHMARK_DATA_ROOT"]),
    weights_dir=Path(os.environ["BENCHMARK_WEIGHTS_DIR"]),
    output_root=Path(os.environ["BENCHMARK_PIPELINE_ROOT"]),
    evaluation_split="test",
    cts_per_patient=1,
    steps=int(os.environ["BENCHMARK_STEPS"]),
    use_lpips=False,
    use_dose_metrics=False,
    structure_labels=[label],
)

ct_paths = [
    config.ct_root / patient / f"{patient}_fraction_{index}_.nii.gz"
    for index in (1, fraction)
]
structure_paths = [
    config.structures_root / patient / f"{patient}_fraction_{index}_.nii.gz"
    for index in (1, fraction)
]
for path in (*ct_paths, *structure_paths):
    if not path.is_file():
        raise FileNotFoundError(f"Required patient input is missing: {path}")

print(f"Preparing {patient}, reference fraction {fraction}, structure label {label}", flush=True)
print(f"Pipeline output: {config.output_root}", flush=True)
# Select the two fractions explicitly instead of relying on patient split membership.
process_and_save_cts(config, [str(path) for path in ct_paths])
encode_latents(config)
generate_ct_variants(config)
# Structure evaluation persists the warped masks consumed by the pair exporter.
evaluate_generated_cts(config)
PY

srun "$python_bin" -m src.benchmarks.prepare_patient_pair \
    --patient "$BENCHMARK_PATIENT" \
    --generated-index 1 \
    --fraction "$BENCHMARK_FRACTION" \
    --label "$BENCHMARK_LABEL" \
    --split test \
    --data-root "$BENCHMARK_DATA_ROOT" \
    --pipeline-root "$BENCHMARK_PIPELINE_ROOT" \
    --output "$BENCHMARK_PAIR_DIR"

echo "Benchmark pair ready: $BENCHMARK_PAIR_DIR"
