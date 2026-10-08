#!/bin/bash
#SBATCH --job-name=metric-profile
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=metric-profile-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

metric="${1:-ssim}"
if [[ "$metric" != "ssim" && "$metric" != "hausdorff" ]]; then
    echo "Metric must be ssim or hausdorff" >&2
    exit 2
fi
shift || true
uv --version
if ! uv sync --locked --extra gpu --no-dev --offline; then
    echo "Environment setup failed. On a JUPITER login node, run: uv sync --locked --extra gpu --no-dev" >&2
    exit 1
fi
python_bin="$(uv run --offline --no-sync python -c 'import sys; print(sys.executable)')"
output_dir="${BENCHMARK_OUTPUT_DIR:-$SLURM_SUBMIT_DIR/RESULTS/benchmarks/$SLURM_JOB_ID}"
mkdir -p "$output_dir"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-4}"

"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_properties(0))'
nsys --version
nvidia-smi > "$output_dir/gpu.txt"
"$python_bin" -c 'from importlib.metadata import distributions; print("\n".join(sorted("{}=={}".format(d.metadata["Name"], d.version) for d in distributions())))' > "$output_dir/packages.txt"

srun "$python_bin" -m src.benchmarks.metric_benchmark \
    --metric "$metric" --device cuda "$@" \
    --warmup 5 --repeats 30 --output "$output_dir/${metric}_baseline"

srun nsys profile --trace=cuda,nvtx,osrt --sample=none --cuda-memory-usage=true \
    --output="$output_dir/${metric}_trace" \
    "$python_bin" -m src.benchmarks.metric_benchmark \
    --metric "$metric" --device cuda "$@" --nvtx \
    --warmup 5 --repeats 3 --output "$output_dir/${metric}_profile"
