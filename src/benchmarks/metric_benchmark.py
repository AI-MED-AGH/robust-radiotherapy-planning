"""Benchmark production metrics: python -m src.benchmarks.metric_benchmark --help."""

import argparse
import csv
import json
import platform
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.benchmarks.profiling import DeviceMemorySampler, cpu_peak_rss_bytes, memory_counters, nvtx_range
from src.pipeline.evaluation.image_metrics import ssim_3d
from src.pipeline.evaluation.structure_metrics import hausdorff_and_hd95
from src.pipeline.helpers.helpers_structures import _surface_voxels


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def load_tensor(path: Path) -> torch.Tensor:
    """Accept prepared 3D tensors only, without silently changing geometry or intensity."""
    if path.suffix == ".npy":
        value = torch.from_numpy(np.load(path, allow_pickle=False))
    else:
        value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise ValueError(f"{path}: expected a single 3D tensor (.pt or .npy)")
    return value.detach().contiguous()


def inputs(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    if args.pred is not None:
        pred, ref = load_tensor(args.pred), load_tensor(args.ref)
        if pred.shape != ref.shape:
            raise ValueError("Input shapes must match")
    elif args.metric == "ssim":
        generator = torch.Generator().manual_seed(args.seed)
        ref = torch.rand(tuple(args.shape), generator=generator)
        pred = (ref + 0.03 * torch.randn(tuple(args.shape), generator=generator)).clamp(0, 1)
        ref = ref * (args.data_max - args.data_min) + args.data_min
        pred = pred * (args.data_max - args.data_min) + args.data_min
    else:
        # Smooth ellipsoids provide bounded, reproducible surface sizes.
        axes = [torch.linspace(-1, 1, n) for n in args.shape]
        x, y, z = torch.meshgrid(*axes, indexing="ij")
        radius = args.mask_radius
        ref = (x / radius) ** 2 + (y / radius) ** 2 + (z / radius) ** 2 <= 1
        pred = ((x - 0.05) / radius) ** 2 + (y / radius) ** 2 + (z / radius) ** 2 <= 1
    if not torch.isfinite(pred).all() or not torch.isfinite(ref).all():
        raise ValueError("Inputs must contain finite values")
    if args.metric == "hausdorff":
        if args.label > 0:
            return pred == args.label, ref == args.label
        return pred > 0, ref > 0
    return pred.float(), ref.float()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--metric", choices=["ssim", "hausdorff"], required=True)
    result.add_argument("--device", default="cuda")
    result.add_argument("--pred", type=Path)
    result.add_argument("--ref", type=Path)
    result.add_argument("--shape", type=int, nargs=3, default=[512, 512, 128])
    result.add_argument("--spacing", type=float, nargs=3, default=[1.171875, 1.171875, 3.0])
    result.add_argument("--data-min", type=float, default=-1000)
    result.add_argument("--data-max", type=float, default=1000)
    result.add_argument("--label", type=int, default=0, help="Hausdorff label; 0 selects positive foreground")
    result.add_argument("--mask-radius", type=float, default=0.45)
    result.add_argument("--chunk-size", type=int, default=512)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--warmup", type=int, default=5)
    result.add_argument("--repeats", type=int, default=30)
    result.add_argument("--workers", type=int, default=1, help="Independent calls on separate CUDA streams")
    result.add_argument("--resident-pairs", type=int, default=1, help="Independent pairs held on device")
    result.add_argument("--scope", choices=["resident", "transfer"], default="resident")
    result.add_argument("--nvtx", action="store_true")
    result.add_argument("--output", type=Path, default=Path("RESULTS/benchmarks/run"))
    return result


def run(args: argparse.Namespace, device: torch.device, metadata: dict[str, Any]) -> None:
    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    load_start = time.perf_counter()
    with nvtx_range("load_inputs", args.nvtx):
        cpu_pair = inputs(args)
    metadata["input_load_ms"] = (time.perf_counter() - load_start) * 1000
    metadata["shape"] = list(cpu_pair[0].shape)
    metadata["input_pair_bytes"] = sum(t.numel() * t.element_size() for t in cpu_pair)
    if args.metric == "hausdorff":
        metadata["surface_points"] = [int(_surface_voxels(t).sum().item()) for t in cpu_pair]
        metadata["metric_definition"] = "symmetric surface HD and pooled symmetric HD95"
    else:
        metadata["metric_definition"] = "MONAI full-volume 3D SSIM, uniform window, production validation"

    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    transfer_start = time.perf_counter()
    with nvtx_range("transfer_inputs", args.nvtx):
        if args.scope == "resident":
            for _ in range(args.resident_pairs):
                pairs.append(tuple(t.to(device=device, copy=True) for t in cpu_pair))  # type: ignore[arg-type]
    synchronize()
    metadata["initial_transfer_ms"] = (time.perf_counter() - transfer_start) * 1000
    streams = (
        [torch.cuda.Stream(device=device) for _ in range(args.workers)]  # type: ignore[no-untyped-call]
        if device.type == "cuda"
        else []
    )

    def metric(pair: tuple[torch.Tensor, torch.Tensor]) -> tuple[float, ...]:
        with nvtx_range(f"metric/{args.metric}", args.nvtx):
            if args.metric == "ssim":
                return (ssim_3d(*pair, args.data_min, args.data_max),)
            return hausdorff_and_hd95(*pair, spacing=tuple(args.spacing), batch_size=args.chunk_size)

    def work(index: int) -> tuple[tuple[float, ...], float | None]:
        def call() -> tuple[float, ...]:
            if args.scope == "transfer":
                with nvtx_range("transfer_inputs", args.nvtx):
                    pair = tuple(t.to(device=device, copy=True) for t in cpu_pair)
            else:
                pair = pairs[index]
            return metric(pair)  # type: ignore[arg-type]

        if device.type != "cuda":
            return call(), None
        with torch.cuda.device(device), torch.cuda.stream(streams[index]):
            start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            start.record()
            score = call()
            end.record()
            end.synchronize()
            return score, start.elapsed_time(end)

    rows: list[dict[str, Any]] = []
    sampler = DeviceMemorySampler(metadata.get("gpu_uuid"))
    sampler.start()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for iteration in range(1 + args.warmup + args.repeats):
                phase = "cold" if iteration == 0 else "warmup" if iteration <= args.warmup else "measured"
                synchronize()
                baseline = memory_counters(device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                start_time = time.perf_counter()
                results = list(executor.map(work, range(args.workers)))
                synchronize()
                wall_ms = (time.perf_counter() - start_time) * 1000
                counters = memory_counters(device)
                row: dict[str, Any] = {
                    "iteration": iteration,
                    "phase": phase,
                    "workers": args.workers,
                    "wall_ms": wall_ms,
                    "comparisons_per_second": args.workers * 1000 / wall_ms,
                    "scores_json": json.dumps([r[0] for r in results]),
                    "cuda_event_ms_json": json.dumps([r[1] for r in results]),
                    "cpu_process_lifetime_peak_rss_bytes": cpu_peak_rss_bytes(),
                    **{f"baseline_{k}": v for k, v in baseline.items() if not k.startswith("peak_")},
                    **counters,
                }
                if device.type == "cuda":
                    row["incremental_peak_allocated_bytes"] = (
                        counters["peak_allocated_bytes"] - baseline["allocated_bytes"]
                    )
                rows.append(row)
    finally:
        sampler.stop()
        metadata["device_wide_sampled_peak_bytes"] = max(sampler.samples, default=None)
        metadata["device_memory_sampling_error"] = sampler.error
        metadata["device_memory_sampling_interval_seconds"] = sampler.interval
        # Save completed iterations even if a later capacity experiment runs out of memory.
        if rows:
            with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    times = [float(row["wall_ms"]) for row in rows if row["phase"] == "measured"]
    metadata["summary"] = {
        "median_wall_ms": statistics.median(times),
        "p95_wall_ms": float(np.percentile(times, 95)),
        "aggregate_comparisons_per_second": args.workers * len(times) * 1000 / sum(times),
        "cpu_process_lifetime_peak_rss_bytes": cpu_peak_rss_bytes(),
    }
    if device.type == "cuda":
        for key in ("peak_allocated_bytes", "peak_reserved_bytes", "incremental_peak_allocated_bytes"):
            metadata["summary"][key] = max(row[key] for row in rows if row["phase"] == "measured")
    print(json.dumps(metadata["summary"], indent=2))


def main() -> None:
    arg_parser = parser()
    args = arg_parser.parse_args()
    if (args.pred is None) != (args.ref is None):
        arg_parser.error("--pred and --ref must be supplied together")
    if min(args.repeats, args.workers, args.resident_pairs, args.chunk_size, *args.shape, *args.spacing) <= 0:
        arg_parser.error("Sizes, repeats, workers and spacing must be positive")
    if args.warmup < 0 or not 0 < args.mask_radius < 1 or args.data_min >= args.data_max:
        arg_parser.error("Invalid warmup, mask radius or data range")
    if args.scope == "resident" and args.resident_pairs < args.workers:
        arg_parser.error("--resident-pairs must be at least --workers")
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        arg_parser.error("Supported devices: cpu and cuda")
    if args.nvtx and device.type != "cuda":
        arg_parser.error("--nvtx requires CUDA")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "torch": torch.__version__,
        "monai": version("monai"),
        "cuda_runtime": torch.version.cuda,
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "nsys": command_output(["nsys", "--version"]),
        "nvidia_smi": command_output(["nvidia-smi"]),
        "status": "running",
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        metadata.update(gpu_name=properties.name, gpu_total_bytes=properties.total_memory)
        uuid = getattr(properties, "uuid", None)
        uuid_text = str(uuid) if uuid is not None else None
        if uuid_text is not None and not uuid_text.startswith(("GPU-", "MIG-")):
            uuid_text = f"GPU-{uuid_text}"
        metadata["gpu_uuid"] = uuid_text
    try:
        run(args, device, metadata)
        metadata["status"] = "completed"
    except Exception as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
