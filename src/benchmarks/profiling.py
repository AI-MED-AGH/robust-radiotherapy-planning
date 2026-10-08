"""Measurement helpers; CUDA counters describe the entire process."""

import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def nvtx_range(name: str, enabled: bool) -> Iterator[None]:
    if enabled:
        torch.cuda.nvtx.range_push(name)  # type: ignore[no-untyped-call]
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()  # type: ignore[no-untyped-call]


def memory_counters(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {}
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def cpu_peak_rss_bytes() -> int | None:
    """Linux process lifetime high-water mark, including setup and input loading."""
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024  # type: ignore[attr-defined]
    except ImportError:
        return None


class DeviceMemorySampler:
    """Sample device-wide usage by UUID; samples may miss brief allocation peaks."""

    def __init__(self, uuid: str | None, interval: float = 0.2):
        self.uuid = uuid
        self.interval = interval
        self.samples: list[int] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "-i", str(self.uuid), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                self.samples.append(int(result.stdout.strip()) * 1024**2)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                self.error = str(exc)
                return
            self._stop.wait(self.interval)

    def start(self) -> None:
        if self.uuid is not None:
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=6)
