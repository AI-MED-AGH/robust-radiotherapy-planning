import logging
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import torch

from src.pipeline.config import MaisiTestingConfig
from src.pipeline.helpers.helpers import _load_hu_tensor

logger = logging.getLogger(__name__)


class LPIPSModel(Protocol):
    """Callable interface implemented by LPIPS model instances."""

    def __call__(self, pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Calculate perceptual distances between two image batches.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted images with shape ``(N, C, H, W)`` and values in the
            range expected by the configured LPIPS model.

        ref : torch.Tensor
            Reference images with the same shape and value range as ``pred``.

        Returns
        -------
        torch.Tensor
            Perceptual distance values produced by the LPIPS model, normally
            one value per input-image pair.
        """

        ...


def _build_lpips_model(config: MaisiTestingConfig) -> LPIPSModel | None:
    """
    Build the LPIPS model requested by the pipeline configuration.

    LPIPS is an optional dependency used only when perceptual metrics are
    enabled. If LPIPS is disabled in the configuration, this function returns
    ``None`` without importing the package. If the package is unavailable, the
    function disables LPIPS on the provided configuration object and returns
    ``None`` so the rest of evaluation can continue.

    Parameters
    ----------
    config : MaisiTestingConfig
        Pipeline configuration containing LPIPS settings, including
        ``use_lpips``, ``lpips_net``, and ``device``.

    Returns
    -------
    lpips_model : LPIPSModel | None
        Initialized LPIPS model in evaluation mode, or ``None`` when LPIPS is
        disabled or unavailable.
    """

    if not config.use_lpips:
        return None

    try:
        import lpips  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("LPIPS package not installed. Skipping LPIPS")
        config.use_lpips = False
        return None

    model = lpips.LPIPS(net=config.lpips_net).to(config.device)
    model.eval()
    return cast(LPIPSModel, model)


def _default_metric_device() -> torch.device:
    """
    Return the fallback device for metric calculations.

    The evaluation metrics operate on torch tensors. When callers provide only
    NumPy arrays, there is no existing tensor device to preserve, so metrics
    prefer CUDA when available and otherwise run on CPU.

    Returns
    -------
    torch.device
        CUDA device when CUDA is available; otherwise CPU.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def _resolve_metric_device(*values: torch.Tensor | np.ndarray) -> torch.device:
    """
    Choose the device used to compare metric inputs.

    Existing tensor devices are preserved. Accelerator tensors take priority
    so mixed tensor/NumPy inputs stay on the accelerator. If all tensor inputs
    are on CPU, CPU is used. If every input is a NumPy array, metrics use the
    default metric device.

    Parameters
    ----------
    *values : torch.Tensor | np.ndarray
        Metric inputs whose devices should be respected when possible.

    Returns
    -------
    torch.device
        Device on which metric tensors should be created or moved.
    """

    cpu_device: torch.device | None = None
    for value in values:
        if isinstance(value, torch.Tensor) and value.device.type != "cpu":
            return value.device

        if isinstance(value, torch.Tensor):
            cpu_device = value.device

    if cpu_device is not None:
        return cpu_device

    return _default_metric_device()


def _as_metric_tensor(value: torch.Tensor | np.ndarray, device: torch.device | str | None = None) -> torch.Tensor:
    """
    Convert a metric input to a float tensor on an optional target device.

    Parameters
    ----------
    value : torch.Tensor | np.ndarray
        Input image or volume used by metric calculations.

    device : torch.device | str | None, optional
        Device to move the returned tensor to. If ``None``, tensors stay on
        their current device and NumPy arrays use the default metric device.

    Returns
    -------
    torch.Tensor
        Detached float tensor suitable for metric calculations.
    """

    target_device = torch.device(device) if device is not None else None

    if isinstance(value, np.ndarray):
        tensor = torch.as_tensor(value, device=target_device or _default_metric_device())
    else:
        tensor = value.detach()

    tensor = tensor.float()

    if target_device is not None:
        tensor = tensor.to(target_device)

    return tensor


def _validate_3d_pair(pred: torch.Tensor, ref: torch.Tensor) -> None:
    """
    Validate two 3D metric tensors before comparing them.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted or generated 3D image tensor.

    ref : torch.Tensor
        Reference 3D image tensor.

    Raises
    ------
    ValueError
        If either tensor is empty, not 3D, shape-mismatched, or contains NaN
        or infinite values.
    """

    if pred.numel() == 0:
        raise ValueError("`pred` cannot be empty")

    if ref.numel() == 0:
        raise ValueError("`ref` cannot be empty")

    if pred.ndim != 3:
        raise ValueError(f"`pred` must be a 3D tensor. Got shape {tuple(pred.shape)}")

    if ref.ndim != 3:
        raise ValueError(f"`ref` must be a 3D tensor. Got shape {tuple(ref.shape)}")

    if pred.shape != ref.shape:
        raise ValueError(f"`pred` and `ref` must have the same shape. Got {tuple(pred.shape)} and {tuple(ref.shape)}")

    if not torch.isfinite(pred).all():
        raise ValueError("`pred` contains NaN or infinite values")

    if not torch.isfinite(ref).all():
        raise ValueError("`ref` contains NaN or infinite values")


def _cache_patient_cts(
    paths: list[Path],
    config: MaisiTestingConfig,
    device: torch.device,
) -> dict[Path, torch.Tensor]:
    """
    Load all CT tensors for one patient into device memory.

    Parameters
    ----------
    paths : list[Path]
        CT tensor paths for a single patient.

    config : MaisiTestingConfig
        Pipeline configuration containing CT intensity range settings.

    device : torch.device
        Device used for metric calculation and cache storage.

    Returns
    -------
    dict[Path, torch.Tensor]
        Mapping from each input path to its loaded HU tensor on ``device``.
    """

    return {path: _load_hu_tensor(path, config=config, device=device) for path in paths}


def _evenly_spaced_slices_for_lpips(
    arr: torch.Tensor | np.ndarray,
    max_slices: int = 32,
) -> torch.Tensor:
    """
    Extract evenly spaced 2D CT slices for LPIPS input format.

    LPIPS expects 2D RGB-like images, so this function:
    - selects up to `max_slices` slices evenly across the full 3D volume
    - converts slices from shape [H, W, Z] to [Z, H, W]
    - adds a channel dimension
    - repeats the single CT channel into 3 channels
    - rescales values from [0, 1] to [-1, 1]

    Parameters
    ----------
    arr : np.ndarray | torch.Tensor
        Input 3D CT array normalized to [0, 1], expected shape [H, W, Z].

    max_slices : int
        Maximum number of evenly spaced slices used for LPIPS calculation.

    Returns
    -------
    tensor : torch.Tensor
        Tensor formatted for LPIPS with shape [N, 3, H, W] and range [-1, 1],
        where N is the number of selected slices.

    Raises
    ------
    ValueError
        If `arr` is empty.
        If `arr` is not 3-dimensional.
        If `arr` contains NaN or infinite values.
        If `arr` is not normalized to [0, 1].
        If `max_slices` is less than 1.
    """

    arr_t = _as_metric_tensor(arr)

    if arr_t.numel() == 0:
        raise ValueError("`arr` cannot be empty")

    if arr_t.ndim != 3:
        raise ValueError(f"`arr` must be a 3D tensor with shape [H, W, Z]. Got shape {tuple(arr_t.shape)}")

    if not torch.isfinite(arr_t).all():
        raise ValueError("`arr` contains NaN or infinite values")

    if arr_t.min() < 0.0 or arr_t.max() > 1.0:
        raise ValueError("`arr` must be normalized to [0, 1] before LPIPS calculation")

    if max_slices < 1:
        raise ValueError("`max_slices` must be at least 1")

    z_dim = int(arr_t.shape[-1])

    if z_dim <= max_slices:
        slice_ids = torch.arange(z_dim, device=arr_t.device)
    else:
        slice_ids = torch.linspace(
            0,
            z_dim - 1,
            steps=max_slices,
            device=arr_t.device,
        ).long()

    tensor = arr_t.index_select(dim=2, index=slice_ids)
    tensor = tensor.permute(2, 0, 1).unsqueeze(1)  # Z, 1, H, W
    tensor = tensor.repeat(1, 3, 1, 1)  # Z, 3, H, W

    tensor = tensor * 2.0 - 1.0

    return tensor
