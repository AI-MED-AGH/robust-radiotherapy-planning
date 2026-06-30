from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from src.pipeline.config import MaisiTestingConfig


def _is_planning_ct_path(path: str | Path) -> bool:
    """Determine whether a CT path identifies the planning scan.

    A planning CT is identified by the ``"fraction_1_"`` marker in its
    filename.

    Parameters
    ----------
    path : str | Path
        Path to a CT image.

    Returns
    -------
    bool
        ``True`` when the filename denotes a planning CT; otherwise, ``False``.
    """

    return "fraction_1_" in Path(path).name


def _extract_all_ct_paths(data_path_list: list[str] | list[dict[str, Any]]) -> list[str]:
    """
    Extract all unique CT paths from the evaluation data list.

    Assumptions:
    - The data list may contain either raw path strings or dictionaries.
    - If dictionaries are used, CT paths are read from "moving_image" and
      "fixed_image" when present.

    Parameters
    ----------
    data_path_list : list[str] | list[dict[str, Any]]
        List of CT paths or dictionaries describing CT pairs.

    Returns
    -------
    ct_paths : list[str]
        List of unique paths pointing to CT images.

    Raises
    ------
    ValueError
        If an item in the data list has an unsupported format.
    """

    ct_paths: list[str] = []
    seen_paths: set[str] = set()

    for item in data_path_list:
        if isinstance(item, str):
            item_paths = [item]

        elif isinstance(item, dict):
            item_paths = [cast(str, item[key]) for key in ("moving_image", "fixed_image") if key in item]
            if len(item_paths) == 0:
                raise ValueError(
                    "Expected dictionary item to contain at least one of 'moving_image' or 'fixed_image'. "
                    f"Got keys: {list(item.keys())}"
                )

        else:
            raise ValueError(f"Each test item must be either a string path or a dictionary. Got: {type(item)}")

        for path in item_paths:
            if path not in seen_paths:
                ct_paths.append(path)
                seen_paths.add(path)

    return ct_paths


def _extract_patient_id(path: str | Path) -> str:
    """
    Extract patient ID from a CT tensor filename or generated output path.

    Supported filename patterns:
    - Planning CT / original CT:
        Patient_1_fraction_1_.pt -> Patient_1
    - Generated CT:
        Patient_1_gen_1.pt -> Patient_1

    If neither pattern is found, the parent folder name is returned as a
    fallback. This supports folder-based generated outputs such as:
        generated_ct/test/Patient_1/Patient_1_gen_1.pt

    Parameters
    ----------
    path : str | Path
        Path to a CT tensor file.

    Returns
    -------
    patient_id : str
        Extracted patient ID.

    Raises
    ------
    ValueError
        If `path` is empty.
        If patient ID cannot be extracted from the filename or parent folder.
    """

    path = Path(path)

    if str(path).strip() == "":
        raise ValueError("`path` cannot be empty")

    name = path.name

    if "_fraction_" in name:
        patient_id = name.split("_fraction_")[0]

    elif "_gen_" in name:
        patient_id = name.split("_gen_")[0]

    else:
        # fallback for folder-based generated outputs
        patient_id = path.parent.name

    if patient_id == "":
        raise ValueError(f"Could not extract patient ID from path: {path}")

    if patient_id in {".", ".."}:
        raise ValueError(f"Invalid patient ID extracted from path: {path}")

    return patient_id


def _exclude_planning_cts(ct_groups: dict[str, list[Path]]) -> dict[str, list[Path]]:
    """Return CT groups with planning CT tensors removed.

    The returned dictionary preserves every patient key and the order of each
    patient's non-planning CT paths.

    Parameters
    ----------
    ct_groups : dict[str, list[Path]]
        CT tensor paths grouped by patient identifier.

    Returns
    -------
    dict[str, list[Path]]
        New patient groups containing only non-planning CT tensor paths.
    """

    return {
        patient_id: [path for path in paths if not _is_planning_ct_path(path)]
        for patient_id, paths in ct_groups.items()
    }


def _load_tensor(path: str | Path) -> torch.Tensor:
    """
    Load a tensor from a `.pt` file and normalize its shape.

    This helper loads tensors saved by different stages of the MAISI testing
    pipeline. It supports files that contain either a raw tensor or a
    dictionary with an `"image"` key.

    Shape handling:
    - If the tensor has 5 dimensions, the first dimension is treated as a
      batch dimension and removed.
    - If the tensor has 4 dimensions, the first dimension is treated as a
      channel dimension and removed.
    - The returned tensor is always converted to float and moved to CPU.

    Parameters
    ----------
    path : str | Path
        Path to the `.pt` tensor file.

    Returns
    -------
    tensor : torch.Tensor
        Loaded tensor as a CPU float tensor, usually with shape:
        `(D, H, W)` or equivalent spatial dimensions.

    Raises
    ------
    ValueError
        If the loaded object is a dictionary but does not contain an `"image"`
        key.
    """
    loaded = cast(torch.Tensor | dict[str, torch.Tensor], torch.load(path, weights_only=True))

    if isinstance(loaded, dict):
        if "image" in loaded:
            tensor = loaded["image"]
        else:
            raise ValueError(f"Unsupported tensor dictionary keys: {loaded.keys()}")
    else:
        tensor = loaded

    tensor = tensor.float()

    # Remove batch dimension if present
    if tensor.ndim == 5:
        tensor = tensor[0]

    # Remove channel dimension if present
    if tensor.ndim == 4:
        tensor = tensor[0]

    return tensor.cpu()


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
    values : torch.Tensor | np.ndarray
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


def _to_tensor_hu(
    tensor: torch.Tensor,
    data_min: float,
    data_max: float,
    device: torch.device | str,
) -> torch.Tensor:
    """
    Convert a CT tensor to rounded HU values on a target device.

    This helper supports tensors already in HU-like units as well as tensors
    normalized to ``[0, 1]``. Normalized tensors are linearly rescaled to
    ``[data_min, data_max]``.

    Parameters
    ----------
    tensor : torch.Tensor
        Input CT tensor.

    data_min : float
        Minimum HU value represented by normalized input tensors.

    data_max : float
        Maximum HU value represented by normalized input tensors.

    device : torch.device | str
        Device where the returned tensor should live.

    Returns
    -------
    torch.Tensor
        Float tensor of rounded HU values on ``device``.

    Raises
    ------
    ValueError
        If the tensor is empty, contains non-finite values, or ``data_min`` is
        greater than or equal to ``data_max``.
    """

    if tensor.numel() == 0:
        raise ValueError("`tensor` cannot be empty")

    tensor = tensor.detach().float().to(device)

    if not torch.isfinite(tensor).all():
        raise ValueError("`tensor` contains NaN or infinite values")

    if data_min >= data_max:
        raise ValueError(f"`data_min` must be smaller than `data_max`. Got data_min={data_min}, data_max={data_max}")

    if tensor.min() < 0.0 or tensor.max() > 1.0:
        return tensor.round()

    return ((data_max - data_min) * tensor + data_min).round()


def _load_hu_tensor(
    path: Path,
    config: MaisiTestingConfig,
    device: torch.device,
) -> torch.Tensor:
    """
    Load one CT tensor file and convert it to HU values on a target device.

    Parameters
    ----------
    path : Path
        Path to the saved CT tensor.

    config : MaisiTestingConfig
        Pipeline configuration containing CT intensity range settings.

    device : torch.device
        Device where the returned tensor should be cached.

    Returns
    -------
    torch.Tensor
        CT tensor in rounded HU values on ``device``.
    """

    return _to_tensor_hu(
        _load_tensor(path),
        data_min=config.data_min,
        data_max=config.data_max,
        device=device,
    )


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
