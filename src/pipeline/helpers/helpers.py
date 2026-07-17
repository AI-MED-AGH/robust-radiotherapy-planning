import gc
from pathlib import Path
from typing import Any, cast

import torch

from src.pipeline.config import MaisiTestingConfig


def _clean_pipeline_memory() -> None:
    """
    Release Python garbage and cached CUDA memory between heavy pipeline stages.

    This helper is intended for use after memory-intensive pipeline stages,
    such as latent encoding, CT generation, and metric calculation. It first
    asks Python to collect unreachable objects, then releases cached CUDA
    allocator blocks back to PyTorch so later stages can reuse GPU memory.

    The CUDA cache call is safe to execute even when CUDA is unavailable.

    Returns
    -------
    None
        This function is called only for its memory-management side effects.
    """

    gc.collect()
    torch.cuda.empty_cache()


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


def _normalised_image_key(path: Path) -> str:
    """
    Return a filename key independent of common medical-image extensions.

    Parameters
    ----------
    path : Path
        Dose or structure file path.

    Returns
    -------
    key : str
        Filename without `.nii.gz`, `.nii`, or `.pt` suffix.
    """

    name = path.name
    for suffix in (".nii.gz", ".nii", ".pt"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem
