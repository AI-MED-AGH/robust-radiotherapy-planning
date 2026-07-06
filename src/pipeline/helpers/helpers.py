import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from monai.data.meta_tensor import MetaTensor
from monai.transforms import (  # type: ignore[attr-defined]
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    SpatialPadd,
)

from src.pipeline.config import MaisiTestingConfig

logger = logging.getLogger(__name__)


@dataclass
class _StructureRegistrationResult:
    """
    Store the CPU registration work product for one generated CT.

    A result contains either a generated CT plus the transform needed for
    structure warping, or a warning explaining why that generated CT should be
    skipped. It is used by the background registration queue so metric
    calculation can consume completed CPU work in generated-path order.

    Attributes
    ----------
    gen_path : Path
        Path to the generated CT tensor associated with this result.

    gen_ct : torch.Tensor | None
        Generated CT tensor loaded on CPU, or ``None`` when registration was
        skipped before a valid transform could be produced.

    transform : Any | None
        SimpleITK transform mapping generated/fixed-grid points into
        planning/moving CT space, or ``None`` when registration was skipped.

    info_message : str | None, optional
        Informational log message to emit after the result is consumed.

    warning : str | None, optional
        Warning message to emit when the generated CT cannot be evaluated.
    """

    gen_path: Path
    gen_ct: torch.Tensor | None
    transform: Any | None
    info_message: str | None = None
    warning: str | None = None


class LPIPSModel(Protocol):
    """Callable interface implemented by LPIPS model instances."""

    def __call__(self, pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor: ...


DemonsRegistrationAlgorithm = (
    sitk.DemonsRegistrationFilter
    | sitk.DiffeomorphicDemonsRegistrationFilter
    | sitk.FastSymmetricForcesDemonsRegistrationFilter
)


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


def _get_structure_label_transform(config: MaisiTestingConfig) -> Compose:
    """
    Create the preprocessing transform for structure label maps.

    This mirrors the geometric part of CT preprocessing but deliberately skips
    intensity scaling. Structure masks are categorical labels, so interpolation
    or HU scaling would corrupt the label values. The transform is built once
    per structure-metric run and reused for all label maps.

    Parameters
    ----------
    config : MaisiTestingConfig
        Pipeline configuration containing target image size.

    Returns
    -------
    transform : Compose
        MONAI dictionary transform that loads a label-map NIfTI file and
        returns an integer tensor on the pipeline target grid.
    """

    return Compose(
        [
            LoadImaged(keys="label"),
            EnsureChannelFirstd(keys="label"),
            Orientationd(keys="label", axcodes="RAS", labels=None),
            SpatialPadd(
                keys="label",
                spatial_size=config.target_image_size,
                mode="constant",
                constant_values=0,
            ),
            CenterSpatialCropd(keys="label", roi_size=config.target_image_size),
            EnsureTyped(keys="label", dtype=torch.int16),
        ]
    )


def _load_structure_label_map(path: Path, transform: Compose) -> torch.Tensor:
    """
    Load and preprocess a structure label map to the pipeline target grid.

    The transform mirrors the CT preprocessing geometry: load image, enforce
    channel-first format, reorient to RAS, pad/crop to the configured target
    image size, and return an integer label tensor.

    Parameters
    ----------
    path : Path
        Path to a structure label-map NIfTI file.

    transform : Compose
        Label-map preprocessing transform returned by
        ``_get_structure_label_transform``.

    Returns
    -------
    label_map : torch.Tensor
        Integer 3D label map on CPU with shape matching
        ``config.target_image_size``.
    """

    transformed = transform({"label": str(path)})
    label_meta: MetaTensor = transformed["label"]
    label = label_meta.as_tensor().detach().cpu()[0]

    return label.round().to(torch.int16)


def _label_to_binary_mask(label_map: torch.Tensor, label: int) -> torch.Tensor:
    """
    Extract a binary structure mask from a label map.

    Label 0 is treated as an aggregate foreground request and returns all
    non-zero labels. Positive labels return only that exact label value.

    Parameters
    ----------
    label_map : torch.Tensor
        3D integer structure label map.

    label : int
        Structure label to extract. Use 0 for any non-background structure.

    Returns
    -------
    mask : torch.Tensor
        Boolean 3D mask for the requested structure.
    """

    if label <= 0:
        return label_map != 0

    return label_map == label


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


def _as_binary_mask_pair(
    pred: torch.Tensor | np.ndarray,
    ref: torch.Tensor | np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert two mask-like inputs to validated boolean 3D tensors.

    This helper centralizes mask validation for DSC and Hausdorff metrics. Any
    non-zero value is treated as foreground, which supports binary masks saved
    as integer, float, or boolean arrays.

    Parameters
    ----------
    pred : torch.Tensor | np.ndarray
        Predicted, generated, or warped mask-like 3D array.

    ref : torch.Tensor | np.ndarray
        Reference mask-like 3D array.

    Returns
    -------
    pred_t : torch.Tensor
        Boolean tensor for the predicted mask.

    ref_t : torch.Tensor
        Boolean tensor for the reference mask.

    Raises
    ------
    ValueError
        If either mask has zero elements.
        If either mask is not 3-dimensional.
        If the masks have different shapes.
    """

    device = _resolve_metric_device(pred, ref)
    pred_t = _as_metric_tensor(pred, device=device) > 0
    ref_t = _as_metric_tensor(ref, device=device) > 0

    if pred_t.numel() == 0 or ref_t.numel() == 0:
        raise ValueError("Masks cannot have zero elements")

    if pred_t.ndim != 3 or ref_t.ndim != 3:
        raise ValueError(f"Masks must be 3D. Got {tuple(pred_t.shape)} and {tuple(ref_t.shape)}")

    if pred_t.shape != ref_t.shape:
        raise ValueError(f"Masks must have the same shape. Got {tuple(pred_t.shape)} and {tuple(ref_t.shape)}")

    return pred_t, ref_t


def _surface_voxels(mask: torch.Tensor) -> torch.Tensor:
    """
    Identify foreground surface voxels in a 3D binary mask.

    A foreground voxel is considered part of the surface when its 3x3x3
    neighborhood is not completely filled with foreground. Padding makes
    foreground voxels on the image boundary count as surface voxels.

    Parameters
    ----------
    mask : torch.Tensor
        Boolean 3D foreground mask.

    Returns
    -------
    surface : torch.Tensor
        Boolean 3D mask containing only foreground surface voxels.
    """

    volume = mask.float().unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32, device=mask.device)
    neighbor_count = F.conv3d(volume, kernel, padding=1).squeeze(0).squeeze(0)

    return mask & (neighbor_count < 27.0)


def _nearest_distances(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    batch_size: int = 512,
) -> torch.Tensor:
    """
    Calculate nearest-neighbor distances from source points to target points.

    Distances are computed in batches to avoid materializing one very large
    pairwise distance matrix for large structure surfaces.

    Parameters
    ----------
    source_points : torch.Tensor
        Tensor of source coordinates with shape ``[N, 3]``.

    target_points : torch.Tensor
        Tensor of target coordinates with shape ``[M, 3]``.

    batch_size : int
        Number of source points processed per distance batch.

    Returns
    -------
    distances : torch.Tensor
        One nearest-target distance for each source point, shape ``[N]``.
    """

    chunks = []
    for start in range(0, source_points.shape[0], batch_size):
        chunk = source_points[start : start + batch_size]
        chunks.append(torch.cdist(chunk, target_points).min(dim=1).values)

    return torch.cat(chunks)


def _structure_path_for_ct(ct_path: Path, config: MaisiTestingConfig) -> Path:
    """
    Resolve the original structure label-map path for a processed CT tensor.

    Processed CT tensors are saved as ``.pt`` files, while structure label maps
    remain in the original ``STRUCTURES`` tree as ``.nii.gz`` files. This helper
    maps the processed tensor filename back to the matching original structure
    filename using the patient ID and fraction name.

    Parameters
    ----------
    ct_path : Path
        Processed CT tensor path, for example
        ``Patient_01_fraction_2_.pt``.

    config : MaisiTestingConfig
        Pipeline configuration containing ``structures_root``.

    Returns
    -------
    structure_path : Path
        Expected path to the corresponding structure label-map NIfTI file.
    """

    patient_id = _extract_patient_id(ct_path)
    structure_name = ct_path.name.replace(".pt", ".nii.gz")
    return config.structures_root / patient_id / structure_name


def _surface_distances(
    pred: torch.Tensor | np.ndarray,
    ref: torch.Tensor | np.ndarray,
    spacing: tuple[float, float, float],
) -> torch.Tensor | float:
    """
    Calculate symmetric foreground-surface distances for two masks.

    Surface voxels are extracted from both masks, converted to physical
    coordinates using ``spacing``, and compared in both directions with
    batched nearest-neighbor distances. Foreground-empty masks follow the
    metric convention used by the structure evaluation: both foreground-empty
    masks return ``0.0`` and one foreground-empty mask returns positive
    infinity.

    Parameters
    ----------
    pred : torch.Tensor | np.ndarray
        Predicted, generated, or warped 3D structure mask.

    ref : torch.Tensor | np.ndarray
        Reference 3D structure mask.

    spacing : tuple[float, float, float]
        Physical voxel spacing for the mask axes.

    Returns
    -------
    distances : torch.Tensor | float
        Symmetric surface distances for masks with foreground voxels, or a
        scalar foreground-empty convention value.

    Raises
    ------
    ValueError
        If either mask has zero elements.
        If either mask is not 3-dimensional.
        If the masks have different shapes.
    """

    pred_t, ref_t = _as_binary_mask_pair(pred, ref)
    pred_empty = int(pred_t.sum().item()) == 0
    ref_empty = int(ref_t.sum().item()) == 0

    if pred_empty and ref_empty:
        return 0.0

    if pred_empty or ref_empty:
        return float("inf")

    # Hausdorff distance is defined on mask boundaries
    pred_surface = _surface_voxels(pred_t)
    ref_surface = _surface_voxels(ref_t)

    # Surface coordinates are much smaller than dense foreground coordinates
    pred_points = pred_surface.nonzero().float()
    ref_points = ref_surface.nonzero().float()
    spacing_t = torch.as_tensor(spacing, dtype=torch.float32, device=pred_points.device)
    pred_points = pred_points * spacing_t
    ref_points = ref_points * spacing_t

    # Use both directions so HD is symmetric
    distances = torch.cat(
        [
            _nearest_distances(pred_points, ref_points),
            _nearest_distances(ref_points, pred_points),
        ]
    )

    return distances


def _tensor_to_sitk_image(tensor: torch.Tensor, config: MaisiTestingConfig, pixel_id: int) -> sitk.Image:
    """
    Convert a pipeline tensor to a SimpleITK image.

    Pipeline tensors use shape ``[H, W, D]``. SimpleITK images are created from
    arrays ordered as ``[D, H, W]``, so the tensor axes are transposed before
    conversion. Spacing is also reversed to match the SimpleITK x/y/z axis
    convention.

    Parameters
    ----------
    tensor : torch.Tensor
        Pipeline image or mask tensor with shape ``[H, W, D]``.

    config : MaisiTestingConfig
        Pipeline configuration containing voxel spacing.

    pixel_id : int
        SimpleITK pixel type, usually ``sitk.sitkFloat32`` for CT images or
        ``sitk.sitkUInt8`` for masks.

    Returns
    -------
    image : sitk.Image
        SimpleITK image with spacing and identity physical metadata assigned.
    """

    arr = tensor.detach().cpu().numpy()
    arr = np.transpose(arr, (2, 0, 1))
    image = sitk.GetImageFromArray(arr.astype(np.float32 if pixel_id == sitk.sitkFloat32 else np.uint8))
    image.SetSpacing((config.spacing[2], config.spacing[1], config.spacing[0]))  # type: ignore[no-untyped-call]
    image.SetOrigin((0.0, 0.0, 0.0))  # type: ignore[no-untyped-call]
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))  # type: ignore[no-untyped-call]
    return image


def _sitk_image_to_tensor(image: sitk.Image) -> torch.Tensor:
    """
    Convert a SimpleITK image back to a pipeline tensor.

    SimpleITK array extraction returns data ordered as ``[D, H, W]``. The array
    is transposed back to the pipeline convention ``[H, W, D]``.

    Parameters
    ----------
    image : sitk.Image
        SimpleITK image.

    Returns
    -------
    tensor : torch.Tensor
        Tensor with shape ``[H, W, D]``.
    """

    arr = sitk.GetArrayFromImage(image)
    arr = np.transpose(arr, (1, 2, 0))
    return torch.as_tensor(arr)


def _smooth_and_resample(image: sitk.Image, shrink_factor: float, smoothing_sigma: float) -> sitk.Image:
    """
    Smooth and downsample an image for multiscale demons registration.

    The image is smoothed in physical units before downsampling, matching the
    multiscale registration strategy used by the DDF workflow. The output
    preserves the original origin and direction while adjusting spacing for the
    lower-resolution grid.

    Parameters
    ----------
    image : sitk.Image
        SimpleITK image to smooth and resample.

    shrink_factor : float
        Factor by which each image dimension is reduced.

    smoothing_sigma : float
        Gaussian smoothing sigma in physical units.

    Returns
    -------
    image : sitk.Image
        Smoothed and resampled SimpleITK image.
    """

    smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_sigma)  # type: ignore[no-untyped-call]

    original_spacing = image.GetSpacing()  # type: ignore[no-untyped-call]
    original_size = image.GetSize()  # type: ignore[no-untyped-call]
    new_size = [max(2, int(sz / shrink_factor + 0.5)) for sz in original_size]
    new_spacing = [
        ((original_sz - 1) * original_spc) / (new_sz - 1)
        for original_sz, original_spc, new_sz in zip(original_size, original_spacing, new_size, strict=True)
    ]

    return sitk.Resample(
        smoothed_image,
        new_size,
        sitk.Transform(),  # type: ignore[no-untyped-call]
        sitk.sitkLinear,
        image.GetOrigin(),  # type: ignore[no-untyped-call]
        new_spacing,
        image.GetDirection(),  # type: ignore[no-untyped-call]
        0.0,
        image.GetPixelID(),  # type: ignore[no-untyped-call]
    )


def _multiscale_demons(
    registration_algorithm: DemonsRegistrationAlgorithm,
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    initial_transform: sitk.Transform,
    shrink_factors: list[float],
    smoothing_sigmas: list[float],
) -> sitk.DisplacementFieldTransform:
    """
    Run demons registration from coarse to full resolution.

    This helper initializes a displacement field on the coarsest requested
    level, then refines it toward the original image grid one level at a time.
    It creates only the current fixed/moving pyramid images, rather than
    keeping the full pyramid in memory.

    Parameters
    ----------
    registration_algorithm : DemonsRegistrationAlgorithm
        SimpleITK demons registration filter with an ``Execute`` method.

    fixed_image : sitk.Image
        SimpleITK fixed image defining the output spatial domain.

    moving_image : sitk.Image
        SimpleITK moving image that is registered into fixed-image space.

    initial_transform : sitk.Transform
        SimpleITK transform used to initialize the displacement field.

    shrink_factors : list[float]
        Pyramid shrink factors, ordered from coarse to fine in configuration
        style. The original resolution is handled implicitly.

    smoothing_sigmas : list[float]
        Gaussian smoothing sigmas for each shrink factor.

    Returns
    -------
    transform : sitk.DisplacementFieldTransform
        SimpleITK displacement-field transform mapping fixed-image points to
        moving-image points for resampling into fixed-image space.
    """

    pyramid_levels = list(zip(shrink_factors, smoothing_sigmas, strict=True))
    if len(pyramid_levels) > 0:
        first_shrink_factor, first_smoothing_sigma = pyramid_levels[0]
        fixed_level = _smooth_and_resample(fixed_image, first_shrink_factor, first_smoothing_sigma)
        moving_level = _smooth_and_resample(moving_image, first_shrink_factor, first_smoothing_sigma)
        remaining_levels = pyramid_levels[1:]
    else:
        fixed_level = fixed_image
        moving_level = moving_image
        remaining_levels = []

    # Demons filters require displacement fields with vector float64 pixels
    displacement_field = sitk.TransformToDisplacementField(  # type: ignore[no-untyped-call]
        initial_transform,
        sitk.sitkVectorFloat64,
        fixed_level.GetSize(),  # type: ignore[no-untyped-call]
        fixed_level.GetOrigin(),  # type: ignore[no-untyped-call]
        fixed_level.GetSpacing(),  # type: ignore[no-untyped-call]
        fixed_level.GetDirection(),  # type: ignore[no-untyped-call]
    )

    # Start registration on the coarsest grid so large deformations are found
    # before the full-resolution details
    displacement_field = registration_algorithm.Execute(  # type: ignore[no-untyped-call]
        fixed_level,
        moving_level,
        displacement_field,
    )

    for shrink_factor, smoothing_sigma in remaining_levels:
        fixed_level = _smooth_and_resample(fixed_image, shrink_factor, smoothing_sigma)
        moving_level = _smooth_and_resample(moving_image, shrink_factor, smoothing_sigma)

        # Carry the current DVF estimate onto the next finer fixed-image grid
        displacement_field = sitk.Resample(displacement_field, fixed_level)
        displacement_field = registration_algorithm.Execute(  # type: ignore[no-untyped-call]
            fixed_level,
            moving_level,
            displacement_field,
        )

    # Finish with one refinement on the original fixed-image grid
    displacement_field = sitk.Resample(displacement_field, fixed_image)
    displacement_field = registration_algorithm.Execute(  # type: ignore[no-untyped-call]
        fixed_image,
        moving_image,
        displacement_field,
    )

    return sitk.DisplacementFieldTransform(displacement_field)  # type: ignore[no-untyped-call]


def _register_planning_ct_to_generated_ct(
    planning_ct: torch.Tensor,
    generated_ct: torch.Tensor,
    config: MaisiTestingConfig,
) -> Any:
    """
    Register a planning CT to a generated CT.

    The generated CT is used as the fixed image and the planning CT as the
    moving image. For SimpleITK resampling, the returned displacement-field
    transform maps generated/fixed-grid physical points back into
    planning/moving CT space. Reusing that transform lets each planning
    structure mask be sampled onto the generated CT grid.

    Parameters
    ----------
    planning_ct : torch.Tensor
        Planning CT in HU-like units.

    generated_ct : torch.Tensor
        Generated CT in HU-like units.

    config : MaisiTestingConfig
        Pipeline configuration containing spacing and demons-registration
        settings.

    Returns
    -------
    transform : Any
        SimpleITK displacement-field transform mapping generated/fixed-grid
        points into planning/moving CT space for resampling.
    """

    # SimpleITK registration runs on CPU.
    # These conversions detach tensors from any accelerator residency.
    fixed = _tensor_to_sitk_image(generated_ct, config, sitk.sitkFloat32)
    moving = _tensor_to_sitk_image(planning_ct, config, sitk.sitkFloat32)

    demons_filter = sitk.DiffeomorphicDemonsRegistrationFilter()  # type: ignore[no-untyped-call]
    demons_filter.SetNumberOfIterations(config.structure_registration_iterations)  # type: ignore[no-untyped-call]
    demons_filter.SetSmoothDisplacementField(True)  # type: ignore[no-untyped-call]
    demons_filter.SetStandardDeviations(config.structure_registration_sigma)  # type: ignore[no-untyped-call]

    initial_transform = sitk.CenteredTransformInitializer(  # type: ignore[no-untyped-call]
        fixed,
        moving,
        sitk.Euler3DTransform(),  # type: ignore[no-untyped-call]
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    # The multiscale helper rasterizes the initial rigid transform into a DVF,
    # then refines that field from coarse to full resolution.
    return _multiscale_demons(
        registration_algorithm=demons_filter,
        fixed_image=fixed,
        moving_image=moving,
        initial_transform=initial_transform,
        shrink_factors=config.structure_registration_shrink_factors,
        smoothing_sigmas=config.structure_registration_smoothing_sigmas,
    )


def _warp_planning_mask_to_generated_ct(
    planning_mask: torch.Tensor,
    generated_ct: torch.Tensor,
    transform: Any,
    config: MaisiTestingConfig,
) -> torch.Tensor:
    """
    Warp a planning structure mask into generated CT space.

    The transform maps each generated-grid output point back into planning
    mask space, which is the direction expected by ``sitk.Resample``.
    Nearest-neighbor interpolation is used to preserve binary mask labels.

    Parameters
    ----------
    planning_mask : torch.Tensor
        Boolean or binary 3D structure mask in planning CT space.

    generated_ct : torch.Tensor
        Generated CT tensor used as the resampling reference grid.

    transform : Any
        SimpleITK transform mapping generated/fixed-grid points into
        planning/moving mask space.

    config : MaisiTestingConfig
        Pipeline configuration containing spacing and image-grid settings.

    Returns
    -------
    warped_mask : torch.Tensor
        Boolean 3D mask in generated CT space.
    """

    # The generated CT is used only as the reference grid: size, spacing,
    # origin, and direction define where the warped planning mask is sampled.
    fixed = _tensor_to_sitk_image(generated_ct, config, sitk.sitkFloat32)
    moving_mask = _tensor_to_sitk_image(planning_mask.to(torch.uint8), config, sitk.sitkUInt8)

    resampler = sitk.ResampleImageFilter()  # type: ignore[no-untyped-call]
    resampler.SetReferenceImage(fixed)  # type: ignore[no-untyped-call]
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # type: ignore[no-untyped-call]
    resampler.SetDefaultPixelValue(0)  # type: ignore[no-untyped-call]
    resampler.SetTransform(transform)  # type: ignore[no-untyped-call]
    warped = resampler.Execute(moving_mask)  # type: ignore[no-untyped-call]

    return _sitk_image_to_tensor(warped) > 0


def _calculate_structure_registration_result(
    gen_path: Path,
    patient_id: str,
    planning_ct: torch.Tensor,
    config: MaisiTestingConfig,
) -> _StructureRegistrationResult:
    """
    Load one generated CT and calculate its planning-to-generated transform.

    This helper is intentionally CPU-only so it can run in a background thread
    while CUDA evaluates metrics for the previous generated CT. Shape
    mismatches are returned as warnings instead of being raised, which lets the
    caller preserve the existing skip-and-continue behavior.

    Parameters
    ----------
    gen_path : Path
        Path to the generated CT tensor that should be registered.

    patient_id : str
        Patient identifier used only for diagnostic messages.

    planning_ct : torch.Tensor
        Planning CT tensor in HU-like units on CPU.

    config : MaisiTestingConfig
        Pipeline configuration containing CT intensity range, spacing, and
        demons-registration settings.

    Returns
    -------
    result : _StructureRegistrationResult
        Registration result containing the generated CT and transform when
        registration succeeds, or a warning when the generated CT is skipped.
    """

    cpu_device = torch.device("cpu")
    gen_ct = _load_hu_tensor(gen_path, config=config, device=cpu_device)

    if gen_ct.shape != planning_ct.shape:
        warning = (
            "Skipping structure registration for "
            f"{patient_id}: {gen_path.name} {tuple(gen_ct.shape)} vs planning {tuple(planning_ct.shape)}"
        )
        return _StructureRegistrationResult(
            gen_path=gen_path,
            gen_ct=None,
            transform=None,
            warning=warning,
        )

    transform = _register_planning_ct_to_generated_ct(
        planning_ct=planning_ct,
        generated_ct=gen_ct,
        config=config,
    )

    return _StructureRegistrationResult(
        gen_path=gen_path,
        gen_ct=gen_ct,
        transform=transform,
        info_message=f"Generated structure DVF for {patient_id}: planning -> {gen_path.name}",
    )
