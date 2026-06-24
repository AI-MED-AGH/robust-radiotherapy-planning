import glob
import itertools
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from scipy.ndimage import sobel
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from src.pipeline.config import MaisiTestingConfig


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
    tensor = torch.load(path, weights_only=True)

    if isinstance(tensor, dict):
        if "image" in tensor:
            tensor = tensor["image"]
        else:
            raise ValueError(f"Unsupported tensor dictionary keys: {tensor.keys()}")

    tensor = tensor.float()

    # Remove batch dimension if present.
    if tensor.ndim == 5:
        tensor = tensor[0]

    # Remove channel dimension if present.
    if tensor.ndim == 4:
        tensor = tensor[0]

    return cast(torch.Tensor, tensor.cpu())


FloatArray = npt.NDArray[np.floating[Any]]


class LPIPSModel(Protocol):
    def __call__(self, pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor: ...


def _to_numpy_hu(
    tensor: torch.Tensor,
    data_min: float,
    data_max: float,
) -> FloatArray:
    """
    Convert a CT tensor to a NumPy array of HU values rounded to the nearest integer.

    This helper supports two expected intensity formats:
    - tensors normalized to the range [0, 1]
    - HU-like tensors in the range approximately [data_min, data_max]

    If the tensor is already outside hu, it is returned rounded as a
    float32 NumPy array. Otherwise, values are linearly rescaled from [0, 1] to [data_min, data_max].

    Parameters
    ----------
    tensor : torch.Tensor
        Input CT tensor.

    data_min: float
        Minimum value allowed for the data.

    data_max: float
        Maximum value allowed for the data.

    Returns
    -------
    arr : np.ndarray
        CT image as a float32 NumPy array normalized to [data_min, data_max].

    Raises
    ------
    ValueError
        If `tensor` is empty.
        If `tensor` contains NaN or infinite values.
        If `data_min` is more than `data_max`.
    """

    if tensor.numel() == 0:
        raise ValueError("`tensor` cannot be empty")

    if not torch.isfinite(tensor).all():
        raise ValueError("`tensor` contains NaN or infinite values")

    if data_min >= data_max:
        raise ValueError(f"`data_min` must be smaller than `data_max`. Got data_min={data_min}, data_max={data_max}")

    arr = tensor.detach().cpu().numpy().astype(np.float32)

    # Case 1: HU-like range [data_min, data_max]
    if arr.min() < 0.0 or arr.max() > 1.0:
        return arr.round()

    # Case 2: normalized to [0, 1]
    arr = ((data_max - data_min) * arr + data_min).astype(np.float32).round()

    return arr


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


def _is_planning_ct_path(path: str | Path) -> bool:
    """
    Check whether a CT tensor path points to the planning CT.
    """

    return "fraction_1_" in Path(path).name


def _exclude_planning_cts(ct_groups: dict[str, list[Path]]) -> dict[str, list[Path]]:
    """
    Remove planning CT paths from grouped CT tensors.
    """

    return {
        patient_id: [path for path in paths if not _is_planning_ct_path(path)]
        for patient_id, paths in ct_groups.items()
    }


def mae_3d(pred: np.ndarray, ref: np.ndarray) -> float:
    """
    Calculate mean absolute error between two 3D images.

    MAE measures the average absolute voxel-wise difference between the
    predicted/generated CT and the reference CT.

    Parameters
    ----------
    pred : np.ndarray
        Predicted or generated 3D image array.

    ref : np.ndarray
        Reference 3D image array.

    Returns
    -------
    mae : float
        Mean absolute error between `pred` and `ref`.

    Raises
    ------
    ValueError
        If `pred` or `ref` is empty.
        If `pred` and `ref` have different shapes.
        If `pred` or `ref` is not 3-dimensional.
        If `pred` or `ref` contains NaN or infinite values.
    """

    if pred.size == 0:
        raise ValueError("`pred` cannot be empty")

    if ref.size == 0:
        raise ValueError("`ref` cannot be empty")

    if pred.ndim != 3:
        raise ValueError(f"`pred` must be a 3D array. Got shape {pred.shape}")

    if ref.ndim != 3:
        raise ValueError(f"`ref` must be a 3D array. Got shape {ref.shape}")

    if pred.shape != ref.shape:
        raise ValueError(f"`pred` and `ref` must have the same shape. Got {pred.shape} and {ref.shape}")

    if not np.isfinite(pred).all():
        raise ValueError("`pred` contains NaN or infinite values")

    if not np.isfinite(ref).all():
        raise ValueError("`ref` contains NaN or infinite values")

    return float(np.mean(np.abs(pred - ref)))


def ssim_3d(pred: np.ndarray, ref: np.ndarray, data_min: float, data_max: float) -> float:
    """
    Calculate the mean Structural Similarity Index (SSIM) for two 3D images.

    This function computes SSIM slice-by-slice along the z-axis and returns
    the average score across all slices. It is intended for comparing
    volumetric medical images such as CT scans while using the standard
    2D SSIM implementation.

    Parameters
    ----------
    pred : np.ndarray
        Predicted 3D image array.

    ref : np.ndarray
        Reference (ground-truth) 3D image array.

    Returns
    -------
    score : float
        Mean SSIM score averaged across all z-axis slices.
        Values closer to 1 indicate higher structural similarity.

    data_min: float
        Minimum value allowed for the data.

    data_max: float
        Maximum value allowed for the data.

    Raises
    ------
    ValueError
        If `pred` or `ref` is empty.
        If `pred` and `ref` have different shapes.
        If `pred` or `ref` is not 3-dimensional.
        If `pred` or `ref` contains NaN or infinite values.
        If `data_min` is greater than `data_max`.
        If data falls outside of [data_min, data_max].
    """

    if pred.size == 0:
        raise ValueError("`pred` cannot be empty")

    if ref.size == 0:
        raise ValueError("`ref` cannot be empty")

    if pred.ndim != 3:
        raise ValueError(f"`pred` must be a 3D array. Got shape {pred.shape}")

    if ref.ndim != 3:
        raise ValueError(f"`ref` must be a 3D array. Got shape {ref.shape}")

    if pred.shape != ref.shape:
        raise ValueError(f"`pred` and `ref` must have the same shape. Got {pred.shape} and {ref.shape}")

    if not np.isfinite(pred).all():
        raise ValueError("`pred` contains NaN or infinite values")

    if not np.isfinite(ref).all():
        raise ValueError("`ref` contains NaN or infinite values")

    if data_min >= data_max:
        raise ValueError(f"`data_min` must be smaller than `data_max`. Got data_min={data_min}, data_max={data_max}")

    if pred.min() < data_min or pred.max() > data_max:
        raise ValueError(
            f"`pred` values must be in the range [data_min, data_max], got min={pred.min()}, max={pred.max()}"
        )

    if ref.min() < data_min or ref.max() > data_max:
        raise ValueError(
            f"`ref` values must be in the range [data_min, data_max], got min={ref.min()}, max={ref.max()}"
        )

    scores = []

    for z in range(pred.shape[-1]):
        pred_slice = pred[:, :, z]
        ref_slice = ref[:, :, z]

        score = cast(
            float,
            ssim(  # type: ignore[no-untyped-call]
                ref_slice,
                pred_slice,
                data_range=data_max - data_min,
            ),
        )
        scores.append(score)

    return float(np.mean(scores))


def sobel_edge_map_3d(arr: np.ndarray) -> FloatArray:
    """
    Calculate a 3D Sobel edge magnitude map.

    This function applies the Sobel operator along all three spatial axes and
    combines the resulting gradients into one edge-magnitude image.

    It can be used for SOB / Sobel-based edge similarity, where the goal is to
    compare whether two CT images have similar anatomical edge structure.

    Parameters
    ----------
    arr : np.ndarray
        Input 3D image array.

    Returns
    -------
    edge : np.ndarray
        3D Sobel edge magnitude map as a float32 NumPy array.

    Raises
    ------
    ValueError
        If `arr` is empty.
        If `arr` is not 3-dimensional.
        If `arr` contains NaN or infinite values.
    """

    if arr.size == 0:
        raise ValueError("`arr` cannot be empty")

    if arr.ndim != 3:
        raise ValueError(f"`arr` must be a 3D array with shape [H, W, Z]. Got shape {arr.shape}")

    if not np.isfinite(arr).all():
        raise ValueError("`arr` contains NaN or infinite values")

    sx = sobel(arr, axis=0)
    sy = sobel(arr, axis=1)
    sz = sobel(arr, axis=2)

    edge = np.sqrt(sx**2 + sy**2 + sz**2)

    return cast(FloatArray, edge.astype(np.float32))


def sob_3d(pred: np.ndarray, ref: np.ndarray) -> float:
    """
    Calculate SOB / Sobel-based edge difference between two 3D images.

    This metric computes Sobel edge maps for the predicted/generated image
    and the reference image, then returns the mean absolute difference between
    those edge maps.

    Lower values mean the generated image has a more similar edge or structure
    pattern to the reference CT.

    Parameters
    ----------
    pred : np.ndarray
        Predicted or generated 3D image array.

    ref : np.ndarray
        Reference 3D image array.

    Returns
    -------
    sob : float
        Mean absolute difference between Sobel edge maps.

    Raises
    ------
    ValueError
        If `pred` or `ref` is empty.
        If `pred` or `ref` is not 3-dimensional.
        If `pred` and `ref` have different shapes.
        If `pred` or `ref` contains NaN or infinite values.
    """

    if pred.size == 0:
        raise ValueError("`pred` cannot be empty")

    if ref.size == 0:
        raise ValueError("`ref` cannot be empty")

    if pred.ndim != 3:
        raise ValueError(f"`pred` must be a 3D array. Got shape {pred.shape}")

    if ref.ndim != 3:
        raise ValueError(f"`ref` must be a 3D array. Got shape {ref.shape}")

    if pred.shape != ref.shape:
        raise ValueError(f"`pred` and `ref` must have the same shape. Got {pred.shape} and {ref.shape}")

    if not np.isfinite(pred).all():
        raise ValueError("`pred` contains NaN or infinite values")

    if not np.isfinite(ref).all():
        raise ValueError("`ref` contains NaN or infinite values")

    pred_edge = sobel_edge_map_3d(pred)
    ref_edge = sobel_edge_map_3d(ref)

    return float(np.mean(np.abs(pred_edge - ref_edge)))


def _evenly_spaced_slices_for_lpips(
    arr: np.ndarray,
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
    arr : np.ndarray
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

    if arr.size == 0:
        raise ValueError("`arr` cannot be empty")

    if arr.ndim != 3:
        raise ValueError(f"`arr` must be a 3D array with shape [H, W, Z]. Got shape {arr.shape}")

    if not np.isfinite(arr).all():
        raise ValueError("`arr` contains NaN or infinite values")

    if arr.min() < 0.0 or arr.max() > 1.0:
        raise ValueError("`arr` must be normalized to [0, 1] before LPIPS calculation")

    if max_slices < 1:
        raise ValueError("`max_slices` must be at least 1")

    z_dim = arr.shape[-1]

    if z_dim <= max_slices:
        slice_ids = np.arange(z_dim)
    else:
        slice_ids = np.linspace(
            0,
            z_dim - 1,
            num=max_slices,
            dtype=int,
        )

    slices = arr[:, :, slice_ids]  # H, W, Z
    slices = np.moveaxis(slices, -1, 0)  # Z, H, W

    tensor = torch.from_numpy(slices).float()
    tensor = tensor.unsqueeze(1)  # Z, 1, H, W
    tensor = tensor.repeat(1, 3, 1, 1)  # Z, 3, H, W

    tensor = tensor * 2.0 - 1.0

    return tensor


def lpips_3d(
    config: MaisiTestingConfig,
    lpips_model: LPIPSModel,
    pred: np.ndarray,
    ref: np.ndarray,
) -> float:
    """
    Calculate slice-wise LPIPS between two 3D images and average the result.

    LPIPS is normally defined for 2D RGB images. For 3D CT volumes, this
    function selects evenly spaced slices across the full volume, converts them
    to LPIPS input format, computes LPIPS slice-wise, and returns the mean score.

    Using evenly spaced slices makes the metric less sensitive to one noisy or
    unrepresentative center slice.

    Interpretation:
    - Higher LPIPS means greater perceptual difference.
    - When comparing generated variants with each other, higher pairwise LPIPS
    usually suggests higher visual diversity.
    - When comparing generated CTs to reference CTs, lower LPIPS means higher
    perceptual similarity.

    Parameters
    ----------
    config : MaisiTestingConfig
         Configuration object containing LPIPS settings.

    lpips_model : LPIPSModel
        Initialized LPIPS model.

    pred : np.ndarray
        Predicted or generated 3D CT array normalized to [data_min, data_max].

    ref : np.ndarray
        Reference 3D CT array normalized to [data_min, data_max].

    Returns
    -------
    lpips_score : float
        Mean LPIPS score across selected slices.

    Raises
    ------
    ValueError
        If `pred` or `ref` is empty.
        If `pred` or `ref` is not 3-dimensional.
        If `pred` and `ref` have different shapes.
        If `pred` or `ref` contains NaN or infinite values.
        If `pred` or `ref` is not normalized to [data_min, data_max].
        If `max_slices` is less than 1.
    """

    if pred.size == 0:
        raise ValueError("`pred` cannot be empty")

    if ref.size == 0:
        raise ValueError("`ref` cannot be empty")

    if pred.ndim != 3:
        raise ValueError(f"`pred` must be a 3D array. Got shape {pred.shape}")

    if ref.ndim != 3:
        raise ValueError(f"`ref` must be a 3D array. Got shape {ref.shape}")

    if pred.shape != ref.shape:
        raise ValueError(f"`pred` and `ref` must have the same shape. Got {pred.shape} and {ref.shape}")

    if not np.isfinite(pred).all():
        raise ValueError("`pred` contains NaN or infinite values")

    if not np.isfinite(ref).all():
        raise ValueError("`ref` contains NaN or infinite values")

    if pred.min() < config.data_min or pred.max() > config.data_max:
        raise ValueError("`pred` must be normalized to [data_min, data_max]")

    if ref.min() < config.data_min or ref.max() > config.data_max:
        raise ValueError("`ref` must be normalized to [data_min, data_max]")

    pred_tensor = _evenly_spaced_slices_for_lpips(
        (pred - config.data_min) / (config.data_max - config.data_min),
        max_slices=config.max_lpips_slices,
    ).to(config.device)

    ref_tensor = _evenly_spaced_slices_for_lpips(
        (ref - config.data_min) / (config.data_max - config.data_min),
        max_slices=config.max_lpips_slices,
    ).to(config.device)

    with torch.no_grad():
        score = lpips_model(pred_tensor, ref_tensor)

    return float(score.mean().item())


def collect_original_cts(original_ct_dir: Path) -> dict[str, list[Path]]:
    """
    Collect available original/reference CT tensors grouped by patient ID.

    This function searches recursively for `.pt` files inside
    `original_ct_dir`, extracts the patient ID from each file path, and groups
    all available original/reference CT tensors by patient.

    It works for both cases:
    - only planning CTs are available
    - multiple real test fractions are available per patient

    Parameters
    ----------
    original_ct_dir : Path
        Directory containing original/reference CT tensors saved as `.pt`
        files.

    Returns
    -------
    originals : dict[str, list[Path]]
        Dictionary mapping each patient ID to a list of CT tensor paths.

        Example:
        {
            "Patient_1": [
                Path(".../Patient_1_fraction_1_.pt"),
                Path(".../Patient_1_fraction_2_.pt"),
            ],
            "Patient_2": [
                Path(".../Patient_2_fraction_1_.pt"),
            ],
        }

    Raises
    ------
    FileNotFoundError
        If `original_ct_dir` does not exist.

    ValueError
        If no `.pt` files are found in `original_ct_dir`.
    """

    if not original_ct_dir.exists():
        raise FileNotFoundError(f"Original CT directory does not exist: {original_ct_dir}")

    paths = sorted(
        glob.glob(
            str(original_ct_dir / "**" / "*.pt"),
            recursive=True,
        )
    )

    if len(paths) == 0:
        raise ValueError(f"No original/reference CT `.pt` files found in: {original_ct_dir}")

    originals: dict[str, list[Path]] = {}

    for path in paths:
        patient_id = _extract_patient_id(path)
        originals.setdefault(patient_id, []).append(Path(path))

    return originals


def collect_generated_cts(generated_ct_dir: Path) -> dict[str, list[Path]]:
    """
    Collect generated CT variants grouped by patient ID.

    This function searches recursively for generated `.pt` CT files inside
    `generated_ct_dir`, extracts the patient ID from each path, and groups
    generated variants by patient.

    Missing patients or missing variants are allowed. The function only
    collects files that actually exist.

    Expected structure:
        generated_ct_dir/
            Patient_1/
                Patient_1_gen_1.pt
                Patient_1_gen_2.pt
            Patient_2/
                Patient_2_gen_1.pt

    Parameters
    ----------
    generated_ct_dir : Path
        Directory containing generated CT tensors saved as `.pt` files.

    Returns
    -------
    generated : dict[str, list[Path]]
        Dictionary mapping each patient ID to a list of generated CT paths.

        Example:
        {
            "Patient_1": [
                Path(".../Patient_1_gen_1.pt"),
                Path(".../Patient_1_gen_2.pt"),
            ],
            "Patient_2": [
                Path(".../Patient_2_gen_1.pt"),
            ],
        }

    Raises
    ------
    FileNotFoundError
        If `generated_ct_dir` does not exist.

    ValueError
        If no generated CT `.pt` files are found in `generated_ct_dir`.
    """

    if not generated_ct_dir.exists():
        raise FileNotFoundError(f"Generated CT directory does not exist: {generated_ct_dir}")

    paths = sorted(
        glob.glob(
            str(generated_ct_dir / "**" / "*.pt"),
            recursive=True,
        )
    )

    if len(paths) == 0:
        raise ValueError(f"No generated CT `.pt` files found in: {generated_ct_dir}")

    generated: dict[str, list[Path]] = {}

    for path in paths:
        patient_id = _extract_patient_id(path)
        generated.setdefault(patient_id, []).append(Path(path))

    return generated


def calculate_similarity_metrics(
    config: MaisiTestingConfig,
) -> None:
    """
    Calculate generated-vs-real CT similarity metrics.

    This function compares generated CT variants with available non-planning
    original/reference CT tensors for the same patient. It is robust to missing
    generated variants and calculates only comparisons that are possible.

    Metrics:
    - MAE: lower is better
    - SSIM: higher is better
    - SOB/Sobel MAE: lower is better
    - LPIPS: higher means greater perceptual difference

    Outputs:
    - generated_vs_real_metrics.csv
        Per-comparison metrics.
    - generated_vs_real_summary.csv
        Patient-level summary statistics.

    Parameters
    ----------
    config : MaisiTestingConfig
        Configuration object containing evaluation settings and directory paths.

    Raises
    ------
    ValueError
        If no valid comparisons can be calculated.
    """

    generated = collect_generated_cts(config.generated_ct_dir)
    originals = collect_original_cts(config.processed_ct_dir)

    lpips_model = None

    if config.use_lpips:
        import lpips  # type: ignore[import-untyped]

        lpips_model = lpips.LPIPS(net=config.lpips_net).to(config.device)
        lpips_model.eval()

    rows = []

    for patient_id, gen_paths in tqdm(
        generated.items(),
        desc="Generated vs real metrics",
    ):
        all_ref_paths = originals.get(patient_id, [])
        ref_paths = _exclude_planning_cts({patient_id: all_ref_paths})[patient_id]

        if len(all_ref_paths) == 0:
            print(f"Skipping {patient_id}: no original/reference CT found")
            continue

        if len(ref_paths) == 0:
            print(f"Skipping {patient_id}: no non-planning original/reference CT found")
            continue

        for gen_path in gen_paths:
            gen_arr = _to_numpy_hu(
                _load_tensor(gen_path),
                data_min=config.data_min,
                data_max=config.data_max,
            )

            for ref_path in ref_paths:
                ref_arr = _to_numpy_hu(
                    _load_tensor(ref_path),
                    data_min=config.data_min,
                    data_max=config.data_max,
                )

                if gen_arr.shape != ref_arr.shape:
                    print(
                        f"Skipping shape mismatch for {patient_id}: "
                        f"{gen_path.name} {gen_arr.shape} vs "
                        f"{ref_path.name} {ref_arr.shape}"
                    )
                    continue

                row = {
                    "patient_id": patient_id,
                    "comparison_type": "generated_vs_real",
                    "generated_path": str(gen_path),
                    "reference_path": str(ref_path),
                    "mae": mae_3d(gen_arr, ref_arr),
                    "ssim": ssim_3d(gen_arr, ref_arr, config.data_min, config.data_max),
                    "sob": sob_3d(gen_arr, ref_arr),
                }

                if config.use_lpips and lpips_model is not None:
                    row["lpips"] = lpips_3d(
                        config=config,
                        lpips_model=lpips_model,
                        pred=gen_arr,
                        ref=ref_arr,
                    )
                else:
                    row["lpips"] = np.nan

                rows.append(row)

    if len(rows) == 0:
        raise ValueError(
            "No valid generated-vs-real comparisons were calculated. "
            "Check whether patient IDs match and tensor shapes are compatible"
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        config.metrics_dir / "generated_vs_real_metrics.csv",
        index=False,
    )

    summary = df.groupby("patient_id")[["mae", "ssim", "sob", "lpips"]].agg(["mean", "std", "min", "max", "count"])

    summary.to_csv(
        config.metrics_dir / "generated_vs_real_summary.csv",
    )


def calculate_pairwise_variety_metrics(
    ct_groups: dict[str, list[Path]],
    config: MaisiTestingConfig,
    output_filename: str,
    comparison_type: str,
) -> None:
    """
    Calculate pairwise variety metrics within groups of CT images.

    This function compares all possible image pairs within each patient group.
    It is used to estimate within-group variety, for example:
    - real_vs_real: real fraction vs real fraction
    - generated_vs_generated: generated variant vs generated variant

    Metrics:
    - MAE: lower means more similar voxel intensities
    - SSIM: higher means more similar structure
    - SOB: lower means more similar edge structure
    - LPIPS: higher usually means more visual/perceptual difference

    Parameters
    ----------
    ct_groups : dict[str, list[Path]]
        Dictionary mapping patient IDs to CT tensor paths.

    config : MaisiTestingConfig
        Configuration object containing evaluation settings.

    output_filename : str
        Name of the per-comparison output CSV file.

    comparison_type : str
        Label describing the comparison type, e.g.:
        "generated_vs_generated" or "real_vs_real".

    Raises
    ------
    ValueError
        If `ct_groups` is empty.
        If `output_filename` does not end with ".csv".
        If `max_lpips_slices` is less than 1.
        If the inputs are invalid. If no valid pairwise comparisons can be
        calculated, the metric file is skipped.
    """

    if len(ct_groups) == 0:
        raise ValueError("`ct_groups` cannot be empty")

    lpips_model = None

    if config.use_lpips:
        try:
            import lpips

            lpips_model = lpips.LPIPS(net=config.lpips_net).to(config.device)
            lpips_model.eval()
        except ImportError:
            print("LPIPS package not installed. Skipping LPIPS")
            config.use_lpips = False

    rows = []

    for patient_id, paths in tqdm(ct_groups.items(), desc=comparison_type):
        if len(paths) < 2:
            print(f"Skipping pairwise {comparison_type} for {patient_id}: only {len(paths)} image(s).")
            continue

        for path_a, path_b in itertools.combinations(paths, 2):
            arr_a = _to_numpy_hu(
                _load_tensor(path_a),
                data_min=config.data_min,
                data_max=config.data_max,
            )
            arr_b = _to_numpy_hu(
                _load_tensor(path_b),
                data_min=config.data_min,
                data_max=config.data_max,
            )

            if arr_a.shape != arr_b.shape:
                print(
                    f"Skipping shape mismatch for {patient_id}: "
                    f"{path_a.name} {arr_a.shape} vs "
                    f"{path_b.name} {arr_b.shape}"
                )
                continue

            row = {
                "patient_id": patient_id,
                "comparison_type": comparison_type,
                "path_a": str(path_a),
                "path_b": str(path_b),
                "mae": mae_3d(arr_a, arr_b),
                "ssim": ssim_3d(arr_a, arr_b, config.data_min, config.data_max),
                "sob": sob_3d(arr_a, arr_b),
            }

            if config.use_lpips and lpips_model is not None:
                row["lpips"] = lpips_3d(
                    pred=arr_a,
                    ref=arr_b,
                    lpips_model=lpips_model,
                    config=config,
                )
            else:
                row["lpips"] = np.nan

            rows.append(row)

    if len(rows) == 0:
        print(f"Skipping {comparison_type}: no valid pairwise comparisons were calculated")
        return

    df = pd.DataFrame(rows)

    df.to_csv(
        config.metrics_dir / output_filename,
        index=False,
    )

    summary = df.groupby("patient_id")[["mae", "ssim", "sob", "lpips"]].agg(["mean", "std", "min", "max", "count"])

    summary_name = output_filename.replace(".csv", "_summary.csv")
    summary.to_csv(config.metrics_dir / summary_name)


def evaluate_generated_cts(
    config: MaisiTestingConfig,
) -> None:
    """
    Run full CT generation evaluation.

    This function produces up to three metric files:

    1. generated_vs_real_metrics.csv
       Generated CTs compared to original/reference CTs.

    2. generated_pairwise_variety_metrics.csv
       Generated CT variants compared with each other.

    3. real_pairwise_variety_metrics.csv
       Real/reference CTs compared with each other.

    The goal is to check:
    - whether generated CTs are similar to real CTs
    - whether generated variety is comparable to real anatomical variety

    Parameters
    ----------
    config : MaisiTestingConfig
        Configuration object containing evaluation settings.

    use_lpips : bool
        Whether to calculate LPIPS.

    Raises
    ------
    FileNotFoundError
        If generated or original CT directories do not exist.

    ValueError
        If no valid generated-vs-real comparisons can be calculated.
    """

    generated = collect_generated_cts(config.generated_ct_dir)
    originals = collect_original_cts(config.processed_ct_dir)
    originals_without_planning = _exclude_planning_cts(originals)

    calculate_similarity_metrics(
        config=config,
    )

    calculate_pairwise_variety_metrics(
        ct_groups=generated,
        config=config,
        output_filename="generated_pairwise_variety_metrics.csv",
        comparison_type="generated_vs_generated",
    )

    calculate_pairwise_variety_metrics(
        ct_groups=originals_without_planning,
        config=config,
        output_filename="real_pairwise_variety_metrics.csv",
        comparison_type="real_vs_real",
    )
