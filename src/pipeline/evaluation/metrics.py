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


def _to_numpy_01(tensor: torch.Tensor) -> FloatArray:
    """
    Convert a CT tensor to a NumPy array normalized to the range [0, 1].

    This helper supports two expected intensity formats:
    - already preprocessed tensors in the range [0, 1]
    - HU-like tensors in the range approximately [-1000, 1000]

    If the tensor is already within [0, 1], it is returned unchanged as a
    float32 NumPy array. Otherwise, values are clipped to [-1000, 1000] and
    linearly rescaled to [0, 1].

    Parameters
    ----------
    tensor : torch.Tensor
        Input CT tensor.

    Returns
    -------
    arr : np.ndarray
        CT image as a float32 NumPy array normalized to [0, 1].

    Raises
    ------
    TypeError
        If `tensor` is not a torch.Tensor.

    ValueError
        If `tensor` is empty.
        If `tensor` contains NaN or infinite values.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"`tensor` must be a torch.Tensor. Got {type(tensor)}")

    if tensor.numel() == 0:
        raise ValueError("`tensor` cannot be empty")

    if not torch.isfinite(tensor).all():
        raise ValueError("`tensor` contains NaN or infinite values")

    arr = tensor.detach().cpu().numpy().astype(np.float32)

    # Case 1: already preprocessed to [0, 1]
    if arr.min() >= 0.0 and arr.max() <= 1.0:
        return arr

    # Case 2: HU-like range [-1000, 1000]
    arr = np.clip(arr, -1000, 1000).astype(np.float32)
    arr = ((arr + 1000) / 2000).astype(np.float32)

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
    TypeError
        If `path` is not a string or Path object.

    ValueError
        If `path` is empty.
        If patient ID cannot be extracted from the filename or parent folder.
    """

    if not isinstance(path, str | Path):
        raise TypeError(f"`path` must be a string or Path object. Got {type(path)}")

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
    TypeError
        If `pred` or `ref` is not a NumPy array.

    ValueError
        If `pred` or `ref` is empty.
        If `pred` and `ref` have different shapes.
        If `pred` or `ref` is not 3-dimensional.
        If `pred` or `ref` contains NaN or infinite values.
    """

    if not isinstance(pred, np.ndarray):
        raise TypeError(f"`pred` must be a NumPy array. Got {type(pred)}")

    if not isinstance(ref, np.ndarray):
        raise TypeError(f"`ref` must be a NumPy array. Got {type(ref)}")

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


def ssim_3d(pred: np.ndarray, ref: np.ndarray) -> float:
    """
    Slice-wise SSIM averaged over the z dimension.
    Assumes arrays are normalized to [0, 1].
    """

    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, ref={ref.shape}")

    scores = []

    for z in range(pred.shape[-1]):
        pred_slice = pred[:, :, z]
        ref_slice = ref[:, :, z]

        score = cast(
            float,
            ssim(  # type: ignore[no-untyped-call]
                ref_slice,
                pred_slice,
                data_range=1.0,
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
    TypeError
        If `arr` is not a NumPy array.

    ValueError
        If `arr` is empty.
        If `arr` is not 3-dimensional.
        If `arr` contains NaN or infinite values.
    """

    if not isinstance(arr, np.ndarray):
        raise TypeError(f"`arr` must be a NumPy array. Got {type(arr)}")

    if arr.size == 0:
        raise ValueError("`arr` cannot be empty")

    if arr.ndim != 3:
        raise ValueError(f"`arr` must be a 3D array. Got shape {arr.shape}")

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
    TypeError
        If `pred` or `ref` is not a NumPy array.

    ValueError
        If `pred` or `ref` is empty.
        If `pred` or `ref` is not 3-dimensional.
        If `pred` and `ref` have different shapes.
        If `pred` or `ref` contains NaN or infinite values.
    """

    if not isinstance(pred, np.ndarray):
        raise TypeError(f"`pred` must be a NumPy array. Got {type(pred)}.")

    if not isinstance(ref, np.ndarray):
        raise TypeError(f"`ref` must be a NumPy array. Got {type(ref)}.")

    if pred.size == 0:
        raise ValueError("`pred` cannot be empty.")

    if ref.size == 0:
        raise ValueError("`ref` cannot be empty.")

    if pred.ndim != 3:
        raise ValueError(f"`pred` must be a 3D array. Got shape {pred.shape}.")

    if ref.ndim != 3:
        raise ValueError(f"`ref` must be a 3D array. Got shape {ref.shape}.")

    if pred.shape != ref.shape:
        raise ValueError(f"`pred` and `ref` must have the same shape. Got {pred.shape} and {ref.shape}.")

    if not np.isfinite(pred).all():
        raise ValueError("`pred` contains NaN or infinite values.")

    if not np.isfinite(ref).all():
        raise ValueError("`ref` contains NaN or infinite values.")

    pred_edge = sobel_edge_map_3d(pred)
    ref_edge = sobel_edge_map_3d(ref)

    return float(np.mean(np.abs(pred_edge - ref_edge)))


def _center_slices_for_lpips(
    arr: np.ndarray,
    max_slices: int = 16,
) -> torch.Tensor:
    """
    Convert selected center 2D CT slices to LPIPS input format.

    LPIPS expects 2D RGB-like images, so this function:
    - selects up to `max_slices` slices from the center of the 3D volume
    - converts slices from shape [H, W, Z] to [Z, H, W]
    - adds a channel dimension
    - repeats the single CT channel into 3 channels
    - rescales values from [0, 1] to [-1, 1]

    Parameters
    ----------
    arr : np.ndarray
        Input 3D CT array normalized to [0, 1], expected shape [H, W, Z].

    max_slices : int
        Maximum number of center slices used for LPIPS calculation.

    Returns
    -------
    tensor : torch.Tensor
        Tensor formatted for LPIPS with shape [N, 3, H, W] and range [-1, 1],
        where N is the number of selected slices.

    Raises
    ------
    TypeError
        If `arr` is not a NumPy array.

    ValueError
        If `arr` is empty.
        If `arr` is not 3-dimensional.
        If `arr` contains NaN or infinite values.
        If `arr` is not normalized to [0, 1].
        If `max_slices` is less than 1.
    """

    if not isinstance(arr, np.ndarray):
        raise TypeError(f"`arr` must be a NumPy array. Got {type(arr)}")

    if arr.size == 0:
        raise ValueError("`arr` cannot be empty.")

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
        slice_ids = list(range(z_dim))
    else:
        center = z_dim // 2
        half = max_slices // 2
        slice_ids = list(range(center - half, center + half))

    slices = arr[:, :, slice_ids]  # H, W, Z
    slices = np.moveaxis(slices, -1, 0)  # Z, H, W

    tensor = torch.from_numpy(slices).float()
    tensor = tensor.unsqueeze(1)  # Z, 1, H, W
    tensor = tensor.repeat(1, 3, 1, 1)  # Z, 3, H, W

    tensor = tensor * 2.0 - 1.0

    return tensor


def lpips_3d(
    pred: np.ndarray,
    ref: np.ndarray,
    lpips_model: LPIPSModel,
    device: torch.device,
    max_slices: int = 16,
) -> float:
    """
    Calculate slice-wise LPIPS between two 3D images and average the result.

    LPIPS is normally defined for 2D RGB images. For 3D CT volumes, this
    function selects center slices from each volume, converts them to LPIPS
    input format, computes LPIPS slice-wise, and returns the mean score.

    Interpretation:
    - Higher LPIPS means greater perceptual difference.
    - When comparing generated variants with each other, higher pairwise LPIPS
      usually suggests higher visual diversity.
    - When comparing generated CTs to reference CTs, lower LPIPS means higher
      perceptual similarity.

    Parameters
    ----------
    pred : np.ndarray
        Predicted or generated 3D CT array normalized to [0, 1].

    ref : np.ndarray
        Reference 3D CT array normalized to [0, 1].

    lpips_model
        Initialized LPIPS model.

    device : torch.device
        Device on which LPIPS should be calculated.

    max_slices : int
        Maximum number of center slices used for LPIPS calculation.

    Returns
    -------
    lpips_score : float
        Mean LPIPS score across selected slices.

    Raises
    ------
    TypeError
        If `pred` or `ref` is not a NumPy array.
        If `device` is not a torch.device.

    ValueError
        If `pred` or `ref` is empty.
        If `pred` or `ref` is not 3-dimensional.
        If `pred` and `ref` have different shapes.
        If `pred` or `ref` contains NaN or infinite values.
        If `pred` or `ref` is not normalized to [0, 1].
        If `max_slices` is less than 1.
    """

    if not isinstance(pred, np.ndarray):
        raise TypeError(f"`pred` must be a NumPy array. Got {type(pred)}")

    if not isinstance(ref, np.ndarray):
        raise TypeError(f"`ref` must be a NumPy array. Got {type(ref)}")

    if not isinstance(device, torch.device):
        raise TypeError(f"`device` must be a torch.device. Got {type(device)}")

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

    if pred.min() < 0.0 or pred.max() > 1.0:
        raise ValueError("`pred` must be normalized to [0, 1]")

    if ref.min() < 0.0 or ref.max() > 1.0:
        raise ValueError("`ref` must be normalized to [0, 1]")

    if max_slices < 1:
        raise ValueError("`max_slices` must be at least 1")

    pred_tensor = _center_slices_for_lpips(
        pred,
        max_slices=max_slices,
    ).to(device)

    ref_tensor = _center_slices_for_lpips(
        ref,
        max_slices=max_slices,
    ).to(device)

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
    TypeError
        If `original_ct_dir` is not a Path object.

    FileNotFoundError
        If `original_ct_dir` does not exist.

    ValueError
        If no `.pt` files are found in `original_ct_dir`.
    """

    if not isinstance(original_ct_dir, Path):
        raise TypeError(f"`original_ct_dir` must be a Path object. Got {type(original_ct_dir)}")

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
    TypeError
        If `generated_ct_dir` is not a Path object.

    FileNotFoundError
        If `generated_ct_dir` does not exist.

    ValueError
        If no generated CT `.pt` files are found in `generated_ct_dir`.
    """

    if not isinstance(generated_ct_dir, Path):
        raise TypeError(f"`generated_ct_dir` must be a Path object. Got {type(generated_ct_dir)}")

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
    generated_ct_dir: Path,
    original_ct_dir: Path,
    metrics_dir: Path,
    use_lpips: bool = True,
    lpips_net: str = "alex",
    max_lpips_slices: int = 16,
    device: str = "cuda",
) -> None:
    """
    Calculate generated-vs-real CT similarity metrics.

    This function compares generated CT variants with available original or
    reference CT tensors for the same patient. It is robust to missing generated
    variants and calculates only comparisons that are possible.

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
    generated_ct_dir : Path
        Directory containing generated CT tensors.

    original_ct_dir : Path
        Directory containing original/reference CT tensors.

    metrics_dir : Path
        Directory where metric CSV files will be saved.

    use_lpips : bool
        Whether to calculate LPIPS.

    lpips_net : str
        LPIPS backbone name, for example "alex", "vgg", or "squeeze".

    max_lpips_slices : int
        Maximum number of center slices used for LPIPS.

    device : str
        Device used for LPIPS calculation. Usually "cuda" or "cpu".

    Raises
    ------
    TypeError
        If path arguments are not Path objects.
        If `use_lpips` is not bool.
        If `lpips_net` or `device` is not a string.

    ValueError
        If `max_lpips_slices` is less than 1.
        If no valid comparisons can be calculated.
    """

    if not isinstance(generated_ct_dir, Path):
        raise TypeError(f"`generated_ct_dir` must be a Path object. Got {type(generated_ct_dir)}")

    if not isinstance(original_ct_dir, Path):
        raise TypeError(f"`original_ct_dir` must be a Path object. Got {type(original_ct_dir)}")

    if not isinstance(metrics_dir, Path):
        raise TypeError(f"`metrics_dir` must be a Path object. Got {type(metrics_dir)}")

    if not isinstance(use_lpips, bool):
        raise TypeError(f"`use_lpips` must be bool. Got {type(use_lpips)}")

    if not isinstance(lpips_net, str):
        raise TypeError(f"`lpips_net` must be str. Got {type(lpips_net)}")

    if not isinstance(device, str):
        raise TypeError(f"`device` must be str. Got {type(device)}")

    if max_lpips_slices < 1:
        raise ValueError("`max_lpips_slices` must be at least 1")

    metrics_dir.mkdir(parents=True, exist_ok=True)

    generated = collect_generated_cts(generated_ct_dir)
    originals = collect_original_cts(original_ct_dir)

    torch_device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")

    lpips_model = None

    if use_lpips:
        try:
            import lpips  # type: ignore

            lpips_model = lpips.LPIPS(net=lpips_net).to(torch_device)
            lpips_model.eval()

        except ImportError:
            print("LPIPS package not installed. Skipping LPIPS")
            use_lpips = False

    rows = []

    for patient_id, gen_paths in tqdm(
        generated.items(),
        desc="Generated vs real metrics",
    ):
        ref_paths = originals.get(patient_id, [])

        if len(ref_paths) == 0:
            print(f"Skipping {patient_id}: no original/reference CT found")
            continue

        for gen_path in gen_paths:
            gen_arr = _to_numpy_01(_load_tensor(gen_path))

            for ref_path in ref_paths:
                ref_arr = _to_numpy_01(_load_tensor(ref_path))

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
                    "ssim": ssim_3d(gen_arr, ref_arr),
                    "sob": sob_3d(gen_arr, ref_arr),
                }

                if use_lpips and lpips_model is not None:
                    row["lpips"] = lpips_3d(
                        pred=gen_arr,
                        ref=ref_arr,
                        lpips_model=lpips_model,
                        device=torch_device,
                        max_slices=max_lpips_slices,
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
        metrics_dir / "generated_vs_real_metrics.csv",
        index=False,
    )

    summary = df.groupby("patient_id")[["mae", "ssim", "sob", "lpips"]].agg(["mean", "std", "min", "max", "count"])

    summary.to_csv(
        metrics_dir / "generated_vs_real_summary.csv",
    )


def calculate_pairwise_variety_metrics(
    ct_groups: dict[str, list[Path]],
    metrics_dir: Path,
    output_filename: str,
    comparison_type: str,
    use_lpips: bool = True,
    lpips_net: str = "alex",
    max_lpips_slices: int = 16,
    device: str = "cuda",
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

    metrics_dir : Path
        Directory where metric CSV files will be saved.

    output_filename : str
        Name of the per-comparison output CSV file.

    comparison_type : str
        Label describing the comparison type, e.g.:
        "generated_vs_generated" or "real_vs_real".

    use_lpips : bool
        Whether to calculate LPIPS.

    lpips_net : str
        LPIPS backbone name, e.g. "alex", "vgg", or "squeeze".

    max_lpips_slices : int
        Maximum number of center slices used for LPIPS.

    device : str
        Device used for LPIPS calculation. Usually "cuda" or "cpu".

    Raises
    ------
    TypeError
        If arguments have invalid types.

    ValueError
        If `ct_groups` is empty.
        If `output_filename` does not end with ".csv".
        If `max_lpips_slices` is less than 1.
        If no valid pairwise comparisons can be calculated.
    """

    if not isinstance(ct_groups, dict):
        raise TypeError(f"`ct_groups` must be a dictionary. Got {type(ct_groups)}")

    if not isinstance(metrics_dir, Path):
        raise TypeError(f"`metrics_dir` must be a Path. Got {type(metrics_dir)}")

    if not isinstance(output_filename, str):
        raise TypeError(f"`output_filename` must be str. Got {type(output_filename)}")

    if not isinstance(comparison_type, str):
        raise TypeError(f"`comparison_type` must be str. Got {type(comparison_type)}")

    if not isinstance(use_lpips, bool):
        raise TypeError(f"`use_lpips` must be bool. Got {type(use_lpips)}")

    if not isinstance(lpips_net, str):
        raise TypeError(f"`lpips_net` must be str. Got {type(lpips_net)}")

    if not isinstance(device, str):
        raise TypeError(f"`device` must be str. Got {type(device)}")

    if len(ct_groups) == 0:
        raise ValueError("`ct_groups` cannot be empty")

    if not output_filename.endswith(".csv"):
        raise ValueError("`output_filename` must end with '.csv'")

    if max_lpips_slices < 1:
        raise ValueError("`max_lpips_slices` must be at least 1")

    metrics_dir.mkdir(parents=True, exist_ok=True)

    torch_device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")

    lpips_model = None

    if use_lpips:
        try:
            import lpips

            lpips_model = lpips.LPIPS(net=lpips_net).to(torch_device)
            lpips_model.eval()
        except ImportError:
            print("LPIPS package not installed. Skipping LPIPS")
            use_lpips = False

    rows = []

    for patient_id, paths in tqdm(ct_groups.items(), desc=comparison_type):
        if len(paths) < 2:
            print(f"Skipping pairwise {comparison_type} for {patient_id}: only {len(paths)} image(s).")
            continue

        for path_a, path_b in itertools.combinations(paths, 2):
            arr_a = _to_numpy_01(_load_tensor(path_a))
            arr_b = _to_numpy_01(_load_tensor(path_b))

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
                "ssim": ssim_3d(arr_a, arr_b),
                "sob": sob_3d(arr_a, arr_b),
            }

            if use_lpips and lpips_model is not None:
                row["lpips"] = lpips_3d(
                    pred=arr_a,
                    ref=arr_b,
                    lpips_model=lpips_model,
                    device=torch_device,
                    max_slices=max_lpips_slices,
                )
            else:
                row["lpips"] = np.nan

            rows.append(row)

    if len(rows) == 0:
        raise ValueError(f"No valid pairwise comparisons were calculated for `{comparison_type}`.")

    df = pd.DataFrame(rows)

    df.to_csv(
        metrics_dir / output_filename,
        index=False,
    )

    summary = df.groupby("patient_id")[["mae", "ssim", "sob", "lpips"]].agg(["mean", "std", "min", "max", "count"])

    summary_name = output_filename.replace(".csv", "_summary.csv")
    summary.to_csv(metrics_dir / summary_name)


def evaluate_generated_cts(
    generated_ct_dir: Path,
    original_ct_dir: Path,
    metrics_dir: Path,
    use_lpips: bool = True,
    device: str = "cuda",
) -> None:
    """
    Run full CT generation evaluation.

    This function produces three metric files:

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
    generated_ct_dir : Path
        Directory containing generated CT tensors.

    original_ct_dir : Path
        Directory containing original/reference CT tensors.

    metrics_dir : Path
        Directory where metric CSV files will be saved.

    use_lpips : bool
        Whether to calculate LPIPS.

    device : str
        Device used for LPIPS calculation. Usually "cuda" or "cpu".

    Raises
    ------
    TypeError
        If path arguments are not Path objects.
        If `use_lpips` is not bool.
        If `device` is not str.

    FileNotFoundError
        If generated or original CT directories do not exist.

    ValueError
        If no valid comparisons can be calculated.
    """

    if not isinstance(generated_ct_dir, Path):
        raise TypeError(f"`generated_ct_dir` must be a Path. Got {type(generated_ct_dir)}")

    if not isinstance(original_ct_dir, Path):
        raise TypeError(f"`original_ct_dir` must be a Path. Got {type(original_ct_dir)}")

    if not isinstance(metrics_dir, Path):
        raise TypeError(f"`metrics_dir` must be a Path. Got {type(metrics_dir)}")

    if not isinstance(use_lpips, bool):
        raise TypeError(f"`use_lpips` must be bool. Got {type(use_lpips)}")

    if not isinstance(device, str):
        raise TypeError(f"`device` must be str. Got {type(device)}")

    metrics_dir.mkdir(parents=True, exist_ok=True)

    generated = collect_generated_cts(generated_ct_dir)
    originals = collect_original_cts(original_ct_dir)

    calculate_similarity_metrics(
        generated_ct_dir=generated_ct_dir,
        original_ct_dir=original_ct_dir,
        metrics_dir=metrics_dir,
        use_lpips=use_lpips,
        device=device,
    )

    calculate_pairwise_variety_metrics(
        ct_groups=generated,
        metrics_dir=metrics_dir,
        output_filename="generated_pairwise_variety_metrics.csv",
        comparison_type="generated_vs_generated",
        use_lpips=use_lpips,
        device=device,
    )

    calculate_pairwise_variety_metrics(
        ct_groups=originals,
        metrics_dir=metrics_dir,
        output_filename="real_pairwise_variety_metrics.csv",
        comparison_type="real_vs_real",
        use_lpips=use_lpips,
        device=device,
    )
