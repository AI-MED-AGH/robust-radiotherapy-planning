from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch


def create_data_split_dict(
    seed: int = 42,
    train_fraction: float = 0.8,
    folds: int = 5,
    data_root: Path | None = None,
    output_json: Path | None = None,
    blacklist: list[str] | None = None,
    save: bool = True,
) -> dict[str | int, Any]:
    """
    Create train/validation/test splits for longitudinal CT registration data.

    Assumptions:
    - This function lives in a Python file somewhere inside `src/`
    - Data is stored in `src/data_full/CT/Patient_x/...`
    - Each patient folder contains:
        Patient_<id>_fraction_1_.nii.gz
        Patient_<id>_fraction_2_.nii.gz
        ...
    - Fraction 1 is always the moving image
    - All other fractions are fixed images

    Returns
    -------
        data_dict : dict
            Dictionary with cross-validation folds and test set:
            {
                0: {"train": [...], "val": [...]},
                1: {"train": [...], "val": [...]},
                ...
                "test": [...]
            }

    Parameters
    ----------
        seed : int
            Random seed for shuffling patient IDs.
        train_fraction : float
            Fraction of patients used for train+validation pool.
            Remaining patients go to test set.
        folds : int
            Number of cross-validation folds.
        data_root : Path | None
            Path to CT data folder. If None, defaults to:
            <src>/data_full/CT
        output_json : Path | None
            Path to output JSON file. If None, defaults to:
            <src>/data_full/data_dict_encoder.json
        blacklist: list[str] | None
            A list of file names to not add to the output if encountered. Assumed to not include planning CTs.
            Defaults to a predefined list.
        save : bool
            Whether to save the resulting dictionary as JSON.

    Raises
    ------
        FileNotFoundError
            If the data directory does not exist.
        ValueError
            If parameters are invalid or no patients are found.
    """

    # Resolve paths relative to this file, assuming file is somewhere inside src/
    this_file = Path(__file__).resolve()

    # Find the src directory by walking upward
    src_root = this_file.parent
    while src_root.name != "src" and src_root.parent != src_root:
        src_root = src_root.parent

    if src_root.name != "src":
        raise FileNotFoundError("Could not locate the 'src' directory relative to this file")

    if data_root is None:
        data_root = src_root / "data_full" / "CT"

    if output_json is None:
        output_json = src_root / "data_full" / "data_dict.json"

    if blacklist is None:
        blacklist = [
            "Patient_20_fraction_14_.nii.gz",
            "Patient_63_fraction_12_.nii.gz",
            "Patient_63_fraction_13_.nii.gz",
            "Patient_63_fraction_14_.nii.gz",
            "Patient_63_fraction_15_.nii.gz",
            "Patient_63_fraction_21_.nii.gz",
            "Patient_63_fraction_22_.nii.gz",
            "Patient_63_fraction_23_.nii.gz",
            "Patient_64_fraction_3_.nii.gz",
            "Patient_64_fraction_4_.nii.gz",
            "Patient_64_fraction_21_.nii.gz",
            "Patient_64_fraction_22_.nii.gz",
            "Patient_64_fraction_23_.nii.gz",
            "Patient_66_fraction_24_.nii.gz",
            "Patient_66_fraction_32_.nii.gz",
            "Patient_66_fraction_33_.nii.gz",
            "Patient_76_fraction_7_.nii.gz",
            "Patient_77_fraction_7_.nii.gz",
        ]

    if not data_root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_root}")

    if not (0 < train_fraction < 1):
        raise ValueError("train_fraction must be between 0 and 1")

    if folds < 2:
        raise ValueError("folds must be at least 2")

    def build_patient_cts(patient_ids: list[str], ct_dir: Path, blacklist: list[str]) -> list[str]:
        files: list[str] = []

        for patient_id in patient_ids:
            patient_dir = ct_dir / f"Patient_{patient_id}"

            fraction_names = [f for f in sorted(patient_dir.glob("*.nii.gz")) if f.name not in blacklist]

            files.extend(str(f) for f in fraction_names)

        return files

    # Get patient IDs from folder names like Patient_1, Patient_2, ...
    patient_dirs = sorted(p for p in data_root.glob("Patient_*") if p.is_dir())
    patient_ids = [p.name.split("_", 1)[1] for p in patient_dirs]

    if not patient_ids:
        raise ValueError(f"No patient folders found in: {data_root}")

    # Shuffle patients
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)

    # Train/test split at patient level
    train_num = int(len(patient_ids) * train_fraction)

    if train_num == 0 or train_num == len(patient_ids):
        raise ValueError(
            "train_fraction produced an empty train set or empty test set. Adjust train_fraction or dataset size"
        )

    train_ids = patient_ids[:train_num]
    test_ids = patient_ids[train_num:]

    # Fold size for validation split within training pool
    if len(train_ids) < folds:
        raise ValueError(f"Number of training patients ({len(train_ids)}) must be >= number of folds ({folds})")

    val_num = len(train_ids) // folds

    data_dict: dict[str | int, Any] = {}

    for fold in range(folds):
        start = fold * val_num
        end = min((fold + 1) * val_num, len(train_ids))

        val_ids = train_ids[start:end]
        fold_train_ids = [pid for pid in train_ids if pid not in val_ids]

        data_dict[fold] = {
            "train": build_patient_cts(fold_train_ids, data_root, blacklist),
            "val": build_patient_cts(val_ids, data_root, blacklist),
        }

    # No blacklist for test data
    data_dict["test"] = build_patient_cts(test_ids, data_root, [])

    if save:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=4)

    return data_dict


def extract_roi(
    image: torch.Tensor,
    step: int,
    axis: int,
    chunk_size: int,
    halo_size: int,
    image_size: int,
    model_type: Literal["encoder", "decoder"],
    factor: int,
) -> tuple[torch.Tensor, int]:
    """
    Extract one image chunk with halo padding along a selected spatial axis.

    This helper is used by custom sliding-window inference. It extracts a
    chunk from a 5D tensor and includes an additional halo region around the
    core chunk to reduce boundary artifacts.

    Expected input shape:
        [B, C, H, W, D]

    Supported split axes:
    - axis=2: split along height
    - axis=3: split along width

    Parameters
    ----------
    image : torch.Tensor
        Input tensor with shape [B, C, H, W, D].

    step : int
        Index of the current chunk along the selected axis.

    axis : int
        Spatial axis along which the chunk is extracted.
        Must be 2 or 3.

    chunk_size : int
        Size of the core chunk before adding halo.

    halo_size : int
        Number of extra voxels added on each side of the chunk.

    image_size : int
        Full image size along the selected splitting axis.

    model_type : Literal["encoder", "decoder"]
        Type of model applied to the chunk.
        Must be either "encoder" or "decoder".

    factor : int
        Spatial scaling factor between input and output.
        For encoder, output is smaller by this factor.
        For decoder, output is larger by this factor.

    Returns
    -------
    image_chunk : torch.Tensor
        Extracted image chunk with halo and padding.

    padding : int
        Offset used later to remove halo from the model output.

    Raises
    ------
    TypeError
        If `image` is not a torch.Tensor.

    ValueError
        If `image` is not 5-dimensional.
        If `axis` is not 2 or 3.
        If `model_type` is not "encoder" or "decoder".
        If `step`, `chunk_size`, `halo_size`, `image_size`, or `factor`
        have invalid values.
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"`image` must be a torch.Tensor. Got {type(image)}")

    if image.ndim != 5:
        raise ValueError(f"`image` must have shape [B, C, H, W, D]. Got {image.shape}")

    if axis not in [2, 3]:
        raise ValueError("`axis` must be either 2 or 3.")

    if model_type not in ["encoder", "decoder"]:
        raise ValueError("`model_type` must be either 'encoder' or 'decoder'")

    if step < 0:
        raise ValueError("`step` must be non-negative")

    if chunk_size <= 0:
        raise ValueError("`chunk_size` must be greater than 0")

    if halo_size < 0:
        raise ValueError("`halo_size` must be non-negative")

    if image_size <= 0:
        raise ValueError("`image_size` must be greater than 0")

    if factor <= 0:
        raise ValueError("`factor` must be greater than 0")

    start_idx = step * chunk_size
    end_idx = start_idx + chunk_size

    if start_idx >= image_size:
        raise ValueError(f"`step` is too large for image size. start_idx={start_idx}, image_size={image_size}")

    input_start = max(0, start_idx - halo_size)
    input_end = min(image_size, end_idx + halo_size)

    if axis == 2:
        image_chunk = image[:, :, input_start:input_end, :, :].contiguous()
    else:
        image_chunk = image[:, :, :, input_start:input_end, :].contiguous()

    current_size = image_chunk.shape[axis]
    target_size = chunk_size + 2 * halo_size
    pad_amount = target_size - current_size

    if pad_amount < 0:
        raise ValueError(
            f"`pad_amount` cannot be negative. Got {pad_amount}. Check chunk_size, halo_size, and image_size"
        )

    if model_type == "encoder":
        padding = (start_idx - input_start) // factor
    else:
        padding = (start_idx - input_start) * factor

    if axis == 2:
        image_chunk = torch.nn.functional.pad(
            image_chunk,
            (0, 0, 0, 0, 0, pad_amount),
        )
    else:
        image_chunk = torch.nn.functional.pad(
            image_chunk,
            (0, 0, 0, pad_amount, 0, 0),
        )

    return image_chunk, padding


def sliding_window_inference(
    image: torch.Tensor,
    chunk_size: int,
    halo_size: int,
    image_size: int,
    model: Callable[[torch.Tensor], torch.Tensor],
    model_type: Literal["encoder", "decoder"],
    factor: int,
) -> torch.Tensor:
    """
    Perform custom sliding-window inference with halo overlap.

    The image is split into smaller chunks along the height and width axes.
    Each chunk is expanded with a halo region before being passed through the
    model. After inference, the halo region is removed from the model output,
    and clean chunks are stitched back together.

    Expected input shape:
        [B, C, H, W, D]

    Splitting axes:
    - axis 2: height
    - axis 3: width

    The depth axis is not split.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor with shape [B, C, H, W, D].

    chunk_size : int
        Size of the core chunk before adding halo.

    halo_size : int
        Number of halo voxels added around each chunk.

    image_size : int
        Full image size along height and width.
        This assumes height and width have the same size.

    model
        Callable model used for inference on each chunk.

    model_type : Literal["encoder", "decoder"]
        Type of model used for inference.
        Must be either:
        - "encoder": output spatial size is smaller by `factor`
        - "decoder": output spatial size is larger by `factor`

    factor : int
        Spatial scaling factor between model input and output.

    Returns
    -------
    canvas : torch.Tensor
        Reconstructed output tensor after chunk inference and stitching.

    Raises
    ------
    TypeError
        If `image` is not a torch.Tensor.
        If `model` is not callable.

    ValueError
        If `image` is not 5-dimensional.
        If `model_type` is not "encoder" or "decoder".
        If `chunk_size`, `image_size`, or `factor` are not positive.
        If `halo_size` is negative.
        If `image_size` is not divisible by `chunk_size`.
        If encoder `chunk_size` is not divisible by `factor`.
        If input height or width is smaller than `image_size`.
    """

    if image.ndim != 5:
        raise ValueError(f"`image` must have shape [B, C, H, W, D]. Got {image.shape}")

    if model_type not in ["encoder", "decoder"]:
        raise ValueError("`model_type` must be either 'encoder' or 'decoder'")

    if chunk_size <= 0:
        raise ValueError("`chunk_size` must be greater than 0")

    if halo_size < 0:
        raise ValueError("`halo_size` must be non-negative")

    if image_size <= 0:
        raise ValueError("`image_size` must be greater than 0")

    if factor <= 0:
        raise ValueError("`factor` must be greater than 0")

    if image_size % chunk_size != 0:
        raise ValueError(
            f"`image_size` must be divisible by `chunk_size`. Got image_size={image_size}, chunk_size={chunk_size}"
        )

    if model_type == "encoder" and chunk_size % factor != 0:
        raise ValueError(
            "For encoder inference, `chunk_size` must be divisible by "
            f"`factor`. Got chunk_size={chunk_size}, factor={factor}"
        )

    if image.shape[2] < image_size or image.shape[3] < image_size:
        raise ValueError(
            f"Input image height and width must be at least `image_size`. "
            f"Got image shape {image.shape} and image_size={image_size}"
        )

    num_steps = image_size // chunk_size
    output_chunks = []

    for step_h in range(num_steps):
        image_chunk, padding_h = extract_roi(
            image=image,
            step=step_h,
            axis=2,
            chunk_size=chunk_size,
            halo_size=halo_size,
            image_size=image_size,
            model_type=model_type,
            factor=factor,
        )

        output_chunk = []

        for step_w in range(num_steps):
            image_subchunk, padding_w = extract_roi(
                image=image_chunk,
                step=step_w,
                axis=3,
                chunk_size=chunk_size,
                halo_size=halo_size,
                image_size=image_size,
                model_type=model_type,
                factor=factor,
            )

            subchunk = model(image_subchunk)

            if model_type == "encoder":
                core_output = chunk_size // factor
            else:
                core_output = chunk_size * factor

            clean_subchunk = subchunk[
                :,
                :,
                padding_h : padding_h + core_output,
                padding_w : padding_w + core_output,
                :,
            ].contiguous()

            output_chunk.append(clean_subchunk)

        output_chunks.append(output_chunk)

    canvas = torch.cat(
        [torch.cat(output_chunk, dim=3) for output_chunk in output_chunks],
        dim=2,
    )

    return canvas
