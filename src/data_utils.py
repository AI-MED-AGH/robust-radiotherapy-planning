from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def create_data_split_dict(
    seed: int = 42,
    train_fraction: float = 0.8,
    folds: int = 5,
    data_root: Path | None = None,
    output_json: Path | None = None,
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
        <src>/data_full/data_dict.json
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

    if not data_root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_root}")

    if not (0 < train_fraction < 1):
        raise ValueError("train_fraction must be between 0 and 1")

    if folds < 2:
        raise ValueError("folds must be at least 2")

    def build_patient_pairs(patient_ids: list[str], ct_dir: Path) -> list[dict[str, str]]:
        files: list[dict[str, str]] = []

        for patient_id in patient_ids:
            patient_dir = ct_dir / f"Patient_{patient_id}"
            first_fraction = patient_dir / f"Patient_{patient_id}_fraction_1_.nii.gz"

            if not first_fraction.exists():
                raise FileNotFoundError(f"Missing reference fraction for patient {patient_id}: {first_fraction}")

            fraction_names = [f for f in sorted(patient_dir.glob("*.nii.gz")) if f.name != first_fraction.name]

            files.extend(
                {
                    "moving_image": str(first_fraction),
                    "fixed_image": str(f),
                }
                for f in fraction_names
            )

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
        raise ValueError(
            f"Number of training patients ({len(train_ids)}) must be >= number of folds ({folds})"
        )

    val_num = len(train_ids) // folds

    data_dict: dict[str | int, Any] = {}

    for fold in range(folds):
        start = fold * val_num
        end = min((fold + 1) * val_num, len(train_ids))

        val_ids = train_ids[start:end]
        fold_train_ids = [pid for pid in train_ids if pid not in val_ids]

        data_dict[fold] = {
            "train": build_patient_pairs(fold_train_ids, data_root),
            "val": build_patient_pairs(val_ids, data_root),
        }

    data_dict["test"] = build_patient_pairs(test_ids, data_root)

    if save:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=4)

    return data_dict
