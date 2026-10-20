import re
from pathlib import Path
from typing import cast

import torch
from tqdm import tqdm

from src.training.helpers.config import MaisiTrainingConfig
from src.training.helpers.dvf_registration import compute_bspline_dvf


def generate_and_save_all_dvfs(config: MaisiTrainingConfig) -> None:
    """
    Batch-generate inter-patient and intra-patient non-rigid deformation fields across a CT dataset.

    Parameters
    ----------
    config : MaisiTrainingConfig
        Global configuration object containing directory paths (`processed_ct_dir_train`,
        `dvf_dir_train`, `atlas_ct_path`) and registration hyperparameters.

    Raises
    ------
    FileNotFoundError
        If configured training directories or Atlas CT path cannot be located.

    ValueError
        If no matching patient files are found in `config.processed_ct_dir_train`.
    """

    train_dir = config.processed_ct_dir_train
    dvf_dir = config.dvf_dir_train
    atlas_path = config.atlas_ct_path

    if not train_dir.exists():
        raise FileNotFoundError(f"Training data directory missing: {train_dir}")

    dvf_dir.mkdir(parents=True, exist_ok=True)

    filename_pattern = re.compile(r"Patient_(\d+)_fraction_(\d+)_\.pt")

    fraction_ones: dict[str, Path] = {}
    other_fractions: list[tuple[str, str, Path]] = []

    for file_path in train_dir.glob("Patient_*_fraction_*_.pt"):
        match = filename_pattern.match(file_path.name)
        if match:
            patient_id, fraction_id = match.groups()
            if fraction_id == "1":
                fraction_ones[patient_id] = file_path
            else:
                other_fractions.append((patient_id, fraction_id, file_path))

    if not fraction_ones:
        raise ValueError(f"No baseline 'fraction_1' CT files found in {train_dir}")

    ct_sum: torch.Tensor | None = None
    count = 0

    for _, f1_path in tqdm(fraction_ones.items(), desc="Atlas CT"):
        ct_tensor = torch.load(f1_path, map_location="cpu", weights_only=False).squeeze(0)

        if ct_sum is None:
            ct_sum = ct_tensor.clone()
        else:
            ct_sum += ct_tensor
        count += 1

    atlas_ct = cast(torch.Tensor, ct_sum) / count
    torch.save(atlas_ct, atlas_path)

    for _, f1_path in tqdm(fraction_ones.items(), desc="Inter-Patient DVFs"):
        out_path = dvf_dir / f1_path.name

        f1_tensor = torch.load(f1_path, map_location="cpu", weights_only=False).squeeze(0)
        dvf_inter = compute_bspline_dvf(
            atlas_ct,
            f1_tensor,
            config.spacing,
        )
        torch.save(dvf_inter, out_path)

    for patient_id, _, moving_path in tqdm(other_fractions, desc="Intra-Patient DVFs"):
        out_path = dvf_dir / moving_path.name

        f1_path = fraction_ones[patient_id]
        f1_tensor = torch.load(f1_path, map_location="cpu", weights_only=False).squeeze(0)
        moving_tensor = torch.load(moving_path, map_location="cpu", weights_only=False).squeeze(0)
        dvf_intra = compute_bspline_dvf(
            f1_tensor,
            moving_tensor,
            config.spacing,
        )
        torch.save(dvf_intra, out_path)


def main() -> None:
    """
    Main execution entry point for training and persisting the global PCA model.

    Run via module execution:
        python -m src.training.prepare_dvfs
    """

    config = MaisiTrainingConfig()
    generate_and_save_all_dvfs(config)


if __name__ == "__main__":
    main()
