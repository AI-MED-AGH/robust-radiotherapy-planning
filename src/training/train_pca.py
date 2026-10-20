import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA  # type: ignore[import-untyped]
from tqdm import tqdm

from src.training.helpers.config import MaisiTrainingConfig
from src.training.helpers.dvf_registration import transport_dvf_to_atlas


def save_pca_model_directory(
    pca_model: dict[str, Any],
    save_dir: Path,
) -> None:
    """
    Save fitted PCA model parameters to disk across separate files within a directory.

    Splitting the model prevents single-file binary stream buffer overflow errors
    during serialization of large 3D fields.

    Parameters
    ----------
    pca_model : dict[str, Any]
        Dictionary containing PCA parameters ('mean', 'components', 'singular_values',
        'explained_variance_ratio', 'spatial_shape').

    save_dir : Path
        Target directory where model files and components will be written.
    """

    save_dir.mkdir(parents=True, exist_ok=True)
    comp_dir = save_dir / "components"
    comp_dir.mkdir(exist_ok=True)

    meta = {
        "mean": pca_model["mean"].float(),
        "singular_values": pca_model["singular_values"].float(),
        "explained_variance_ratio": pca_model["explained_variance_ratio"].float(),
        "spatial_shape": pca_model["spatial_shape"],
        "n_components": pca_model["components"].shape[0],
    }
    torch.save(meta, save_dir / "meta_and_mean.pt")

    components = pca_model["components"]
    for idx in range(components.shape[0]):
        comp_tensor = components[idx].to(torch.float32)
        torch.save(comp_tensor, comp_dir / f"component_{idx:03d}.pt")


def fit_pca_on_atlas_dvfs(
    config: MaisiTrainingConfig,
    n_components: int,
    batch_size: int,
) -> dict[str, Any]:
    """
    Transport intra-patient DVFs into Atlas coordinate space and fit an Incremental PCA model.

    Parameters
    ----------
    config : MaisiTrainingConfig
        Configuration object containing dataset paths and voxel spacing metadata.

    n_components : int
        Number of principal components (eigenvectors) to fit.

    batch_size : int
        Number of samples processed per incremental SVD update batch.

    Returns
    -------
    dict[str, Any]
        Fitted PCA model parameters containing spatial mean field, eigenvector fields,
        singular values, and variance ratios.

    Raises
    ------
    FileNotFoundError
        If data or DVF directories do not exist.

    ValueError
        If parameters violate sample count bounds or batch size limits.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dir = config.processed_ct_dir_train
    dvf_dir = config.dvf_dir_train

    if not train_dir.exists():
        raise FileNotFoundError(f"Training data directory does not exist: {train_dir}")
    if not dvf_dir.exists():
        raise FileNotFoundError(f"DVF directory does not exist: {dvf_dir}")

    filename_pattern = re.compile(r"Patient_(\d+)_fraction_(\d+)_\.pt")

    fraction_ones: dict[str, Path] = {}
    other_fractions: list[tuple[str, Path]] = []

    for file_path in train_dir.glob("Patient_*_fraction_*_.pt"):
        match = filename_pattern.match(file_path.name)
        if match:
            patient_id, fraction_id = match.groups()
            if fraction_id == "1":
                fraction_ones[patient_id] = file_path
            else:
                other_fractions.append((patient_id, file_path))

    tasks: list[tuple[Path, Path]] = []
    for patient_id, moving_path in other_fractions:
        if patient_id in fraction_ones:
            dvf_intra_path = dvf_dir / moving_path.name
            dvf_inter_path = dvf_dir / fraction_ones[patient_id].name
            if dvf_intra_path.exists() and dvf_inter_path.exists():
                tasks.append((dvf_intra_path, dvf_inter_path))

    num_samples = len(tasks)
    if num_samples == 0:
        raise ValueError(f"No matching intra/inter DVF tensor pairs found in '{dvf_dir}'")

    if n_components > num_samples:
        raise ValueError(f"n_components ({n_components}) cannot exceed total sample count ({num_samples})")

    if batch_size < n_components:
        raise ValueError(f"batch_size ({batch_size}) must be >= n_components ({n_components}) for IncrementalPCA")

    sample_tensor = torch.load(tasks[0][0], map_location="cpu", weights_only=False)
    spatial_shape = tuple(sample_tensor.shape)

    # Shuffle the tasks for stability
    random.shuffle(tasks)

    batches: list[list[tuple[Path, Path]]] = []
    i = 0
    while i < num_samples:
        if num_samples - (i + batch_size) < n_components and (i + batch_size) < num_samples:
            batches.append(tasks[i:])
            break
        batches.append(tasks[i : i + batch_size])
        i += batch_size

    ipca = IncrementalPCA(n_components=n_components)

    for batch_tasks in tqdm(batches, desc="Fitting Incremental PCA in Atlas Space"):
        batch_matrices: list[np.ndarray] = []

        for dvf_intra_path, dvf_inter_path in batch_tasks:
            dvf_intra = torch.load(dvf_intra_path, map_location=device, weights_only=False)
            dvf_inter = torch.load(dvf_inter_path, map_location=device, weights_only=False)

            dvf_atlas = transport_dvf_to_atlas(
                dvf_intra=dvf_intra,
                dvf_inter=dvf_inter,
                voxel_spacing=config.spacing,
            )

            flat_vector = dvf_atlas.detach().cpu().numpy().astype(np.float32).reshape(-1)
            batch_matrices.append(flat_vector)

            del dvf_intra, dvf_inter, dvf_atlas

        batch_array = np.stack(batch_matrices, axis=0)
        ipca.partial_fit(batch_array)

    mean_tensor = torch.from_numpy(ipca.mean_.reshape(spatial_shape)).float()
    components_tensor = torch.from_numpy(ipca.components_.reshape((n_components, *spatial_shape))).float()

    singular_values = torch.from_numpy(ipca.singular_values_).float()
    explained_variance_ratio = torch.from_numpy(ipca.explained_variance_ratio_).float()

    return {
        "mean": mean_tensor,
        "components": components_tensor,
        "singular_values": singular_values,
        "explained_variance_ratio": explained_variance_ratio,
        "spatial_shape": spatial_shape,
    }


def process_and_save_pca(
    config: MaisiTrainingConfig,
    n_components: int = 5,
    batch_size: int = 10,
) -> None:
    """
    Fit Global PCA in Atlas space and write checkpoint directory output to disk.

    Parameters
    ----------
    config : MaisiTrainingConfig
        Training configuration object.

    n_components : int, default=5
        Number of principal components to compute.

    batch_size : int, default=10
        Number of volumes processed per incremental batch.
    """

    pca_model = fit_pca_on_atlas_dvfs(
        config=config,
        n_components=n_components,
        batch_size=batch_size,
    )

    save_pca_model_directory(
        pca_model=pca_model,
        save_dir=config.pca_model_dir,
    )


def main() -> None:
    """
    Main execution entry point for training and persisting the Global PCA model.

    Run via module execution:
        python -m src.training.train_pca
    """

    config = MaisiTrainingConfig()
    process_and_save_pca(
        config=config,
    )


if __name__ == "__main__":
    main()
