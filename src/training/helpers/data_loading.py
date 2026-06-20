import glob
import os

import torch
from monai.data import Dataset, ThreadDataLoader  # type: ignore[attr-defined]
from monai.transforms import (  # type: ignore[attr-defined]
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandSpatialCropd,
    ScaleIntensityRanged,
    SpatialPadd,
)
from torch.utils.data.distributed import DistributedSampler

from src.data_utils import LoadProcessedTensord
from src.training.helpers.config import MaisiTrainingConfig


def get_ct_preprocessing_transform(config: MaisiTrainingConfig) -> Compose:
    """
    Generate the MONAI transformation pipeline used for processing raw NIfTI scans.

    Parameters
    ----------
    config : MaisiTrainingConfig
        Training configuration object.

    Returns
    -------
    Compose
        A MONAI Compose object containing the preprocessing sequence.
    """

    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS", labels=None),
            # This order of operations makes it so the image is padded with water and not air.
            # A lot of what would have been in the image in the padded space is unknown,
            # so whether it's filled with water or air does not matter too much.
            SpatialPadd(
                keys=["image"],
                spatial_size=config.image_size,
                mode="constant",
                constant_values=0,
            ),
            CenterSpatialCropd(
                keys=["image"],
                roi_size=config.image_size,
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=config.data_min,
                a_max=config.data_max,
                b_min=0,
                b_max=1,
                clip=True,
            ),
            EnsureTyped(keys=["image"], dtype=torch.float32),
        ]
    )


def _get_ct_transform_train(config: MaisiTrainingConfig) -> Compose:
    """
    Generate the runtime training transformation pipeline for preprocessed tensors.

    Parameters
    ----------
    config : MaisiTrainingConfig
        Training configuration object.

    Returns
    -------
    Compose
        Transformation pipeline including random spatial cropping.
    """

    return Compose(
        [
            LoadProcessedTensord(keys=["image"]),
            # Take a random crop dictated by chunk size
            RandSpatialCropd(
                keys=["image"],
                roi_size=config.chunk_size_encoder,
                random_size=False,
            ),
            EnsureTyped(keys=["image"], track_meta=False),
        ]
    )


def _get_ct_transform_val(config: MaisiTrainingConfig) -> Compose:
    """
    Generate the runtime validation/inference pipeline for preprocessed tensors.

    Parameters
    ----------
    config : MaisiTrainingConfig
        Training configuration object.

    Returns
    -------
    Compose
        Transformation pipeline optimized for validation evaluations.
    """

    return Compose(
        [
            LoadProcessedTensord(keys=["image"]),
            EnsureTyped(keys=["image"], track_meta=False),
        ]
    )


def get_ct_dataloaders(
    config: MaisiTrainingConfig, is_distributed: bool, rank: int, world_size: int, inference: bool = False
) -> tuple[ThreadDataLoader, ThreadDataLoader]:
    """
    Build and return thread-backed data loaders for training and validation datasets.

    Parameters
    ----------
    config : MaisiTrainingConfig
        Training configuration object.

    is_distributed : bool
        Flag indicating whether training is running across distributed processes (DDP).

    rank : int
        The identifier of the current GPU process.

    world_size : int
        The total number of distributed worker processes.

    inference : bool, optional
        If True, configures the training data loader to mirror single-sample inference behavior.

    Returns
    -------
    tuple[ThreadDataLoader, ThreadDataLoader]
        A tuple containing the training and validation ThreadDataLoaders.

    Raises
    ------
    ValueError
        If no preprocessed files are found, or if distributed parameters are invalid.
    """

    if is_distributed:
        if world_size <= 0:
            raise ValueError(f"world_size must be greater than 0 in distributed mode, got: {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(
                f"Distributed rank must be between 0 and world_size-1, but got rank={rank}, world_size={world_size}"
            )

    train_files = [
        {"image": f, "image_path": f} for f in sorted(glob.glob(os.path.join(config.processed_ct_dir_train, "*.pt")))
    ]
    val_files = [
        {"image": f, "image_path": f} for f in sorted(glob.glob(os.path.join(config.processed_ct_dir_val, "*.pt")))
    ]

    if not train_files:
        raise ValueError(f"No preprocessed '.pt' files found in training directory: {config.processed_ct_dir_train}")
    if not val_files:
        raise ValueError(f"No preprocessed '.pt' files found in validation directory: {config.processed_ct_dir_val}")

    train_ds = Dataset(
        data=train_files, transform=_get_ct_transform_val(config) if inference else _get_ct_transform_train(config)
    )
    val_ds = Dataset(data=val_files, transform=_get_ct_transform_val(config))

    train_sampler: DistributedSampler[dict[str, torch.Tensor]] | None = None
    val_sampler: DistributedSampler[dict[str, torch.Tensor]] | None = None
    if is_distributed:
        # Create samplers for distributed mode
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=not inference,
            drop_last=False,
        )
        # Validation sampler does NOT shuffle, but still splits data across GPUs
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    # Create DataLoaders
    train_loader = ThreadDataLoader(
        train_ds,
        batch_size=1 if inference else config.vae_batch_size_train,
        shuffle=False if inference else (train_sampler is None),
        sampler=train_sampler,
        drop_last=False,
        pin_memory=True,
    )
    val_loader = ThreadDataLoader(
        val_ds,
        batch_size=1 if inference else config.vae_batch_size_val,
        shuffle=False,
        sampler=val_sampler,
        pin_memory=True,
    )

    return train_loader, val_loader
