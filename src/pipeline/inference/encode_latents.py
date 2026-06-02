import glob
import os
from pathlib import Path
from typing import Any, cast

import torch
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import (
    AutoencoderKlMaisi,
)
from monai.data import Dataset, ThreadDataLoader  # type: ignore[attr-defined]
from monai.transforms import Compose, EnsureTyped, MapTransform  # type: ignore[attr-defined]
from tqdm import tqdm

from src.data_utils import sliding_window_inference
from src.pipeline.config import MaisiTestingConfig


class LoadProcessedTensord(MapTransform):
    """
    Load processed CT tensors saved as `.pt` files.

    This transform is used after the preprocessing step, where planning CTs
    are already loaded, scaled, cropped/padded, and saved as torch tensors.

    Assumptions:
    - Input files are `.pt` tensors.
    - Each file contains one processed planning CT.
    - The tensor was created by `prepare_test_data.py`.
    - The dictionary contains keys listed in `self.keys`.

    Parameters
    ----------
    keys : list[str]
        Dictionary keys pointing to `.pt` tensor files.

    Returns
    -------
    d : dict
        Dictionary with loaded torch tensors replacing file paths.

    Raises
    ------
    FileNotFoundError
        If a tensor file does not exist.
    """

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        d = dict(data)

        for key in self.keys:
            key_str = cast(str, key)
            tensor_path = Path(d[key_str])

            if not tensor_path.exists():
                raise FileNotFoundError(f"Processed CT tensor does not exist: {tensor_path}")

            d[key_str] = torch.load(tensor_path, weights_only=True)

        return d


def load_vae_model(
    config: MaisiTestingConfig,
    device: torch.device,
) -> AutoencoderKlMaisi:
    """
    Initialize the MAISI VAE model and load pretrained weights.

    Assumptions:
    - `config.vae_config` matches the architecture used to train the weights.
    - `config.vae_weight_path` points to a valid MAISI VAE checkpoint.
    - The checkpoint is compatible with `AutoencoderKlMaisi`.

    Parameters
    ----------
    config : MaisiTestingConfig
        Configuration object containing VAE architecture and weight path.

    device : torch.device
        Device on which the model should be loaded.

    Returns
    -------
    vae_model : AutoencoderKlMaisi
        Initialized VAE model in evaluation mode.

    Raises
    ------
    FileNotFoundError
        If the VAE weight file does not exist.

    RuntimeError
        If the checkpoint cannot be loaded into the VAE model.
        This usually means that `config.vae_config` does not match the
        checkpoint architecture.
    """

    if config.vae_config is None:
        raise ValueError("`config.vae_config` is None. The VAE configuration must be set before loading the model")

    if not config.vae_weight_path.exists():
        raise FileNotFoundError(f"VAE weight file does not exist: {config.vae_weight_path}")

    vae_model = AutoencoderKlMaisi(**config.vae_config).to(device)

    state_dict = torch.load(
        config.vae_weight_path,
        map_location=device,
        weights_only=True,
    )

    vae_model.load_state_dict(state_dict)
    vae_model.eval()

    return vae_model


def build_processed_ct_loader(
    config: MaisiTestingConfig,
    batch_size: int = 1,
) -> ThreadDataLoader:
    """
    Build a dataloader for processed planning CT tensors.

    This function scans `config.processed_ct_dir` for `.pt` files and creates
    a MONAI dataloader that loads them for VAE encoding.

    Each sample has the format:
        {
            "image": path_to_processed_ct_tensor,
            "save_path": path_to_output_latent_tensor
        }

    Assumptions:
    - Processed CTs are stored in:
        config.processed_ct_dir
    - Latent CTs should be saved to:
        config.latent_ct_dir
    - Processed CTs are saved as `.pt` files.
    - The filename is preserved when saving latent tensors.

    Parameters
    ----------
    config : MaisiTestingConfig
        Configuration object containing input/output directories.

    batch_size : int
        Batch size used by the dataloader.
        For large 3D CT volumes, this is usually 1.

    Returns
    -------
    loader : ThreadDataLoader
        MONAI dataloader over processed CT tensors.

    Raises
    ------
    FileNotFoundError
        If `config.processed_ct_dir` does not exist.

    ValueError
        If no `.pt` files are found or batch size is invalid.
    """

    if batch_size < 1:
        raise ValueError("`batch_size` must be at least 1")

    if not config.processed_ct_dir.exists():
        raise FileNotFoundError(f"Processed CT directory does not exist: {config.processed_ct_dir}")

    config.latent_ct_dir.mkdir(parents=True, exist_ok=True)

    transform = Compose(
        [
            LoadProcessedTensord(keys=["image"]),
            EnsureTyped(keys=["image"], track_meta=False),
        ]
    )

    test_files = []

    for file_path in sorted(glob.glob(str(config.processed_ct_dir / "*.pt"))):
        filename = os.path.basename(file_path)

        test_files.append(
            {
                "image": file_path,
                "save_path": str(config.latent_ct_dir / filename),
            }
        )

    if len(test_files) == 0:
        raise ValueError(
            f"No processed CT `.pt` files found in: {config.processed_ct_dir}. Run `prepare_test_data(config)` first"
        )

    dataset = Dataset(data=test_files, transform=transform)

    return ThreadDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=config.device == "cuda",
    )


def encode_latents(config: MaisiTestingConfig) -> None:
    """
    Encode processed planning CTs into latent representations using MAISI VAE.

    This function:
    - loads the pretrained MAISI VAE
    - loads processed planning CT tensors from `config.processed_ct_dir`
    - applies sliding-window VAE encoding
    - saves latent tensors to `config.latent_ct_dir`

    Assumptions:
    - `prepare_test_data(config)` has already been run.
    - Processed CTs are available as `.pt` files.
    - The VAE checkpoint is available at `config.vae_weight_path`.
    - The VAE config matches the checkpoint architecture.
    - The encoder output is deterministic by using the latent mean `mu`,
      not a randomly sampled latent.

    Parameters
    ----------
    config : MaisiTestingConfig
        Configuration object containing model paths, output paths, and
        inference settings.

    Raises
    ------
    FileNotFoundError
        If processed CTs or VAE weights are missing.

    ValueError
        If no processed CTs are found or config values are invalid.

    RuntimeError
        If the VAE checkpoint is incompatible with the VAE config.
    """

    device = _resolve_device(config)

    config.latent_ct_dir.mkdir(parents=True, exist_ok=True)

    vae_model = load_vae_model(
        config=config,
        device=device,
    )

    loader = build_processed_ct_loader(
        config=config,
        batch_size=1,
    )

    def vae_encoder_wrapper(image_patch: torch.Tensor) -> torch.Tensor:
        """
        Encode an image patch into latent space using the VAE encoder.

        The MAISI VAE encoder returns distribution parameters. For testing,
        the mean is used instead of sampling to keep encoding deterministic.

        Parameters
        ----------
        image_patch : torch.Tensor
            Input CT image patch.

        Returns
        -------
        mu : torch.Tensor
            Mean latent representation of the image patch.
        """

        mu, _ = vae_model.encode(image_patch)
        return mu

    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding planning CTs"):
            encoded_ct = sliding_window_inference(
                image=batch["image"].to(device, non_blocking=True),
                chunk_size=config.chunk_size_encoder,
                halo_size=config.halo_encoder,
                image_size=config.encoder_image_size,
                model=vae_encoder_wrapper,
                model_type="encoder",
                factor=config.encoder_decoder_factor,
            )

            save_path = Path(batch["save_path"][0])
            save_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(encoded_ct[0, ...].cpu(), save_path)


def _resolve_device(config: MaisiTestingConfig) -> torch.device:
    """
    Resolve inference device from config.

    If `config.device` is "cuda" but CUDA is unavailable, CPU is used instead.

    Parameters
    ----------
    config : MaisiTestingConfig
        Configuration object containing selected device.

    Returns
    -------
    device : torch.device
        Resolved torch device.
    """

    if config.device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")

    return torch.device(config.device)
