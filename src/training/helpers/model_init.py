# NOTICE: This file utilizes configurations and pre-trained weights from: https://huggingface.co/nvidia/NV-Generate-CT
# Licensed by NVIDIA Corporation under the NVIDIA Open Model License.

from typing import Any, cast

import torch
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import DiffusionModelUNetMaisi
from monai.networks.nets import PatchDiscriminator  # type: ignore[attr-defined]
from monai.networks.utils import normal_init

from src.training.helpers.config import MaisiTrainingConfig


def vae_init(config: MaisiTrainingConfig) -> AutoencoderKlMaisi:
    """
    Initialize the MAISI Variational Autoencoder (VAE) model and apply normal weight initialization.

    Parameters
    ----------
    config : MaisiTrainingConfig
        Training configuration object.

    Returns
    -------
    AutoencoderKlMaisi
        The initialized and randomly weighted VAE model.
    """
    model = AutoencoderKlMaisi(**cast(dict[str, Any], config.vae_config))
    model.apply(lambda m: normal_init(m, std=config.init_std))

    return model


def discriminator_init(config: MaisiTrainingConfig) -> PatchDiscriminator:
    """
    Initialize the Patch Discriminator model for adversarial training.

    Parameters
    ----------
    config : MaisiTrainingConfig
        Training configuration object containing discriminator architecture parameters.

    Returns
    -------
    PatchDiscriminator
        The initialized Patch Discriminator model.
    """
    model = PatchDiscriminator(**cast(dict[str, Any], config.discriminator_config))

    return model


def unet_init(config: MaisiTrainingConfig) -> DiffusionModelUNetMaisi:
    """
    Initialize the MAISI UNet Diffusion model, load pre-trained foundation weights,
    and adapt the input layer convolution matrix to support expanded multi-channel conditions.

    Parameters
    ----------
    config : MaisiTrainingConfig
        Training configuration object containing UNet configurations and foundation weight file paths.

    Returns
    -------
    DiffusionModelUNetMaisi
        The initialized UNet model adapted with loaded foundation parameters.

    Raises
    ------
    RuntimeError
        If the foundation weight file fails to load into memory via PyTorch.

    KeyError
        If required keys ('unet_state_dict' or 'conv_in.conv.weight') are missing from
        the checkpoint or the target model state dictionaries.
    """

    model = DiffusionModelUNetMaisi(**cast(dict[str, Any], config.rflow_config))

    # Load the foundation model weights securely
    try:
        checkpoint = torch.load(config.rflow_foundation_weights, weights_only=False)
    except Exception as err:
        raise RuntimeError(
            f"Failed to load foundation weights checkpoint from {config.rflow_foundation_weights}"
        ) from err

    if "unet_state_dict" not in checkpoint:
        raise KeyError(
            f"The loaded checkpoint at {config.rflow_foundation_weights} is missing the required 'unet_state_dict' key"
        )

    rflow_state_dict = checkpoint["unet_state_dict"]
    first_layer_key = "conv_in.conv.weight"

    if first_layer_key not in rflow_state_dict:
        raise KeyError(f"The key '{first_layer_key}' was not found within the loaded 'unet_state_dict'")

    model_state = model.state_dict()
    if first_layer_key not in model_state:
        raise KeyError(f"The key '{first_layer_key}' is missing from the initialized UNet model architecture")

    pretrained_weights = rflow_state_dict[first_layer_key]
    target_shape = model_state[first_layer_key].shape

    # Create new weights for the convolutions since it now has extra channels
    new_weights = torch.zeros(target_shape)

    new_weights[:, :4, ...] = pretrained_weights[:, :4, ...]  # The noise part
    new_weights[:, 4:, ...] = 0.0  # The Planning CT part

    # Update the dictionary and load smoothly into the model
    rflow_state_dict[first_layer_key] = new_weights

    try:
        model.load_state_dict(rflow_state_dict)
    except Exception as err:
        raise RuntimeError("Failed to map the adapted state_dict keys onto the UNet model structure.") from err

    return model
