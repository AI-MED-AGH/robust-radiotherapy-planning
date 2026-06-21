# NOTICE: This file utilizes configurations from: https://huggingface.co/nvidia/NV-Generate-CT
# Licensed by NVIDIA Corporation under the NVIDIA Open Model License.

from typing import cast

import torch
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss

_adv_loss_class = PatchAdversarialLoss(criterion="least_squares")
_perceptual_loss_class = PerceptualLoss(
    spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2
).eval()


def intensity_loss(recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Mean Absolute Error (L1 Loss) between the reconstructed and target tensors.

    Parameters
    ----------
    recon_x : torch.Tensor
        The reconstructed tensor output from the model.
    x : torch.Tensor
        The original ground-truth target tensor.

    Returns
    -------
    torch.Tensor
        The scalar L1 loss tensor.

    Raises
    ------
    ValueError
        If the spatial shapes of `recon_x` and `x` do not match.
    """

    if recon_x.shape != x.shape:
        raise ValueError(f"Shape mismatch for intensity loss: recon_x {recon_x.shape} vs x {x.shape}")

    return torch.nn.functional.l1_loss(recon_x, x)


def kl_loss(z_mu: torch.Tensor, z_sigma: torch.Tensor, norm_factor: int) -> torch.Tensor:
    """
    Compute the Kullback-Leibler (KL) divergence loss for a variational bottleneck.

    Parameters
    ----------
    z_mu : torch.Tensor
        The latent distribution mean tensor.

    z_sigma : torch.Tensor
        The latent distribution standard deviation tensor.

    norm_factor : int
        The normalization factor (typically total number of elements or batch dimensions)
        used to scale the computed summation.

    Returns
    -------
    torch.Tensor
        The normalized scalar KL divergence loss tensor.

    Raises
    ------
    ValueError
        If `z_mu` and `z_sigma` spatial shapes do not match, or if `norm_factor` is non-positive.
    """

    if z_mu.shape != z_sigma.shape:
        raise ValueError(f"Shape mismatch for KL loss: z_mu {z_mu.shape} vs z_sigma {z_sigma.shape}")

    if norm_factor <= 0:
        raise ValueError(f"The 'norm_factor' must be a positive non-zero integer, got: {norm_factor}")

    eps = 1e-10
    return 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2) + eps) - 1) / norm_factor


def perceptual_loss(recon_x: torch.Tensor, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Compute the perceptual similarity loss using a pre-trained SqueezeNet architecture.

    Parameters
    ----------
    recon_x : torch.Tensor
        The reconstructed tensor output from the model.
    x : torch.Tensor
        The original ground-truth target tensor.
    device : torch.device
        The target hardware device (CPU/CUDA) to transition the perceptual backbone onto.

    Returns
    -------
    torch.Tensor
        The scalar perceptual loss tensor.

    Raises
    ------
    ValueError
        If the spatial shapes of `recon_x` and `x` do not match.

    RuntimeError
        If moving the underlying perceptual evaluation network to the target device fails.
    """

    if recon_x.shape != x.shape:
        raise ValueError(f"Shape mismatch for perceptual loss: recon_x {recon_x.shape} vs x {x.shape}")

    try:
        _perceptual_loss_class.to(device)
    except Exception as err:
        raise RuntimeError(f"Failed to push the perceptual loss evaluator to device: {device}") from err

    return cast(torch.Tensor, _perceptual_loss_class(recon_x, x))


def adv_loss(
    logits_fake: torch.Tensor, for_discriminator: bool, logits_real: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Compute Least Squares Patch Adversarial loss for either the Generator or Discriminator step.

    Parameters
    ----------
    logits_fake : torch.Tensor
        Discriminator predictions evaluated on generated fake samples.

    for_discriminator : bool
        Determines the training context. If True, computes the dual real/fake joint loss
        for updating the discriminator. If False, evaluates the generator optimization step.

    logits_real : torch.Tensor | None, optional
        Discriminator predictions evaluated on authentic samples. Required explicitly
        when `for_discriminator` is evaluated as True.

    Returns
    -------
    torch.Tensor
        The calculated patch adversarial loss tensor.

    Raises
    ------
    ValueError
        If `for_discriminator` is True but no `logits_real` tensor is supplied.
    """

    if for_discriminator:
        if logits_real is None:
            raise ValueError("The 'logits_real' tensor must be supplied when 'for_discriminator' is True")
        loss_d_fake = _adv_loss_class(logits_fake, target_is_real=False, for_discriminator=True)
        loss_d_real = _adv_loss_class(logits_real, target_is_real=True, for_discriminator=True)
        return cast(torch.Tensor, (loss_d_fake + loss_d_real) * 0.5)
    else:
        return cast(torch.Tensor, _adv_loss_class(logits_fake, target_is_real=True, for_discriminator=False))


def unet_loss(pred: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """
    Compute the Mean Squared Error (MSE Loss) between predicted and ground-truth velocities.

    Parameters
    ----------
    pred : torch.Tensor
        The predicted velocity from the UNet pipeline.
    real : torch.Tensor
        The original ground-truth velocity tensor.

    Returns
    -------
    torch.Tensor
        The scalar MSE loss tensor.

    Raises
    ------
    ValueError
        If the spatial shapes of `pred` and `real` do not match.
    """

    if pred.shape != real.shape:
        raise ValueError(f"Shape mismatch for U-Net MSE loss: pred {pred.shape} vs real {real.shape}")

    return torch.nn.functional.mse_loss(pred, real)
