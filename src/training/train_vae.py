# NOTICE: This file utilizes configurations from: https://huggingface.co/nvidia/NV-Generate-CT
# Licensed by NVIDIA Corporation under the NVIDIA Open Model License.

import copy
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
import wandb
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.distributed import DistributedSampler

from src.data_utils import sliding_window_inference
from src.training.helpers.config import MaisiTrainingConfig
from src.training.helpers.data_loading import get_ct_dataloaders
from src.training.helpers.distributed_learning import init_distributed, synchronize_loss
from src.training.helpers.ema import EMA
from src.training.helpers.losses import adv_loss, intensity_loss, kl_loss, perceptual_loss
from src.training.helpers.model_init import discriminator_init, vae_init

# Optional optimizations
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def save_checkpoint(state: dict[str, Any], file: str | Path) -> None:
    """
    Save the current training state dictionary to a specified file checkpoint path.

    Parameters
    ----------
    state : dict[str, Any]
        The checkpoint payload containing states for models, optimizers, and tracking metrics.

    file : str | Path
        The destination file path where the checkpoint will be saved.

    Raises
    ------
    RuntimeError
        If saving the checkpoint to disk fails due to I/O or permissions issues.
    """

    try:
        torch.save(state, file)
    except Exception as err:
        raise RuntimeError(f"Failed to write training checkpoint to path: {file}") from err


def load_checkpoint(
    file: str | Path,
    autoencoder: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    scheduler_g: LambdaLR,
    scheduler_d: LambdaLR,
    ema: EMA,
    device: torch.device,
) -> tuple[int, int, float, dict[str, torch.Tensor]]:
    """
    Load a saved checkpoint file and restore weights and states for all training components.

    Parameters
    ----------
    file : str | Path
        Path to the saved checkpoint file.

    autoencoder : torch.nn.Module
        The generative variational autoencoder model.

    discriminator : torch.nn.Module
        The adversarial patch discriminator model.

    optimizer_g : torch.optim.Optimizer
        Optimizer assigned to the generative network.

    optimizer_d : torch.optim.Optimizer
        Optimizer assigned to the adversarial discriminator network.

    scheduler_g : LambdaLR
        Learning rate scheduler for the generator.

    scheduler_d : LambdaLR
        Learning rate scheduler for the discriminator.

    ema : EMA
        Exponential Moving Average coordinator tracking the VAE weights.

    device : torch.device
        The localized execution hardware target (CPU/GPU).

    Returns
    -------
    tuple[int, int, float, dict[str, torch.Tensor]]
        A tuple containing:
        - The next starting epoch integer.
        - The current accumulation of epochs without validation loss improvement.
        - The historical minimum validation loss float.
        - A dictionary containing the best model weights cached on the CPU.

    Raises
    ------
    FileNotFoundError
        If the target checkpoint file does not exist on disk.

    RuntimeError
        If the checkpoint file is corrupted or cannot be read by PyTorch.

    KeyError
        If any critical architectural states are missing from the loaded payload.
    """

    file_path = Path(file)
    if not file_path.is_file():
        raise FileNotFoundError(f"No checkpoint file found at target path: {file_path}")

    try:
        checkpoint = torch.load(file_path, map_location=device)
    except Exception as err:
        raise RuntimeError(f"Failed to read or parse checkpoint file at: {file_path}") from err

    required_keys = [
        "autoencoder_state_dict",
        "discriminator_state_dict",
        "optimizer_g_dict",
        "optimizer_d_dict",
        "scheduler_g_dict",
        "scheduler_d_dict",
        "ema_dict",
        "best_model_dict",
        "epoch",
        "epochs_no_improve",
        "best_loss",
    ]
    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(f"Required parameter key '{key}' is missing from checkpoint file: {file_path}")

    try:
        autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g_dict"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d_dict"])
        scheduler_g.load_state_dict(checkpoint["scheduler_g_dict"])
        scheduler_d.load_state_dict(checkpoint["scheduler_d_dict"])
        ema.ema_model.load_state_dict(checkpoint["ema_dict"])
    except Exception as err:
        raise RuntimeError("Failed to remap saved state dictionary weights onto the current model components") from err

    # Keep the best model weights on CPU to avoid unnecessary VRAM usage
    best_model_wts_cpu = {k: v.cpu() for k, v in checkpoint["best_model_dict"].items()}

    return (
        checkpoint["epoch"],
        checkpoint["epochs_no_improve"],
        checkpoint["best_loss"],
        best_model_wts_cpu,
    )


def train(config: MaisiTrainingConfig) -> None:
    """
    Main training pipeline execution sequence for the MAISI VAE architecture.

    Configures infrastructure, instantiates models and data contexts, and runs cross-epoch
    generative adversarial loops accompanied by automated checkpoint persistence and tracking.

    Parameters
    ----------
    config : MaisiTrainingConfig
        Training configuration object.

    Raises
    ------
    ValueError
        If key operational thresholds (epochs, val_interval, patience) are structurally invalid.
    """

    # Init distributed
    rank, world_size, local_rank, is_distributed, device = init_distributed()

    # W&B INITIALIZATION
    if rank == 0:
        run_id = None
        # Safely peek inside the checkpoint to extract the previous W&B run ID if it exists
        if config.vae_checkpoint_path.is_file():
            try:
                checkpoint_peek = torch.load(config.vae_checkpoint_path, map_location="cpu")
                run_id = checkpoint_peek.get("wandb_run_id")
            # Fallback to starting a fresh run if file is corrupted
            except Exception:
                pass

        # Initialize the run
        wandb.init(
            project="maisi-vae-training",
            config=asdict(config),
            id=run_id,
            resume="allow",
        )

        # Structure the X-axis mapping to handle sparse validation seamlessly
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("epoch_time", step_metric="epoch")

    # Define learning rate warmup
    first_lr_jump = config.vae_lr_warmup // 2
    second_lr_jump = config.vae_lr_warmup

    def warmup_rule(epoch: int) -> float:
        if epoch < first_lr_jump:
            return 0.1
        elif epoch < second_lr_jump:
            return 0.5
        else:
            return 1.0

    # Define models
    autoencoder: torch.nn.Module = vae_init(config).to(device)
    discriminator: torch.nn.Module = discriminator_init(config).to(device)

    # Define data loaders
    train_loader, val_loader = get_ct_dataloaders(config, is_distributed, rank, world_size)

    # Define optimizers
    optimizer_g = torch.optim.AdamW(
        params=autoencoder.parameters(), lr=config.vae_lr, weight_decay=config.vae_weight_decay
    )
    optimizer_d = torch.optim.AdamW(
        params=discriminator.parameters(), lr=config.vae_lr, weight_decay=config.vae_weight_decay
    )

    # Define learning rate schedulers
    scheduler_g = LambdaLR(optimizer_g, lr_lambda=warmup_rule)
    scheduler_d = LambdaLR(optimizer_d, lr_lambda=warmup_rule)

    # Define EMA for the VAE
    ema = EMA(autoencoder, decay=config.vae_ema_decay)

    # Load checkpoint if it is available
    if config.vae_checkpoint_path.is_file():
        start_epoch, epochs_no_improve, best_val_loss, best_model_wts = load_checkpoint(
            config.vae_checkpoint_path,
            autoencoder,
            discriminator,
            optimizer_g,
            optimizer_d,
            scheduler_g,
            scheduler_d,
            ema,
            device,
        )
    # Define starting values if there is no checkpoint
    else:
        start_epoch, epochs_no_improve, best_val_loss, best_model_wts = (
            0,
            0,
            float("inf"),
            copy.deepcopy(autoencoder.state_dict()),
        )

    # Make sure the models work correctly with DDP
    if is_distributed:
        autoencoder = DDP(autoencoder, device_ids=[local_rank], output_device=local_rank)
        discriminator = DDP(discriminator, device_ids=[local_rank], output_device=local_rank)

    autoencoder.compile()
    discriminator.compile()

    # Helper function to process mu and sigma as one image
    @torch.compile
    def encode_image(image: torch.Tensor) -> torch.Tensor:
        model = cast(AutoencoderKlMaisi, autoencoder.module if is_distributed else autoencoder)
        mu, sigma = model.encode(image)
        return torch.cat((mu, sigma), dim=1)

    # Helper function to decode latent image
    decode_fn = (
        cast(AutoencoderKlMaisi, autoencoder.module).decode
        if is_distributed
        else cast(AutoencoderKlMaisi, autoencoder).decode
    )
    decode_fn = torch.compile(decode_fn)

    # Training and validation loops
    for epoch in range(start_epoch, config.vae_epochs):
        # This is used to measure epoch time
        epoch_start_time = time.time()

        # Set the epoch for the samplers
        if is_distributed and hasattr(train_loader, "sampler") and train_loader.sampler is not None:
            cast(DistributedSampler[dict[str, torch.Tensor]], train_loader.sampler).set_epoch(epoch)

        # Save at the beginning of every epoch
        if rank == 0:
            # Unpack the raw model from DDP if running distributed
            raw_state_g = (
                cast(AutoencoderKlMaisi, autoencoder.module).state_dict()
                if is_distributed
                else autoencoder.state_dict()
            )
            raw_state_d = (
                cast(AutoencoderKlMaisi, discriminator.module).state_dict()
                if is_distributed
                else discriminator.state_dict()
            )

            save_checkpoint(
                {
                    "epoch": epoch,
                    "epochs_no_improve": epochs_no_improve,
                    "autoencoder_state_dict": raw_state_g,
                    "discriminator_state_dict": raw_state_d,
                    "best_model_dict": best_model_wts,
                    "ema_dict": ema.ema_model.state_dict(),
                    "best_loss": best_val_loss,
                    "optimizer_g_dict": optimizer_g.state_dict(),
                    "optimizer_d_dict": optimizer_d.state_dict(),
                    "scheduler_g_dict": scheduler_g.state_dict(),
                    "scheduler_d_dict": scheduler_d.state_dict(),
                    "wandb_run_id": wandb.run.id if wandb.run is not None else None,
                },
                config.vae_checkpoint_path,
            )

        # Safety barrier
        if is_distributed:
            dist.barrier()

        # TRAINING PHASE
        autoencoder.train()
        discriminator.train()
        train_epoch_losses = {
            "loss": torch.tensor(0.0, device=device),
            "recons_loss": torch.tensor(0.0, device=device),
            "kl_loss": torch.tensor(0.0, device=device),
            "p_loss": torch.tensor(0.0, device=device),
            "adv_loss": torch.tensor(0.0, device=device),
        }

        for batch in train_loader:
            images = batch["image"].to(device)
            optimizer_g.zero_grad(set_to_none=True)
            optimizer_d.zero_grad(set_to_none=True)

            # Normalize the KL by the number of elements in the batch
            norm_factor = images.numel()

            # Get loss values
            reconstruction, z_mu, z_sigma = cast(tuple[torch.Tensor, torch.Tensor, torch.Tensor], autoencoder(images))

            # Train Generator
            logits_fake = discriminator(reconstruction.contiguous())[-1]
            generator_loss = adv_loss(logits_fake, False)
            losses_train: dict[str, torch.Tensor] = {
                "recons_loss": intensity_loss(reconstruction, images),
                "kl_loss": kl_loss(z_mu, z_sigma, norm_factor) * config.kl_weight,
                "p_loss": perceptual_loss(reconstruction, images, device) * config.perceptual_weight,
                "adv_loss": generator_loss * config.adv_weight,
            }
            loss_g = torch.stack(list(losses_train.values())).sum(dim=0)
            loss_g.backward()  # type: ignore
            optimizer_g.step()

            # Train Discriminator
            logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
            logits_real = discriminator(images.contiguous().detach())[-1]
            loss_d = adv_loss(logits_fake, True, logits_real)
            loss_d.backward()  # type: ignore
            optimizer_d.step()

            # Update EMA
            ema.update(autoencoder)

            # Update loss values
            train_epoch_losses["loss"] += loss_g.detach()
            for key in losses_train.keys():
                loss_value = losses_train[key].detach()
                train_epoch_losses[key] += loss_value

        # Move the schedulers forward
        scheduler_g.step()
        scheduler_d.step()

        # Normalize the loss values
        for key in train_epoch_losses:
            train_epoch_losses[key] /= len(train_loader)

        # Synchronize and average the losses across all GPUs
        train_epoch_losses_float = synchronize_loss(train_epoch_losses, is_distributed, world_size)

        # VALIDATION PHASE
        autoencoder.eval()
        discriminator.eval()

        validation_occurred = False

        if (epoch + 1) % config.val_interval == 0:
            validation_occurred = True
            val_epoch_losses = {
                "loss": torch.tensor(0.0, device=device),
                "recons_loss": torch.tensor(0.0, device=device),
                "kl_loss": torch.tensor(0.0, device=device),
                "p_loss": torch.tensor(0.0, device=device),
            }
            ema.apply_shadow(autoencoder)
            for batch in val_loader:
                with torch.no_grad():
                    images = batch["image"].to(device)

                    # Normalize the KL by the number of elements in the batch
                    norm_factor = images.numel()

                    # Calculate the reconstruction and latent representation through sliding windows
                    latent_image = sliding_window_inference(
                        images,
                        config.chunk_size_encoder,
                        config.halo_encoder,
                        encode_image,
                        "encoder",
                        config.vae_factor,
                    )
                    z_mu, z_sigma = latent_image[:, :4, ...].contiguous(), latent_image[:, 4:, ...].contiguous()

                    reconstruction = sliding_window_inference(
                        z_mu,
                        config.chunk_size_decoder,
                        config.halo_decoder,
                        decode_fn,
                        "decoder",
                        config.vae_factor,
                    )
                    losses_val: dict[str, torch.Tensor] = {
                        "recons_loss": intensity_loss(reconstruction, images),
                        "kl_loss": kl_loss(z_mu, z_sigma, norm_factor) * config.kl_weight,
                        "p_loss": perceptual_loss(reconstruction, images, device) * config.perceptual_weight,
                    }

                    # Update loss values
                    val_epoch_losses["loss"] += torch.stack(list(losses_val.values())).sum(dim=0).detach()
                    for key in losses_val.keys():
                        loss_value = losses_val[key].detach()
                        val_epoch_losses[key] += loss_value

            for key in val_epoch_losses:
                val_epoch_losses[key] /= len(val_loader)

            # Synchronize and average the losses across all GPUs
            val_epoch_losses_float = synchronize_loss(val_epoch_losses, is_distributed, world_size)

            # EARLY STOPPING LOGIC (Runs on all ranks!)
            if val_epoch_losses_float["loss"] < best_val_loss:
                best_val_loss = val_epoch_losses_float["loss"]
                epochs_no_improve = 0

                # Only Rank 0 saves the actual weights to memory
                if rank == 0:
                    # Keep the best model weights on CPU to avoid uncessary VRAM usage
                    raw_model = cast(AutoencoderKlMaisi, autoencoder.module) if is_distributed else autoencoder
                    best_model_wts = {k: v.cpu() for k, v in raw_model.state_dict().items()}
            else:
                epochs_no_improve += config.val_interval
                # EVERY rank breaks simultaneously
                if epochs_no_improve >= config.patience:
                    break

            ema.restore(autoencoder)

        # LOGGING
        if rank == 0:
            epoch_time = time.time() - epoch_start_time
            log_dict = {
                "epoch": epoch,
                "epoch_time": epoch_time,
            }

            # Map training losses
            for key, val in train_epoch_losses_float.items():
                log_dict[f"train/{key}"] = val

            # Safely inject validation losses only on the epochs they were computed
            if validation_occurred:
                for key, val in val_epoch_losses_float.items():
                    log_dict[f"val/{key}"] = val

            wandb.log(log_dict)

    # Save the final weights
    if rank == 0:
        try:
            torch.save(best_model_wts, config.vae_final_weights_path)
            wandb.finish()
        except Exception as err:
            raise RuntimeError(
                f"Failed to persist final model weights to path: {config.vae_final_weights_path}"
            ) from err


def main() -> None:
    """
    The main entry point for training the VAE-GAN model.

    In order to run this script, `prepare_data.py` must be run first in order to have training data.

    The script should be run only on linux and from the repository root using:

        PYTHONPATH=$(pwd) torchrun --nproc_per_node=4 src/training/train_vae.py

    The script can also be run with no distributed capabilities:

        python -m src.training.train_vae
    """
    config = MaisiTrainingConfig()

    train(config)


if __name__ == "__main__":
    main()
