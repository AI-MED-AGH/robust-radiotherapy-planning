# NOTICE: This file utilizes configurations from: https://huggingface.co/nvidia/NV-Generate-CT
# Licensed by NVIDIA Corporation under the NVIDIA Open Model License.

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
import wandb
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import DiffusionModelUNetMaisi
from monai.data import ThreadDataLoader  # type: ignore[attr-defined]
from monai.networks.schedulers import RFlowScheduler  # type: ignore[attr-defined]
from torch.amp import autocast  # type: ignore[attr-defined]
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LinearLR, PolynomialLR, SequentialLR
from torch.utils.data.distributed import DistributedSampler

from src.training.helpers.config import MaisiTrainingConfig
from src.training.helpers.datasets import PatientFractionPairDataset
from src.training.helpers.distributed_learning import init_distributed, synchronize_loss
from src.training.helpers.ema import EMA
from src.training.helpers.losses import unet_loss
from src.training.helpers.model_init import unet_init

# Optional optimizations
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def save_checkpoint(state: dict[str, Any], file: str | Path) -> None:
    """
    Save the current training state dictionary to a specified checkpoint file.

    Parameters
    ----------
    state : dict[str, Any]
        The checkpoint payload containing structural models, optimizers, and tracking weights.

    file : str | Path
        The destination file path where the checkpoint will be saved.

    Raises
    ------
    RuntimeError
        If saving the checkpoint to disk fails due to I/O or permission issues.
    """

    try:
        torch.save(state, file)
    except Exception as err:
        raise RuntimeError(f"Failed to write training checkpoint to path: {file}") from err


def load_checkpoint(
    file: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: SequentialLR,
    ema: EMA,
    device: torch.device,
) -> int:
    """
    Load a saved Rectified Flow training checkpoint and restore state variables.

    Parameters
    ----------
    file : str | Path
        Path to the saved checkpoint file.

    model : torch.nn.Module
        The active UNet diffusion neural network module.

    optimizer : torch.optim.Optimizer
        Optimizer assigned to the model optimization step.

    scheduler : SequentialLR
        The composite sequential learning rate warmup and decay scheduler.

    ema : EMA
        Exponential Moving Average tracking container for weight smoothing.

    device : torch.device
        The localized hardware acceleration target (CPU/GPU).

    Returns
    -------
    int
        The next starting epoch index parsed from the checkpoint state.

    Raises
    ------
    FileNotFoundError
        If the target checkpoint file does not exist on disk.

    RuntimeError
        If the checkpoint file is corrupted or failed to map onto active states.

    KeyError
        If required tracking keys are missing from the parsed payload dictionary.

    """

    file_path = Path(file)
    if not file_path.is_file():
        raise FileNotFoundError(f"No checkpoint file found at target path: {file_path}")

    try:
        checkpoint = torch.load(file_path, map_location=device)
    except Exception as err:
        raise RuntimeError(f"Failed to read or parse checkpoint file at: {file_path}") from err

    required_keys = ["model_dict", "optimizer_dict", "scheduler_dict", "ema_dict", "epoch"]
    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(f"Required structural key '{key}' is missing from checkpoint file: {file_path}")

    try:
        model.load_state_dict(checkpoint["model_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_dict"])
        ema.ema_model.load_state_dict(checkpoint["ema_dict"])
    except Exception as err:
        raise RuntimeError("Failed to remap saved state dictionary weights onto the training components") from err

    return cast(int, checkpoint["epoch"])


def train(config: MaisiTrainingConfig) -> None:
    """
    Main pipeline optimization execution wrapper for the Rectified Flow UNet context.

    Parameters
    ----------
    config : MaisiTrainingConfig
        Training configuration object storing network hyper-parameters and storage configurations.

    Raises
    ------
    ValueError
        If core iteration limits or parameter values fail sanity boundaries.
    """

    # Init distributed
    rank, world_size, local_rank, is_distributed, device = init_distributed()

    # W&B INITIALIZATION
    if rank == 0:
        run_id = None
        # Safely peek inside the checkpoint to extract the previous W&B run ID if it exists
        if config.rflow_checkpoint_path.is_file():
            try:
                checkpoint_peek = torch.load(config.rflow_checkpoint_path, map_location="cpu")
                run_id = checkpoint_peek.get("wandb_run_id")
            # Fallback to starting a fresh run if file is corrupted
            except Exception:
                pass

        # Initialize the run
        wandb.init(
            project="maisi-unet-training",
            config=asdict(config),
            id=run_id,
            resume="allow",
        )

        # Structure the X-axis mapping to handle sparse validation seamlessly
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("epoch_time", step_metric="epoch")

    # Define the model
    model: torch.nn.Module = unet_init(config).to(device)
    model.train()
    # Define data loader
    dataset = PatientFractionPairDataset(config.latent_ct_dir_train)

    train_sampler: DistributedSampler[dict[str, torch.Tensor]] | None = None
    if is_distributed:
        # Create sampler for distributed mode
        train_sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )

    train_loader = ThreadDataLoader(
        dataset,  # type: ignore
        batch_size=config.rflow_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=False,
        pin_memory=True,
    )

    # Define optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.rflow_lr, weight_decay=config.rflow_weight_decay)

    # Define the learning rate scheduler
    num_batches = len(train_loader)
    total_steps = config.rflow_epochs * num_batches
    warmup_steps = config.rflow_lr_warmup * num_batches
    decay_steps = total_steps - warmup_steps
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    decay_scheduler = PolynomialLR(
        optimizer,
        total_iters=decay_steps,
        power=2.0,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[warmup_steps],
    )

    # Define EMA for the model
    ema = EMA(model, decay=config.rflow_ema_decay)

    # Define the noise scheduler
    noise_scheduler = RFlowScheduler(**cast(dict[str, Any], config.scheduler_config))

    # Load checkpoint if it is available
    if config.rflow_checkpoint_path.is_file():
        start_epoch = load_checkpoint(
            config.rflow_checkpoint_path,
            model,
            optimizer,
            scheduler,
            ema,
            device,
        )
    # Define starting values if there is no checkpoint
    else:
        start_epoch = 0

    # Make sure the model works correctly with DDP
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    model.compile()

    # The spacing and class tensors are consistent so they can be defined here
    class_labels = torch.tensor([2], dtype=torch.long).to(device)
    spacing_tensor = torch.tensor([[*config.spacing]], dtype=torch.float32).to(device)

    # Training loop
    for epoch in range(start_epoch, config.rflow_epochs):
        # This is used to measure epoch time
        epoch_start_time = time.time()

        # Set the epoch for the samplers
        if is_distributed and hasattr(train_loader, "sampler") and train_loader.sampler is not None:
            cast(DistributedSampler[dict[str, torch.Tensor]], train_loader.sampler).set_epoch(epoch)

        # Save at the beginning of every epoch
        if rank == 0:
            # Unpack the raw model from DDP if running distributed
            raw_state = (
                cast(DiffusionModelUNetMaisi, model.module).state_dict() if is_distributed else model.state_dict()
            )

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_dict": raw_state,
                    "optimizer_dict": optimizer.state_dict(),
                    "scheduler_dict": scheduler.state_dict(),
                    "ema_dict": ema.ema_model.state_dict(),
                    "wandb_run_id": wandb.run.id if wandb.run is not None else None,
                },
                config.rflow_checkpoint_path,
            )

        # Safety barrier
        if is_distributed:
            dist.barrier()

        # TRAINING PHASE
        train_epoch_losses = {
            "loss": torch.tensor(0.0, device=device),
        }

        for batch in train_loader:
            condition = batch["condition"].to(device) * config.latent_scale
            target = batch["target"].to(device) * config.latent_scale
            optimizer.zero_grad(set_to_none=True)

            # Expand labels and spacing by batch size
            batch_size = condition.shape[0]
            current_class_labels = class_labels.expand(batch_size)
            current_spacing_tensor = spacing_tensor.expand(batch_size, -1)

            # Create the noisy latent
            noise = torch.randn_like(condition)
            timesteps = noise_scheduler.sample_timesteps(target)  # type: ignore
            noisy_latent = noise_scheduler.add_noise(original_samples=target, noise=noise, timesteps=timesteps)

            # Get the output velocity
            model_input = torch.cat([noisy_latent, condition], dim=1)
            with autocast(device.type, dtype=torch.bfloat16):
                pred = model(
                    model_input, timesteps, class_labels=current_class_labels, spacing_tensor=current_spacing_tensor
                )

                # Calculate loss
                real = (target - noise).to(pred.dtype)
                losses: dict[str, torch.Tensor] = {
                    "loss": unet_loss(pred, real),
                }

            # Step the optimizer
            losses["loss"].backward()  # type: ignore
            optimizer.step()

            # Step the scheduler after the optimizers
            scheduler.step()

            # Update EMA
            ema.update(model)

            # Update loss values
            for key in losses.keys():
                loss_value = losses[key].detach()
                train_epoch_losses[key] += loss_value

        # Normalize the loss values
        for key in train_epoch_losses:
            train_epoch_losses[key] /= len(train_loader)

        # Synchronize and average the losses across all GPUs
        train_epoch_losses_float = synchronize_loss(train_epoch_losses, is_distributed, world_size)

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

            wandb.log(log_dict)

        # Save smoothed weights periodically
        if (epoch + 1) % config.rflow_save_interval == 0:
            ema.apply_shadow(model)
            # Unpack the raw model from DDP if running distributed
            raw_state = (
                cast(DiffusionModelUNetMaisi, model.module).state_dict() if is_distributed else model.state_dict()
            )
            current_model_wts = {k: v.detach().cpu().clone() for k, v in raw_state.items()}
            if rank == 0:
                try:
                    torch.save(current_model_wts, config.rflow_weights_path / f"epoch_{epoch}.pt")
                except Exception as err:
                    raise RuntimeError(
                        f"Failed to persist evaluation tracking states to directory: {config.rflow_weights_path}"
                    ) from err
            ema.restore(model)

    # Finish the training run
    if rank == 0:
        wandb.finish()


def main() -> None:
    """
    The main entry point for training the Rectified flow model.

    In order to run this script, `encode_data.py` must be run first in order to have training data.

    The script should be run only on Linux and from the repository root using:

        PYTHONPATH=$(pwd) torchrun --nproc_per_node=8 src/training/train_rflow.py

    The script can also be run with no distributed capabilities:

        python -m src.training.train_rflow
    """
    config = MaisiTrainingConfig()

    train(config)


if __name__ == "__main__":
    main()
