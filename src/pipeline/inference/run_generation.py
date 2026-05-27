import glob
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import (
    AutoencoderKlMaisi,
)
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)
from monai.data import ThreadDataLoader
from monai.networks.schedulers import RFlowScheduler
from torch.utils.data import Dataset
from tqdm import tqdm

from src.pipeline.config import MaisiTestingConfig
from src.pipeline.inference.encode_latents import _resolve_device, load_vae_model
from src.pipeline.inference.sliding_window_inference import sliding_window_inference


@dataclass
class PatientConditionDataset(Dataset):
    """
    Dataset for loading latent planning CT condition tensors.

    This dataset reads latent `.pt` tensors produced by the VAE encoding step.
    Each tensor represents the latent representation of a planning CT and is
    used as a condition for MAISI rectified flow generation.

    Assumptions:
    - `root_dir` contains latent `.pt` files.
    - Latent files are produced by `encode_latents(config)`.
    - File names follow the patient/fraction naming convention, e.g.:
        Patient_1_fraction_1_.pt
    - Fraction 1 is the planning CT condition.

    Parameters
    ----------
    root_dir : Path
        Directory containing latent planning CT tensors.

    samples : list[str]
        List of latent `.pt` tensor paths. This is initialized automatically
        in `__post_init__`.

    Raises
    ------
    FileNotFoundError
        If `root_dir` does not exist.

    ValueError
        If no latent `.pt` files are found.
    """

    root_dir: Path
    samples: list[str] = field(init=False)

    def __post_init__(self) -> None:
        """
        Validate the latent directory and collect latent `.pt` files.
        """

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Latent CT directory does not exist: {self.root_dir}")

        self.samples = sorted(glob.glob(str(self.root_dir / "*.pt")))

        if len(self.samples) == 0:
            raise ValueError(f"No latent `.pt` files found in: {self.root_dir}. Run `encode_latents(config)` first")

    def __len__(self) -> int:
        """
        Return the number of latent CT condition tensors.
        """

        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        """
        Load one latent planning CT tensor.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        data : torch.Tensor
            Latent planning CT tensor.

        path : str
            Path to the loaded latent tensor.
        """

        path = self.samples[idx]
        data = torch.load(path, weights_only=True)

        return data, path


def load_rflow_model(
    config: MaisiTestingConfig,
    device: torch.device,
) -> DiffusionModelUNetMaisi:
    """
    Load the MAISI Rectified Flow model.

    Assumptions
    -----------
    - The checkpoint already contains weights adapted for conditional generation.
    - The first convolution already expects 8 input channels:
        4 channels for the sampled latent noise
        4 channels for the planning CT latent condition
    - No manual 4-channel to 8-channel weight expansion is performed here.

    Parameters
    ----------
    config : MaisiTestingConfig
        Configuration object containing the rectified flow model config and
        checkpoint path.

    device : torch.device
        Device on which the model should be loaded.

    Returns
    -------
    model : DiffusionModelUNetMaisi
        Loaded MAISI rectified flow model in evaluation mode.

    Raises
    ------
    ValueError
        If `config.rflow_config` is None.

    FileNotFoundError
        If the rectified flow checkpoint file does not exist.

    RuntimeError
        If the checkpoint does not match the model architecture.
    """

    if config.rflow_config is None:
        raise ValueError("`config.rflow_config` cannot be None. Provide a valid rectified flow model configuration")

    if not config.rflow_weight_path.exists():
        raise FileNotFoundError(f"Rectified flow checkpoint does not exist: {config.rflow_weight_path}")

    model = DiffusionModelUNetMaisi(**config.rflow_config).to(device)

    checkpoint = torch.load(
        config.rflow_weight_path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "unet_state_dict" in checkpoint:
        state_dict = checkpoint["unet_state_dict"]
    else:
        state_dict = checkpoint

    first_layer_key = "conv_in.conv.weight"

    if first_layer_key in state_dict:
        checkpoint_shape = state_dict[first_layer_key].shape
        model_shape = model.state_dict()[first_layer_key].shape

        if checkpoint_shape != model_shape:
            raise RuntimeError(
                "Rectified flow checkpoint does not match the current model "
                "architecture.\n"
                f"Layer: {first_layer_key}\n"
                f"Checkpoint shape: {checkpoint_shape}\n"
                f"Model shape: {model_shape}\n\n"
                "This probably means that the checkpoint was not trained or "
                "adapted for the current `rflow_config`. If your model expects "
                "8 input channels, the checkpoint must also contain 8-channel "
                "weights"
            )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model


def decode_latent_to_ct(
    z_t: torch.Tensor,
    vae_model: AutoencoderKlMaisi,
    config: MaisiTestingConfig,
) -> torch.Tensor:
    """
    Decode a generated latent tensor back into CT image space.

    The latent tensor is decoded using the MAISI VAE decoder with
    sliding-window inference. The decoded image is then mapped back from
    the normalized range [0, 1] to an HU-like range [-1000, 1000].

    Assumptions:
    - `z_t` is a generated latent tensor.
    - `vae_model.decode` maps latent tensors back to normalized CT space.
    - CT intensities were previously scaled from [-1000, 1000] to [0, 1].
    - The inverse transformation is:
        HU = 2000 * x - 1000

    Parameters
    ----------
    z_t : torch.Tensor
        Generated latent tensor to decode.

    vae_model : AutoencoderKlMaisi
        Loaded MAISI VAE model.

    config : MaisiTestingConfig
        Configuration object containing decoder window size and inference
        settings.

    Returns
    -------
    reconstructed_ct : torch.Tensor
        Generated CT tensor in HU-like range, clipped to [-1000, 1000].
    """

    reconstructed_ct = sliding_window_inference(
        image=z_t,
        chunk_size=config.chunk_size_decoder,
        halo_size=config.halo_decoder,
        image_size=config.decoder_image_size,
        model=vae_model.decode,
        model_type="decoder",
        factor=config.decoder_factor,
    )

    reconstructed_ct = torch.clamp(
        2000 * reconstructed_ct - 1000,
        min=-1000,
        max=1000,
    )

    return reconstructed_ct


def generate_ct_variants(config: MaisiTestingConfig) -> None:
    """
    Generate CT variants from planning CT latent conditions.

    This function:
    - loads the VAE decoder
    - loads the rectified flow model
    - loads latent planning CT tensors from `config.latent_ct_dir`
    - samples random latent noise
    - conditions generation on the planning CT latent
    - decodes generated latents back to CT space
    - saves generated CT tensors to `config.generated_ct_dir`

    Assumptions:
    - `encode_latents(config)` has already been run.
    - `config.latent_ct_dir` contains `.pt` latent tensors.
    - Each latent filename contains `_fraction_`, e.g.:
        Patient_1_fraction_1_.pt
    - The rectified flow model expects concatenated input:
        [noise_latent, condition_latent]
    - The RFlow checkpoint is already compatible with the current
      conditional model architecture.

    Parameters
    ----------
    config : MaisiTestingConfig
        Pipeline configuration containing paths, model configs, scheduler
        settings, and inference parameters.

    Raises
    ------
    ValueError
        If scheduler config is missing required settings.
    """

    device = _resolve_device(config)

    if config.scheduler_config is None:
        raise ValueError("`config.scheduler_config` cannot be None")

    if "base_img_size_numel" not in config.scheduler_config:
        raise ValueError("`config.scheduler_config` must contain 'base_img_size_numel'")

    config.generated_ct_dir.mkdir(parents=True, exist_ok=True)

    vae_model = load_vae_model(config, device)
    rflow_model = load_rflow_model(config, device)

    scheduler = RFlowScheduler(**config.scheduler_config)
    scheduler.set_timesteps(
        config.steps,
        input_img_size_numel=config.scheduler_config["base_img_size_numel"],
    )

    dataset = PatientConditionDataset(config.latent_ct_dir)

    loader = ThreadDataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    class_labels = torch.tensor([0], dtype=torch.long, device=device)
    spacing_tensor = torch.tensor(
        [[*config.spacing]],
        dtype=torch.float32,
        device=device,
    )

    vae_model.eval()
    rflow_model.eval()

    with torch.no_grad():
        for condition_latent, condition_path in tqdm(
            loader,
            desc="Generating CT variants",
        ):
            condition_latent = condition_latent.to(device, non_blocking=True)

            filename = os.path.basename(condition_path[0])

            if "_fraction_" not in filename:
                raise ValueError(f"Cannot extract patient ID from filename: {filename}")

            patient_id = filename.split("_fraction_")[0]
            patient_dir = config.generated_ct_dir / patient_id
            patient_dir.mkdir(parents=True, exist_ok=True)

            for variant_idx in range(1, config.cts_per_patient + 1):
                z_t = torch.randn_like(condition_latent).to(device) * config.latent_scale

                for timestep in scheduler.timesteps:
                    timestep_tensor = torch.tensor([timestep], device=device)

                    model_input = torch.cat(
                        [z_t, condition_latent],
                        dim=1,
                    )

                    velocity_pred = rflow_model(
                        model_input,
                        timestep_tensor,
                        class_labels=class_labels,
                        spacing_tensor=spacing_tensor,
                    )

                    z_t, _ = scheduler.step(
                        velocity_pred,
                        timestep,
                        z_t,
                    )

                reconstructed_ct = decode_latent_to_ct(
                    z_t=z_t,
                    vae_model=vae_model,
                    config=config,
                )

                save_path = patient_dir / f"{patient_id}_gen_{variant_idx}.pt"
                torch.save(reconstructed_ct[0, ...].cpu(), save_path)
