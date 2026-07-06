import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import torch
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import (
    AutoencoderKlMaisi,
)
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)
from monai.data import ThreadDataLoader  # type: ignore[attr-defined]
from monai.data.dataset import Dataset as MonaiDataset
from monai.networks.schedulers import RFlowScheduler  # type: ignore[attr-defined]
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from src.data_utils import sliding_window_inference
from src.pipeline.config import MaisiTestingConfig
from src.pipeline.helpers.cleanup import clear_directory_contents
from src.pipeline.helpers.helpers import _clean_pipeline_memory
from src.pipeline.inference.encode_latents import load_vae_model


@dataclass
class PatientConditionDataset(TorchDataset[tuple[torch.Tensor, str]]):
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

    state_dict = torch.load(
        config.rflow_weight_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(state_dict)
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
    the normalized range [0, 1] to the CT intensity range defined in the
    pipeline configuration.

    Assumptions:
    - `z_t` is a generated latent tensor.
    - `vae_model.decode` maps latent tensors back to normalized CT space.
    - CT intensities were previously scaled from
    [`config.data_min`, `config.data_max`] to [0, 1].
    - The inverse transformation is:
        CT = (config.data_max - config.data_min) * x + config.data_min

    Parameters
    ----------
    z_t : torch.Tensor
        Generated latent tensor to decode.

    vae_model : AutoencoderKlMaisi
        Loaded MAISI VAE model.

    config : MaisiTestingConfig
        Configuration object containing decoder window size, inference settings,
        and CT intensity range values.

    Returns
    -------
    reconstructed_ct : torch.Tensor
        Generated CT tensor mapped back to the configured CT intensity range and
        clipped to [`config.data_min`, `config.data_max`].
    """

    reconstructed_ct = sliding_window_inference(
        image=z_t,
        chunk_size=config.chunk_size_decoder,
        halo_size=config.halo_decoder,
        model=vae_model.decode,
        model_type="decoder",
        factor=config.encoder_decoder_factor,
    )

    reconstructed_ct = (
        torch.clamp(
            (config.data_max - config.data_min) * reconstructed_ct + config.data_min,
            min=config.data_min,
            max=config.data_max,
        )
        .round()
        .int()
    )

    return reconstructed_ct


def generate_ct_variants(config: MaisiTestingConfig) -> None:
    """
    Generate CT variants from planning CT latent conditions.

    This function:
    - clears `config.generated_ct_dir`
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

    OSError
        If the generated CT output directory cannot be cleared.
    """

    device = torch.device(config.device)

    if config.scheduler_config is None:
        raise ValueError("`config.scheduler_config` cannot be None")

    if "base_img_size_numel" not in config.scheduler_config:
        raise ValueError("`config.scheduler_config` must contain 'base_img_size_numel'")

    clear_directory_contents(config.generated_ct_dir)

    vae_model = load_vae_model(config, device)
    rflow_model = load_rflow_model(config, device)

    scheduler = RFlowScheduler(**config.scheduler_config)
    scheduler.set_timesteps(
        config.steps,
        input_img_size_numel=config.scheduler_config["base_img_size_numel"],
    )

    dataset = PatientConditionDataset(config.latent_ct_dir)

    loader = ThreadDataLoader(
        dataset=cast(MonaiDataset, dataset),
        batch_size=1,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    class_labels = torch.tensor([2], dtype=torch.long, device=device)
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
            condition_latent = condition_latent.to(device, non_blocking=True) * config.latent_scale

            filename = os.path.basename(condition_path[0])

            if "_fraction_" not in filename:
                raise ValueError(f"Cannot extract patient ID from filename: {filename}")

            patient_id = filename.split("_fraction_")[0]
            patient_dir = config.generated_ct_dir / patient_id
            patient_dir.mkdir(parents=True, exist_ok=True)

            for variant_idx in range(1, config.cts_per_patient + 1):
                z_t = torch.randn_like(condition_latent).to(device)

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
                    z_t=z_t / config.latent_scale,
                    vae_model=vae_model,
                    config=config,
                )

                save_path = patient_dir / f"{patient_id}_gen_{variant_idx}.pt"
                torch.save(reconstructed_ct[0, ...].cpu(), save_path)

    del loader, rflow_model, scheduler, vae_model
    del class_labels, condition_latent, reconstructed_ct, spacing_tensor, z_t
    _clean_pipeline_memory()
