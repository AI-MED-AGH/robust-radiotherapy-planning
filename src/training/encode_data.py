from pathlib import Path

import torch
from tqdm import tqdm

from src.data_utils import sliding_window_inference
from src.training.helpers.config import MaisiTrainingConfig
from src.training.helpers.data_loading import get_ct_dataloaders
from src.training.helpers.model_init import vae_init

# Optional optimizations
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def process_and_save_latents(
    config: MaisiTrainingConfig,
) -> None:
    """
    Load the final trained VAE model, encode training and validation images into
    their corresponding latent spaces using sliding window inference, and save them.

    Parameters
    ----------
    config : MaisiTrainingConfig
        Training configuration object containing path alignments and extraction settings.

    Raises
    ------
    FileNotFoundError
        If the trained weights file specified in `config.vae_final_weights_path` is missing.

    RuntimeError
        If model checkpoint parsing or saving the processed latent representations fails.
    """

    # Resolve device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model
    model = vae_init(config).to(device)
    try:
        state_dict = torch.load(config.vae_final_weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as err:
        raise RuntimeError(f"Failed to load or apply state dictionary from {config.vae_final_weights_path}") from err
    model.eval()

    # Define the train and val data loader
    train_loader, val_loader = get_ct_dataloaders(config, False, 0, 1, True)

    # Use the mean of the distribution for determinism
    def vae_encoder_wrapper(image_patch: torch.Tensor) -> torch.Tensor:
        mu, _ = model.encode(image_patch)
        return mu

    # Pair each loader with its target destination directory
    data_tasks = [
        (train_loader, config.latent_ct_dir_train, "training"),
        (val_loader, config.latent_ct_dir_val, "validation"),
    ]

    with torch.no_grad():
        for loader, save_dir, name in data_tasks:
            for batch in tqdm(loader, desc=f"Encoding {name} data"):
                images = batch["image"].to(device)
                image_path = batch["image_path"][0]

                latent_image = sliding_window_inference(
                    images,
                    config.chunk_size_encoder,
                    config.halo_encoder,
                    vae_encoder_wrapper,
                    "encoder",
                    config.vae_factor,
                )

                out_path = save_dir / Path(image_path).name
                try:
                    torch.save(latent_image[0, ...].cpu(), out_path)
                except Exception as err:
                    raise RuntimeError(f"Failed to save processed latent space representation to: {out_path}") from err


def main() -> None:
    """
    The main entry point for encoding processed CT scans in .pt format through the trained VAE-GAN.

    The VAE-GAN must be trained before this is run (`train_vae.py`).

    The script should be run from the repository root using:

        python -m src.training.encode_data
    """
    config = MaisiTrainingConfig()

    process_and_save_latents(config)


if __name__ == "__main__":
    main()
