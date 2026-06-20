# NOTICE: This file utilizes configurations and pre-trained weights from: https://huggingface.co/nvidia/NV-Generate-CT
# Licensed by NVIDIA Corporation under the NVIDIA Open Model License.

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MaisiTrainingConfig:
    """
    Store configuration and architectural parameters for the MAISI CT training pipeline.

    This class maintains, normalizes, and validates spatial, intensity, hyperparameter,
    and filesystem structural requirements for both VAE and U-Net optimization tasks.

    Assumptions
    -----------
    - This class lives inside the `src/` package.
    - Full CT data is stored in:
        src/data_full/CT/Patient_x/...
    - Each patient folder contains longitudinal CT fractions:
        Patient_<id>_fraction_1_.nii.gz
        Patient_<id>_fraction_2_.nii.gz
        ...
    - Fraction 1 is treated as the planning CT (conditioning CT).
    - Training is performed on scans processed to have fixed image size and a range of [0, 1].
    These are saved in float32 (important due to precision needs) to .pt files.
    - MAISI model weights are stored in (the resulting ones and the ones used for warm starting):
        RESULTS/weights/

    Pipeline stages
    ---------------
    1. Prepare CT data for training by convering it to .pt format (`prepare_data.py`).
    2. Train the VAE (`train_vae.py`).
    3. Encode CTs into the latent space using the VAE (`encode_data.py`).
    4. Train the Rectified Flow U-Net (`train_rflow.py`).

    Parameters
    ----------
    data_dict_path : Path
        Path to the dataset split JSON configuration file.

    data_min : int
        Minimum Hounsfield Unit (HU) intensity value used for clipping and scale clipping normalization.

    data_max : int
        Maximum Hounsfield Unit (HU) intensity value used for clipping and scale clipping normalization.

    processed_ct_dir_train : Path
        Directory where preprocessed training CT volumes are stored or generated.

    processed_ct_dir_val : Path
        Directory where preprocessed validation CT volumes are stored or generated.

    latent_ct_dir_train : Path
        Directory where compressed training latent arrays (.pt) are stored or generated.

    latent_ct_dir_val : Path
        Directory where compressed validation latent arrays (.pt) are stored or generated.

    rflow_weight_path : Path
        Output destination for Rectified Flow weights saved throughout training.

    latent_scale : float
        Scaling factor applied to the condition to target image to make their distribution have unit variance.

    spacing : tuple[float, float, float]
        Target voxel resolution spacing layout sequence (H, W, D) configured during preprocessing.

    chunk_size_encoder : int
        Sliding-window sub-region context size utilized across spatial axes during VAE downsampling.

    chunk_size_decoder : int
        Sliding-window sub-region context size utilized across spatial axes during VAE upsampling.

    halo_encoder : int
        Boundary element extension padding added to encoder chunks to eliminate extraction edge artifacts.

    halo_decoder : int
        Boundary element extension padding added to decoder chunks to eliminate extraction edge artifacts.

    vae_factor : int
        Spatial downsampling ratio factor generated under the configured autoencoder model.

    image_size : tuple[int, int, int]
        Target matrix layout boundary size (H, W, D) processed under volume normalization pipelines.

    adv_weight : float
        Loss scale assigned to adversarial loss for VAE training.

    init_std : float
        Standard deviation multiplier for normal VAE weight initialization.

    kl_weight : float
        Loss scale assigned to the KL loss.

    patience : int
        Maximum epochs permitted without validation improvements before triggering early stopping (VAE).

    perceptual_weight : float
        Loss scale assigned to the perceptual loss.

    vae_batch_size_train : int
        Batch layout size utilized during VAE training (per process).

    vae_batch_size_val : int
        Batch layout size utilized during VAE validation (per process).

    vae_checkpoint_path : Path
        Destination path for saving the VAE checkpoint for resuming in the even of a crash.

    vae_ema_decay : float
        Exponential moving average decay factor for VAE parameters.

    vae_epochs : int
        Maximum epochs the VAE will run for.

    vae_final_weights_path : Path
        Path where the final VAE weights are saved.

    vae_lr : float
        Learning rate for both the VAE and the discriminator.

    vae_lr_warmup : int
        Total learning rate warmup in epochs for the VAE and the discriminator.

    vae_weight_decay : float
        Weight decay for AdamW for both the VAE and the discriminator.

    val_interval : int
        How many epochs to wait between calculating validation metrics for the VAE.

    rflow_batch_size : int
        Batch size for Rectified Flow training (per process).

    rflow_checkpoint_path : Path
        Destination path for saving the Rectified Flow checkpoint for resuming in the even of a crash.

    rflow_ema_decay : float
        Exponential moving average decay factor for Rectified Flow parameters.

    rflow_epochs : int
        Number of epochs to run for Rectified Flow training.

    rflow_foundation_weights : Path
        Path to the MAISI weights.

    rflow_weights_path : Path
        Directory in which to periodically save smoothed Rectified Flow weights in.

    rflow_lr : float
        Learning rate for the Rectified Flow U-Net.

    rflow_lr_warmup : int
        Total learning rate warmup in epochs for the Rectified Flow U-Net.

    rflow_save_interval : int
        How many epochs to wait between saving weights.

    rflow_weight_decay : float
        Weight decay for AdamW for the Rectified Flow U-Net.

    vae_config : dict[str, Any] | None
        VAE architecture configuration.
        If None, a default MAISI configuration is used.

    discriminator_config : dict[str, Any] | None
        Discriminator architecture configuration.
        If None, a default MAISI configuration is used.

    rflow_config : dict[str, Any] | None
        Rectified flow U-Net configuration.
        If None, a default MAISI configuration is used.

    scheduler_config : dict[str, Any] | None
        Sampling scheduler configuration.
        If None, a default configuration is used.

    Raises
    ------
    FileNotFoundError
        If required structural base files or data configurations do not exist on disk.

    ValueError
        If initialization variables contain logically invalid dimensional metrics or negative weights.

    Notes
    -----
    Training uses a custom sliding-window implementation based on
    chunk extraction with halo overlap. The depth dimension is processed
    as a whole, while sliding is performed only across height and width.

    Output directories are automatically created during initialization.
    This class stores and validates configuration values only and does
    not load data or models.
    """

    # Input data
    data_dict_path: Path = Path("src/data_full/data_dict.json")
    data_min: int = -1000
    data_max: int = 1500

    # Directories for outputs
    processed_ct_dir_train: Path = Path("RESULTS/MAISI_TRAINING/processed_ct/train")
    processed_ct_dir_val: Path = Path("RESULTS/MAISI_TRAINING/processed_ct/val")
    latent_ct_dir_train: Path = Path("RESULTS/MAISI_TRAINING/latents/train")
    latent_ct_dir_val: Path = Path("RESULTS/MAISI_TRAINING/latents/val")

    # Model weights
    rflow_weight_path: Path = Path("RESULTS/weights/diff_unet_3d_rflow-ct.pt")

    # Latent sampling config
    latent_scale: float = 1.949401

    # MAISI-specific config
    spacing: tuple[float, float, float] = (1.171875, 1.171875, 3.0)

    # Custom sliding-window inference
    # Chunk size c means the sliding window is c x c x d,
    # where d is the full image / latent depth.
    chunk_size_encoder: int = 128
    chunk_size_decoder: int = 32

    # Halos allow the sliding window to be lossless.
    halo_encoder: int = 8
    halo_decoder: int = 2

    # MAISI VAE spatial scaling factor.
    # Encoder: image space -> latent space, /4
    # Decoder: latent space -> image space, *4
    vae_factor: int = 4

    # H/W size in the space where sliding is performed.
    # Encoder receives CTs after preprocessing: 512 x 512 x 128.
    # Decoder receives latents: 128 x 128 x 32.
    image_size: tuple[int, int, int] = (512, 512, 128)

    # VAE training config
    adv_weight: float = 0.1
    init_std: float = 0.01
    kl_weight: float = 0.02
    patience: int = 100
    perceptual_weight: float = 0.3
    vae_batch_size_train: int = 1
    vae_batch_size_val: int = 4
    vae_checkpoint_path: Path = Path("RESULTS/weights/vae_checkpoint.pt")
    vae_ema_decay: float = 0.999
    vae_epochs: int = 500
    vae_final_weights_path: Path = Path("RESULTS/weights/autoencoder.pt")
    vae_lr: float = 1e-5
    vae_lr_warmup: int = 20
    vae_weight_decay: float = 1e-2
    val_interval: int = 10

    # U-Net training config
    rflow_batch_size: int = 2
    rflow_checkpoint_path: Path = Path("RESULTS/weights/rflow_checkpoint.pt")
    rflow_ema_decay: float = 0.999
    rflow_epochs: int = 5000
    rflow_foundation_weights: Path = Path("RESULTS/weights/diff_unet_3d_rflow-ct.pt")
    rflow_weights_path: Path = Path("RESULTS/weights/diff_unet")
    rflow_lr: float = 1e-5
    rflow_lr_warmup: int = 100
    rflow_save_interval: int = 100
    rflow_weight_decay: float = 1e-2

    # Model configs (custom if needed, otherwise defaults are set in __post_init__)
    vae_config: dict[str, Any] | None = None
    discriminator_config: dict[str, Any] | None = None
    rflow_config: dict[str, Any] | None = None
    scheduler_config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """
        Initialize default model configurations, create output directories,
        and validate basic pipeline settings.

        This method is called automatically by the dataclass immediately after
        object creation.

        It performs three tasks:
        - creates output directories if they do not exist
        - fills missing model/scheduler configs with defaults
        - validates paths and training/inference parameters

        Raises
        ------
        FileNotFoundError
            If required input data paths or model weight files do not exist.

        ValueError
            If settings are invalid.
        """

        self._set_default_vae_config()
        self._set_default_discriminator_config()
        self._set_default_rflow_config()
        self._set_default_scheduler_config()

        self._create_output_dirs()
        self._validate_paths()
        self._validate_settings()

    def _validate_paths(self) -> None:
        """
        Validate the presence of required input files.

        Raises
        ------
        FileNotFoundError
            If CT data, data split JSON, weights directory, VAE weights,
            or rectified flow weights do not exist.
        """

        if not self.data_dict_path.is_file():
            raise FileNotFoundError(f"The required dataset dictionary file was not found at: {self.data_dict_path}")

        if not self.rflow_foundation_weights.is_file():
            raise FileNotFoundError(
                f"The foundational pre-trained model weights checkpoint file "
                f"was not found at: {self.rflow_foundation_weights}"
            )

    def _validate_settings(self) -> None:
        """
        Validate settings values.

        Raises
        ------
        ValueError
            If inference settings are invalid.
        """

        # Spatial settings
        if self.latent_scale <= 0:
            raise ValueError(f"The operational 'latent_scale' must be strictly positive, got: {self.latent_scale}")

        if len(self.spacing) != 3 or any(val <= 0 for val in self.spacing):
            raise ValueError(
                f"The 'spacing' configuration must contain exactly 3 positive dimensions, got: {self.spacing}"
            )

        if self.data_min >= self.data_max:
            raise ValueError(
                f"Intensity threshold 'data_min' must be lower than 'data_max'. "
                f"Got: min={self.data_min}, max={self.data_max}"
            )

        if self.chunk_size_encoder <= 0 or self.chunk_size_decoder <= 0:
            raise ValueError("Sliding window chunk extraction dimensions must be greater than zero")

        if self.halo_encoder < 0 or self.halo_decoder < 0:
            raise ValueError("Sliding window safety margins (halos) cannot be configured with negative boundaries")

        if self.vae_factor <= 0:
            raise ValueError(
                f"The parameter 'vae_factor' transformation scalar must be positive, got: {self.vae_factor}"
            )

        if self.chunk_size_encoder % self.vae_factor != 0:
            raise ValueError(
                f"The configured 'chunk_size_encoder' ({self.chunk_size_encoder}) "
                f"must be evenly divisible by 'vae_factor' ({self.vae_factor})."
            )

        if len(self.image_size) != 3 or any(dim <= 0 for dim in self.image_size):
            raise ValueError(
                f"The structural matrix 'image_size' must contain exactly 3 positive dimensions, got: {self.image_size}"
            )

        if any(dim % self.vae_factor != 0 for dim in self.image_size):
            raise ValueError(
                f"All dimensions specified inside target 'image_size' ({self.image_size}) "
                f"must be cleanly divisible by 'vae_factor' ({self.vae_factor})"
            )

        # Base loop boundaries
        if self.vae_epochs <= 0 or self.rflow_epochs <= 0:
            raise ValueError("Number of epochs must be positive")

        if self.patience <= 0:
            raise ValueError(f"Early stopping 'patience' must be positive, got: {self.patience}")

        if self.val_interval <= 0 or self.rflow_save_interval <= 0:
            raise ValueError("Evaluation intervals and weight saving frequencies must be greater than zero")

        # Batch sizes
        if self.vae_batch_size_train <= 0 or self.vae_batch_size_val <= 0 or self.rflow_batch_size <= 0:
            raise ValueError("All training and validation batch sizes must be strictly positive integers")

        # Learning Rates & Warmup
        if self.vae_lr <= 0 or self.rflow_lr <= 0:
            raise ValueError("Baseline learning rates (LR) must be greater than zero")

        if self.vae_lr_warmup <= 0 or self.rflow_lr_warmup <= 0:
            raise ValueError("Learning rate warmup epochs must be positive numbers")

        if self.vae_lr_warmup >= self.vae_epochs or self.rflow_lr_warmup >= self.rflow_epochs:
            raise ValueError("Learning rate warmup epochs must be less than total epochs")

        if self.vae_weight_decay < 0 or self.rflow_weight_decay < 0:
            raise ValueError("Regularization weight decay scaling penalties cannot be configured below zero")

        # Loss and weight constraints
        if self.adv_weight < 0 or self.kl_weight < 0 or self.perceptual_weight < 0:
            raise ValueError("Loss weights must be mapped to be non-negative")

        if self.init_std <= 0:
            raise ValueError(f"Normal weight initialization standard deviation must be positive, got: {self.init_std}")

        # EMA weight tracking
        if not (0.0 < self.vae_ema_decay < 1.0):
            raise ValueError(f"'vae_ema_decay' must be strictly inside (0.0, 1.0), got: {self.vae_ema_decay}")

        if not (0.0 < self.rflow_ema_decay < 1.0):
            raise ValueError(f"'rflow_ema_decay' must be strictly inside (0.0, 1.0), got: {self.rflow_ema_decay}")

    def _create_output_dirs(self) -> None:
        """
        Create all output directories required by training.
        """

        self.processed_ct_dir_train.mkdir(parents=True, exist_ok=True)
        self.processed_ct_dir_val.mkdir(parents=True, exist_ok=True)
        self.latent_ct_dir_train.mkdir(parents=True, exist_ok=True)
        self.latent_ct_dir_val.mkdir(parents=True, exist_ok=True)
        self.rflow_weights_path.mkdir(parents=True, exist_ok=True)
        self.vae_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def _set_default_vae_config(self) -> None:
        """
        Set default MAISI VAE configuration if no custom config was provided.
        """

        if self.vae_config is None:
            self.vae_config = {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 1,
                "latent_channels": 4,
                "num_channels": [64, 128, 256],
                "num_res_blocks": [2, 2, 2],
                "norm_num_groups": 32,
                "norm_eps": 1e-06,
                "attention_levels": [False, False, False],
                "with_encoder_nonlocal_attn": False,
                "with_decoder_nonlocal_attn": False,
                "use_checkpointing": False,
                "use_convtranspose": False,
                "norm_float16": False,
                "num_splits": 1,
                "dim_split": 0,
                "save_mem": False,
            }

    def _set_default_discriminator_config(self) -> None:
        """
        Set default MAISI discriminator configuration if no custom config
        was provided.
        """

        if self.discriminator_config is None:
            self.discriminator_config = {
                "spatial_dims": 3,
                "channels": 32,
                "in_channels": 1,
                "out_channels": 1,
                "num_layers_d": 3,
                "norm": "INSTANCE",
            }

    def _set_default_rflow_config(self) -> None:
        """
        Set default MAISI rectified flow U-Net configuration if no custom config
        was provided.
        """

        if self.rflow_config is None:
            self.rflow_config = {
                "spatial_dims": 3,
                "in_channels": 8,
                "out_channels": 4,
                "num_channels": [64, 128, 256, 512],
                "attention_levels": [False, False, True, True],
                "num_head_channels": [0, 0, 32, 32],
                "num_res_blocks": 2,
                "use_flash_attention": True,
                "include_top_region_index_input": False,
                "include_bottom_region_index_input": False,
                "include_spacing_input": True,
                "num_class_embeds": 128,
                "resblock_updown": True,
                "include_fc": True,
            }

    def _set_default_scheduler_config(self) -> None:
        """
        Set default scheduler configuration if no custom config was provided.
        """

        if self.scheduler_config is None:
            self.scheduler_config = {
                "num_train_timesteps": 1000,
                "use_discrete_timesteps": False,
                "use_timestep_transform": True,
                "sample_method": "uniform",
                "base_img_size_numel": 128 * 128 * 32,
            }
