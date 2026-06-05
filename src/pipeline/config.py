from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class MaisiTestingConfig:
    """
    Store configuration for the MAISI CT image-to-image testing pipeline.

    This configuration is intended for testing and inference only.

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
    - The pipeline preprocesses CTs, encodes them into latent
    representations, generates synthetic CT variants using MAISI,
    decodes generated latents back into image space, and evaluates
    the generated outputs.
    - MAISI model weights are stored in:
        src/pipeline/weights/

    Pipeline stages
    ---------------
    1. Prepare CT data for MAISI inference.
    2. Encode CTs into latent space using the VAE.
    3. Generate latent representations using the rectified flow model.
    4. Decode generated latents into CT images.
    5. Save generated outputs and evaluation metrics.

    Parameters
    ----------
    data_root : Path
        Root directory containing project data.

    ct_root : Path
        Directory containing patient CT folders.

    data_dict_path : Path
        Path to the dataset split JSON file.

    output_root : Path
        Root directory for all pipeline outputs.

    processed_ct_dir : Path
        Directory containing preprocessed CT data.

    latent_ct_dir : Path
        Directory containing encoded latent representations.

    generated_ct_dir : Path
        Directory containing generated CT images.

    metrics_dir : Path
        Directory containing evaluation metrics.

    logs_dir : Path
        Directory containing pipeline logs.

    weights_dir : Path
        Directory containing MAISI model weights.

    vae_weight_path : Path
        Path to the MAISI VAE checkpoint.

    rflow_weight_path : Path
        Path to the MAISI rectified flow checkpoint.

    cts_per_patient : int
        Number of synthetic CTs generated per patient.

    steps : int
        Number of rectified-flow sampling steps.

    latent_scale : float
        Scaling factor applied to latent representations..

    spacing : tuple[float, float, float]
        Target voxel spacing used during preprocessing.

    data_min : float
        Minimum CT intensity value used for normalization and clipping.

    data_max : float
        Maximum CT intensity value used for normalization and clipping.

    chunk_size_encoder : int
        Core sliding-window chunk size used during VAE encoding.

    chunk_size_decoder : int
        Core sliding-window chunk size used during VAE decoding.

    halo_encoder : int
        Halo size added around encoder chunks to remove boundary artifacts.

    halo_decoder : int
        Halo size added around decoder chunks to remove boundary artifacts.

    encoder_decoder_factor : int
        Spatial scaling factor of the VAE encoder.
        Converts image space to latent space.

    encoder_image_size : int
        Height and width of the encoder input image space on which
        sliding-window inference is performed.

    decoder_image_size : int
        Height and width of the decoder latent space on which
        sliding-window inference is performed.

    vae_config : dict | None
        VAE architecture configuration.
        If None, a default MAISI configuration is used.

    rflow_config : dict | None
        Rectified flow U-Net configuration.
        If None, a default MAISI configuration is used.

    scheduler_config : dict | None
        Sampling scheduler configuration.
        If None, a default configuration is used.

    validate_paths : bool
        Whether filesystem paths should be validated during initialization.

    Raises
    ------
    FileNotFoundError
        If required input data or model checkpoints are missing.

    ValueError
        If inference parameters are invalid.

    Notes
    -----
    The pipeline uses a custom sliding-window implementation based on
    chunk extraction with halo overlap. The depth dimension is processed
    as a whole, while sliding is performed only across height and width.

    Output directories are automatically created during initialization.
    This class stores and validates configuration values only and does
    not load data or models.
    """

    # Data paths
    data_root: Path = Path("src/data_full")
    ct_root: Path = data_root / "CT"
    data_dict_path: Path = data_root / "data_dict.json"

    # Output paths
    output_root: Path = Path("RESULTS/MAISI_TESTING")

    processed_ct_dir: Path = output_root / "processed_ct" / "test"
    latent_ct_dir: Path = output_root / "latents" / "test"
    generated_ct_dir: Path = output_root / "generated_ct" / "test"
    metrics_dir: Path = output_root / "metrics"
    logs_dir: Path = output_root / "logs"

    # Model weights
    weights_dir: Path = Path("src/pipeline/weights")

    vae_weight_path: Path = weights_dir / "autoencoder.pt"
    rflow_weight_path: Path = weights_dir / "diff_unet.pt"

    # Inference config
    cts_per_patient: int = 1
    steps: int = 30
    latent_scale: float = 1.0
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # MAISI-specific config
    spacing: tuple[float, float, float] = (1.171875, 1.171875, 3.0)

    # CT intensity normalization range
    data_min: float = -1000.0
    data_max: float = 1500.0

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
    encoder_decoder_factor: int = 4

    # H/W size in the space where sliding is performed.
    # Encoder receives CTs after preprocessing: 512 x 512 x 128.
    # Decoder receives latents: 128 x 128 x 32.
    encoder_image_size: int = 512
    decoder_image_size: int = 128

    # Model configs (custom if needed, otherwise defaults are set in __post_init__)
    vae_config: dict[str, Any] | None = None
    rflow_config: dict[str, Any] | None = None
    scheduler_config: dict[str, Any] | None = None

    # Validation settings
    validate_paths: bool = True

    def __post_init__(self) -> None:
        """
        Initialize default model configurations, create output directories,
        and validate basic pipeline settings.

        This method is called automatically by the dataclass immediately after
        object creation.

        It performs three tasks:
        - creates output directories if they do not exist
        - fills missing model/scheduler configs with defaults
        - validates paths and inference parameters

        Raises
        ------
        FileNotFoundError
            If required input data paths or model weight files do not exist.

        ValueError
            If inference settings are invalid.
        """

        self._set_default_vae_config()
        self._set_default_rflow_config()
        self._set_default_scheduler_config()

        self._create_output_dirs()
        if self.validate_paths:
            self._validate_paths()
        self._validate_inference_settings()

    def _create_output_dirs(self) -> None:
        """
        Create all output directories required by the testing pipeline.
        """

        self.output_root.mkdir(parents=True, exist_ok=True)
        self.processed_ct_dir.mkdir(parents=True, exist_ok=True)
        self.latent_ct_dir.mkdir(parents=True, exist_ok=True)
        self.generated_ct_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _validate_paths(self) -> None:
        """
        Validate required input data paths and model weight paths.

        Raises
        ------
        FileNotFoundError
            If CT data, data split JSON, weights directory, VAE weights,
            or rectified flow weights do not exist.
        """

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root does not exist: {self.data_root}")

        if not self.ct_root.exists():
            raise FileNotFoundError(f"CT root does not exist: {self.ct_root}")

        if not self.data_dict_path.exists():
            raise FileNotFoundError(f"Data split JSON does not exist: {self.data_dict_path}")

        if not self.weights_dir.exists():
            raise FileNotFoundError(f"Weights directory does not exist: {self.weights_dir}")

        if not self.vae_weight_path.exists():
            raise FileNotFoundError(f"VAE weight file does not exist: {self.vae_weight_path}")

        if not self.rflow_weight_path.exists():
            raise FileNotFoundError(f"Rectified flow weight file does not exist: {self.rflow_weight_path}")

    def _validate_inference_settings(self) -> None:
        """
        Validate basic inference and image-processing settings.

        Raises
        ------
        ValueError
            If inference settings are invalid.
        """

        if self.cts_per_patient < 1:
            raise ValueError("`cts_per_patient` must be at least 1")

        if self.steps < 1:
            raise ValueError("`steps` must be at least 1")

        if self.latent_scale <= 0:
            raise ValueError("`latent_scale` must be greater than 0")

        if self.device not in {"cuda", "cpu"}:
            raise ValueError("`device` must be either 'cuda' or 'cpu'")

        if len(self.spacing) != 3:
            raise ValueError("`spacing` must contain exactly 3 values")

        if any(value <= 0 for value in self.spacing):
            raise ValueError("All `spacing` values must be greater than 0")

        if self.data_min >= self.data_max:
            raise ValueError(
                f"`data_min` must be smaller than `data_max`. Got data_min={self.data_min}, data_max={self.data_max}"
            )

        if self.chunk_size_encoder <= 0:
            raise ValueError("`chunk_size_encoder` must be greater than 0")

        if self.chunk_size_decoder <= 0:
            raise ValueError("`chunk_size_decoder` must be greater than 0")

        if self.halo_encoder < 0:
            raise ValueError("`halo_encoder` must be non-negative")

        if self.halo_decoder < 0:
            raise ValueError("`halo_decoder` must be non-negative")

        if self.encoder_decoder_factor <= 0:
            raise ValueError("`encoder_decoder_factor` must be greater than 0")

        if self.encoder_image_size <= 0:
            raise ValueError("`encoder_image_size` must be greater than 0")

        if self.decoder_image_size <= 0:
            raise ValueError("`decoder_image_size` must be greater than 0")

        if self.encoder_image_size % self.chunk_size_encoder != 0:
            raise ValueError(
                "`encoder_image_size` must be divisible by `chunk_size_encoder`. "
                f"Got encoder_image_size={self.encoder_image_size}, "
                f"chunk_size_encoder={self.chunk_size_encoder}"
            )

        if self.decoder_image_size % self.chunk_size_decoder != 0:
            raise ValueError(
                "`decoder_image_size` must be divisible by `chunk_size_decoder`. "
                f"Got decoder_image_size={self.decoder_image_size}, "
                f"chunk_size_decoder={self.chunk_size_decoder}"
            )

        if self.chunk_size_encoder % self.encoder_decoder_factor != 0:
            raise ValueError(
                "`chunk_size_encoder` must be divisible by `encoder_decoder_factor`. "
                f"Got chunk_size_encoder={self.chunk_size_encoder}, "
                f"encoder_decoder_factor={self.encoder_decoder_factor}"
            )

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
                "base_img_size_numel": 64 * 64 * 16,
            }
