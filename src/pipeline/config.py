from dataclasses import dataclass
from pathlib import Path


@dataclass
class MaisiTestingConfig:
    """
    Store configuration for the MAISI CT image-to-image testing pipeline.

    This configuration is intended for testing/inference, not training.

    Assumptions:
    - This class lives in a Python file somewhere inside `src/`
    - Full CT data is stored in:
        src/data_full/CT/Patient_x/...
    - Each patient folder contains longitudinal CT fractions:
        Patient_<id>_fraction_1_.nii.gz
        Patient_<id>_fraction_2_.nii.gz
        ...
    - Fraction 1 is treated as the planning CT / conditioning CT
    - The pipeline processes CTs, encodes them into latent representations,
      generates synthetic CT variants, saves outputs, and computes metrics
    - MAISI model weights are stored in:
        src/pipeline/weights/

    Pipeline stages:
    - Prepare CT data for MAISI-compatible inference
    - Encode processed CTs into latent space using the VAE
    - Generate CT variants using the rectified flow model
    - Decode generated latents back into CT space
    - Save generated CTs and evaluation metrics

    Parameters
    ----------
    data_root : Path
        Root directory containing full project data.
        Defaults to:
        src/data_full

    ct_root : Path
        Directory containing patient CT folders.
        Defaults to:
        src/data_full/CT

    data_dict_path : Path
        Path to the data split JSON file.
        Defaults to:
        src/data_full/data_dict.json

    output_root : Path
        Root directory for all MAISI testing outputs.
        Defaults to:
        RESULTS/MAISI_TESTING

    processed_ct_dir : Path
        Directory where preprocessed CT tensors/images are saved.
        Defaults to:
        RESULTS/MAISI_TESTING/processed_ct/test

    latent_ct_dir : Path
        Directory where encoded CT latent representations are saved.
        Defaults to:
        RESULTS/MAISI_TESTING/latents/test

    generated_ct_dir : Path
        Directory where generated CT images are saved.
        Defaults to:
        RESULTS/MAISI_TESTING/generated_ct/test

    metrics_dir : Path
        Directory where metric outputs are saved.
        Defaults to:
        RESULTS/MAISI_TESTING/metrics

    logs_dir : Path
        Directory where logs are saved.
        Defaults to:
        RESULTS/MAISI_TESTING/logs

    weights_dir : Path
        Directory containing MAISI model weights.
        Defaults to:
        src/pipeline/weights

    vae_weight_path : Path
        Path to the MAISI VAE / autoencoder weights.
        Defaults to:
        src/pipeline/weights/autoencoder_v1.pt

    rflow_weight_path : Path
        Path to the MAISI rectified flow U-Net weights.
        Defaults to:
        src/pipeline/weights/diff_unet_3d_rflow-ct.pt

    cts_per_patient : int
        Number of generated CT variants to create per patient.

    steps : int
        Number of sampling steps used during generation.

    overlap_ratio : float
        Overlap ratio used in sliding-window inference.

    sw_batch_size : int
        Sliding-window batch size.

    latent_scale : float
        Scaling factor applied to latent representations.

    device : str
        Device used for inference.
        Usually one of:
        "cuda", "cpu"

    spacing : tuple[float, float, float]
        Target voxel spacing used by the pipeline.

    window_size_encoder : tuple[int, int, int]
        Sliding-window ROI size used during VAE encoding.

    window_size_decoder : tuple[int, int, int]
        Sliding-window ROI size used during VAE decoding.

    vae_config : dict | None
        Model configuration dictionary for the VAE.
        If None, a default MAISI-compatible configuration is used.

    rflow_config : dict | None
        Model configuration dictionary for the rectified flow U-Net.
        If None, a default MAISI-compatible configuration is used.

    scheduler_config : dict | None
        Scheduler configuration dictionary used during sampling.
        If None, a default scheduler configuration is used.

    Raises
    ------
    FileNotFoundError
        If required input paths or model weight files do not exist.

    ValueError
        If selected inference parameters are invalid.

    Notes
    -----
    This class creates output directories automatically in `__post_init__`.

    It does not load models or data by itself. It only stores and validates
    configuration values used by the testing pipeline.
    """

    # Input data
    data_root: Path = Path("src/data_full")
    ct_root: Path = Path("src/data_full/CT")
    data_dict_path: Path = Path("src/data_full/data_dict.json")

    # Data/output paths
    output_root: Path = Path("RESULTS/MAISI_TESTING")

    processed_ct_dir: Path = Path("RESULTS/MAISI_TESTING/processed_ct/test")
    latent_ct_dir: Path = Path("RESULTS/MAISI_TESTING/latents/test")
    generated_ct_dir: Path = Path("RESULTS/MAISI_TESTING/generated_ct/test")
    metrics_dir: Path = Path("RESULTS/MAISI_TESTING/metrics")
    logs_dir: Path = Path("RESULTS/MAISI_TESTING/logs")

    # Model weights
    weights_dir: Path = Path("src/pipeline/weights")

    vae_weight_path: Path = Path("src/pipeline/weights/autoencoder_v1.pt")
    rflow_weight_path: Path = Path("src/pipeline/weights/diff_unet_3d_rflow-ct.pt")

    # Inference config
    cts_per_patient: int = 1
    steps: int = 30
    latent_scale: float = 1.0
    device: str = "cuda"

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
    encoder_factor: int = 4
    decoder_factor: int = 4

    # H/W size in the space where sliding is performed.
    # Encoder receives CTs after preprocessing: 512 x 512 x 128.
    # Decoder receives latents: 128 x 128 x 32.
    encoder_image_size: int = 512
    decoder_image_size: int = 128

    # Model configs (custom if needed, otherwise defaults are set in __post_init__)
    vae_config: dict | None = None
    rflow_config: dict | None = None
    scheduler_config: dict | None = None

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

        if not 0 <= self.overlap_ratio < 1:
            raise ValueError("`overlap_ratio` must be in the range [0, 1)")

        if self.sw_batch_size < 1:
            raise ValueError("`sw_batch_size` must be at least 1")

        if self.latent_scale <= 0:
            raise ValueError("`latent_scale` must be greater than 0")

        if self.device not in {"cuda", "cpu"}:
            raise ValueError("`device` must be either 'cuda' or 'cpu'")

        if len(self.spacing) != 3:
            raise ValueError("`spacing` must contain exactly 3 values")

        if any(value <= 0 for value in self.spacing):
            raise ValueError("All `spacing` values must be greater than 0")

        if len(self.window_size_encoder) != 3:
            raise ValueError("`window_size_encoder` must contain exactly 3 values")

        if len(self.window_size_decoder) != 3:
            raise ValueError("`window_size_decoder` must contain exactly 3 values")

        if any(value <= 0 for value in self.window_size_encoder):
            raise ValueError("All `window_size_encoder` values must be greater than 0")

        if any(value <= 0 for value in self.window_size_decoder):
            raise ValueError("All `window_size_decoder` values must be greater than 0")

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
