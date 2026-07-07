from pathlib import Path
from typing import Any

import torch
from monai.transforms import (  # type: ignore[attr-defined]
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    SpatialPadd,
)
from tqdm import tqdm

from src.data_utils import create_data_split_dict
from src.pipeline.config import MaisiTestingConfig
from src.pipeline.helpers.cleanup import clear_directory_contents
from src.pipeline.helpers.helpers import _extract_all_ct_paths


def get_ct_preprocessing_transform(config: MaisiTestingConfig) -> Compose:
    """
    Create preprocessing transform for MAISI planning CT inference.

    The transform:
    - loads a CT image from disk
    - ensures channel-first format
    - reorients image to RAS
    - clips and scales HU values from [data_min, data_max] to [0, 1]
    - pads image to a fixed spatial size
    - center-crops image to a fixed spatial size
    - converts the output to a torch float32 tensor

    Returns
    -------
    transform : Compose
        MONAI preprocessing transform for CT images.
    """

    return Compose(
        [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes="RAS", labels=None),
            ScaleIntensityRanged(
                keys="image",
                a_min=config.data_min,
                a_max=config.data_max,
                b_min=0,
                b_max=1,
                clip=True,
            ),
            SpatialPadd(
                keys="image",
                spatial_size=config.target_image_size,
                mode="constant",
                constant_values=0,
            ),
            CenterSpatialCropd(
                keys="image",
                roi_size=config.target_image_size,
            ),
            EnsureTyped(keys="image", dtype=torch.float32),
        ]
    )


def process_and_save_cts(
    config: MaisiTestingConfig,
    data_path_list: list[str] | list[dict[str, Any]],
    output_dir: Path | None = None,
) -> None:
    """
    Preprocess all CTs and save them as `.pt` tensors.

    Assumptions:
    - Output tensors are saved as `.pt` files.
    - Saved planning CT tensors are later used by the MAISI VAE encoder.
    - Saved non-planning CT tensors are used as real CTs during evaluation.

    Parameters
    ----------
    config : MaisiTestingConfig
        Configuration object containing pipeline paths and settings.

    data_path_list : list[str] | list[dict[str, Any]]
        List of CT paths or dictionary items from the test split.
        Supported formats:
            "path/to/Patient_1_fraction_2_.nii.gz"

        or:
            {
                "moving_image": "path/to/Patient_1_fraction_1_.nii.gz",
                "fixed_image": "path/to/Patient_1_fraction_2_.nii.gz"
            }

    output_dir : Path | None
        Directory where processed CT tensors should be saved.
        If None, defaults to:
            config.processed_ct_dir

    Raises
    ------
    FileNotFoundError
        If one of the CT files does not exist.

    ValueError
        If no CTs are found in the provided data list.
    """

    if output_dir is None:
        output_dir = config.processed_ct_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    transform = get_ct_preprocessing_transform(config)
    ct_paths = _extract_all_ct_paths(data_path_list)

    if len(ct_paths) == 0:
        raise ValueError("No CTs were found in the provided data list")

    for path_raw in tqdm(ct_paths, desc="Preparing CTs"):
        path = Path(path_raw)

        if not path.exists():
            raise FileNotFoundError(f"CT file does not exist: {path}")

        transformed = transform({"image": str(path)})
        tensor_data = transformed["image"]

        original_name = path.name.replace(".nii.gz", ".pt")
        save_path = output_dir / original_name

        clean_tensor = tensor_data.as_tensor().clone().detach()
        torch.save(clean_tensor, save_path)


def prepare_test_data(
    config: MaisiTestingConfig,
    seed: int = 42,
    train_fraction: float = 0.8,
    folds: int = 5,
) -> None:
    """
    Prepare MAISI evaluation data from the configured split.

    This function:
    - creates or loads the longitudinal CT train/validation/test split
    - selects the configured test or validation split
    - extracts all CTs from the selected split
    - clears config.processed_ct_dir
    - preprocesses all CTs
    - saves processed CT tensors to config.processed_ct_dir

    Assumptions:
    - Data is stored in:
        src/data_full/CT/Patient_x/...
    - Each patient folder contains:
        Patient_<id>_fraction_1_.nii.gz
        Patient_<id>_fraction_2_.nii.gz
        ...
    - Fraction 1 is the planning CT.
    - Planning CT is used as the condition image for MAISI generation.
    - All CT fractions are saved so evaluation can compare generated CTs
      against real non-planning CTs.
    - The output directory is:
        config.processed_ct_dir

    Parameters
    ----------
    config : MaisiTestingConfig
        Configuration object containing data paths, output paths, model paths,
        and inference settings.

    seed : int
        Random seed used for creating train/validation/test splits.

    train_fraction : float
        Fraction of patients used for train+validation pool.
        Remaining patients go to test set.

    folds : int
        Number of cross-validation folds.

    Raises
    ------
    KeyError
        If the generated data dictionary does not contain the requested split.

    ValueError
        If the requested split is empty or invalid.

    FileNotFoundError
        If required CT files do not exist.

    OSError
        If the processed CT output directory cannot be cleared.
    """

    clear_directory_contents(config.processed_ct_dir)

    data_dict = create_data_split_dict(
        seed=seed,
        train_fraction=train_fraction,
        folds=folds,
        data_root=config.ct_root,
        output_json=config.data_dict_path,
        save=False,
    )

    if config.evaluation_split == "test":
        if "test" not in data_dict:
            raise KeyError("The data dictionary does not contain a 'test' split")

        data_path_list = data_dict["test"]

    else:
        if config.validation_fold not in data_dict:
            raise KeyError(f"The data dictionary does not contain validation fold {config.validation_fold}")

        fold_dict = data_dict[config.validation_fold]

        if "val" not in fold_dict:
            raise KeyError(f"Validation fold {config.validation_fold} does not contain a 'val' split")

        data_path_list = fold_dict["val"]

    if len(data_path_list) == 0:
        raise ValueError(
            f"The {config.evaluation_split} split is empty. Cannot prepare MAISI {config.evaluation_split} data"
        )

    process_and_save_cts(
        config=config,
        output_dir=config.processed_ct_dir,
        data_path_list=data_path_list,
    )
