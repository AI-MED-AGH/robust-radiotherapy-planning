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


def get_ct_preprocessing_transform(config: MaisiTestingConfig) -> Compose:
    """
    Create preprocessing transform for MAISI planning CT inference.

    The transform:
    - loads a CT image from disk
    - ensures channel-first format
    - reorients image to RAS
    - clips and scales HU values from [-1000, 1000] to [0, 1]
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


def _extract_planning_ct_paths(data_path_list: list[str | dict[str, Any]]) -> list[str]:
    """
    Extract planning CT paths from a test data list.

    Assumptions:
    - Planning CT is fraction 1.
    - The data list may contain either raw path strings or dictionaries.
    - If dictionaries are used, the planning CT path is expected under
      the key "moving_image".

    Parameters
    ----------
    data_path_list : list[str | dict[str, Any]]
        List of CT paths or dictionaries describing CT pairs.

    Returns
    -------
    planning_ct_paths : list[str]
        List of paths pointing to planning CT images.

    Raises
    ------
    ValueError
        If an item in the data list has an unsupported format.
    """

    planning_ct_paths = []

    for item in data_path_list:
        if isinstance(item, str):
            path = item

        elif isinstance(item, dict):
            if "moving_image" not in item:
                raise ValueError(
                    f"Expected dictionary item to contain key 'moving_image'. Got keys: {list(item.keys())}"
                )

            path = item["moving_image"]

        else:
            raise ValueError(f"Each test item must be either a string path or a dictionary. Got: {type(item)}")

        if "fraction_1_" in path:
            planning_ct_paths.append(path)

    return planning_ct_paths


def process_and_save_planning_cts(
    config: MaisiTestingConfig,
    data_path_list: list[str | dict[str, Any]],
    output_dir: Path | None = None,
) -> None:
    """
    Preprocess planning CTs and save them as `.pt` tensors.

    Assumptions:
    - Planning CTs are identified by "fraction_1_" in the filename.
    - Only planning CTs are currently used as the condition image.
    - Output tensors are saved as `.pt` files.
    - Saved tensors are later used by the MAISI VAE encoder.

    Parameters
    ----------
    config : MaisiTestingConfig
        Configuration object containing pipeline paths and settings.

    data_path_list : list[str | dict[str, Any]]
        List of CT paths or dictionary items from the test split.
        Supported formats:
            "path/to/Patient_1_fraction_1_.nii.gz"

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
        If one of the planning CT files does not exist.

    ValueError
        If no planning CTs are found in the provided data list.
    """

    if output_dir is None:
        output_dir = config.processed_ct_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    transform = get_ct_preprocessing_transform(config)
    planning_ct_paths = _extract_planning_ct_paths(data_path_list)

    if len(planning_ct_paths) == 0:
        raise ValueError("No planning CTs were found. Expected filenames containing 'fraction_1_'.")

    for path_raw in tqdm(planning_ct_paths, desc="Preparing planning CTs"):
        path = Path(path_raw)

        if not path.exists():
            raise FileNotFoundError(f"Planning CT file does not exist: {path}")

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
    Prepare MAISI testing data from the test split.

    This function:
    - creates or loads the longitudinal CT train/validation/test split
    - selects the test split
    - extracts planning CTs from the test set
    - preprocesses planning CTs
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
        If the generated data dictionary does not contain the "test" key.

    ValueError
        If the test split is empty or invalid.

    FileNotFoundError
        If required CT files do not exist.
    """

    data_dict = create_data_split_dict(
        seed=seed,
        train_fraction=train_fraction,
        folds=folds,
        data_root=config.ct_root,
        output_json=config.data_dict_path,
        save=False,
    )

    if "test" not in data_dict:
        raise KeyError("The data dictionary does not contain a 'test' split")

    if len(data_dict["test"]) == 0:
        raise ValueError("The test split is empty. Cannot prepare MAISI test data")

    process_and_save_planning_cts(
        config=config,
        output_dir=config.processed_ct_dir,
        data_path_list=data_dict["test"],
    )
