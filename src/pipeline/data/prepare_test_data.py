from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

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


def _is_planning_ct_path(path: str) -> bool:
    """
    Check whether a CT path points to the planning CT.
    """

    return "fraction_1_" in Path(path).name


def _extract_all_ct_paths(data_path_list: Sequence[str | dict[str, Any]]) -> list[str]:
    """
    Extract all unique CT paths from a test data list.

    Assumptions:
    - The data list may contain either raw path strings or dictionaries.
    - If dictionaries are used, CT paths are read from "moving_image" and
      "fixed_image" when present.

    Parameters
    ----------
    data_path_list : Sequence[str | dict[str, Any]]
        List of CT paths or dictionaries describing CT pairs.

    Returns
    -------
    ct_paths : list[str]
        List of unique paths pointing to CT images.

    Raises
    ------
    ValueError
        If an item in the data list has an unsupported format.
    """

    ct_paths: list[str] = []
    seen_paths: set[str] = set()

    for item in data_path_list:
        if isinstance(item, str):
            item_paths = [item]

        elif isinstance(item, dict):
            item_paths = [cast(str, item[key]) for key in ("moving_image", "fixed_image") if key in item]
            if len(item_paths) == 0:
                raise ValueError(
                    "Expected dictionary item to contain at least one of 'moving_image' or 'fixed_image'. "
                    f"Got keys: {list(item.keys())}"
                )

        else:
            raise ValueError(f"Each test item must be either a string path or a dictionary. Got: {type(item)}")

        for path in item_paths:
            if path not in seen_paths:
                ct_paths.append(path)
                seen_paths.add(path)

    return ct_paths


def _extract_planning_ct_paths(data_path_list: Sequence[str | dict[str, Any]]) -> list[str]:
    """
    Extract unique planning CT paths from a test data list.
    """

    return [path for path in _extract_all_ct_paths(data_path_list) if _is_planning_ct_path(path)]


def process_and_save_cts(
    config: MaisiTestingConfig,
    data_path_list: Sequence[str | dict[str, Any]],
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

    data_path_list : Sequence[str | dict[str, Any]]
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
        raise ValueError("No CTs were found in the provided data list.")

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


def process_and_save_planning_cts(
    config: MaisiTestingConfig,
    data_path_list: Sequence[str | dict[str, Any]],
    output_dir: Path | None = None,
) -> None:
    """
    Preprocess planning CTs and save them as `.pt` tensors.

    This compatibility wrapper preserves the previous public helper behavior.
    The full pipeline uses :func:`process_and_save_cts` so evaluation has all
    real CT fractions available.
    """

    planning_ct_paths = _extract_planning_ct_paths(data_path_list)

    if len(planning_ct_paths) == 0:
        raise ValueError("No planning CTs were found. Expected filenames containing 'fraction_1_'.")

    process_and_save_cts(
        config=config,
        data_path_list=planning_ct_paths,
        output_dir=output_dir,
    )


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
    """

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
