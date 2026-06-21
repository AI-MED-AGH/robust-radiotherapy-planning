import json
from pathlib import Path
from typing import Any, cast

import torch
from monai.data import MetaTensor  # type: ignore[attr-defined]
from monai.transforms import Compose  # type: ignore[attr-defined]
from tqdm import tqdm

from src.training.helpers.config import MaisiTrainingConfig
from src.training.helpers.data_loading import get_ct_preprocessing_transform


def process_ct(path: Path, transform: Compose, config: MaisiTrainingConfig, validation: bool) -> None:
    """
    Process a CT scan and save it to a directory specified in the training config.

    Parameters
    ----------
    path : Path
        The path to the original file in NIfTI format.

    transform : Compose
        The transform to perform on the given image.

    config : MaisiTrainingConfig
        Training configuration object.

    validation : bool
        Whether validation data is being processed. In that case data is saved in its own directory.

    Raises
    ------
    FileNotFoundError
        If the specified `path` does not exist on disk.

    ValueError
        If the target file does not have a '.nii.gz' extension.

    KeyError
        If the 'image' key is missing from the output dictionary after applying the
        transform pipeline.
    """

    if not path.exists():
        raise FileNotFoundError(f"The provided path does not exist: {path}")

    if not path.name.endswith(".nii.gz"):
        raise ValueError(f"Expected a '.nii.gz' file, but received: '{path.name}'")

    transformed = transform({"image": str(path)})

    if "image" not in transformed:
        raise KeyError(
            "The key 'image' was removed or is missing from the dictionary "
            "after applying the transform compose pipeline"
        )

    tensor_data = cast(MetaTensor, transformed["image"])

    original_name = path.name.replace(".nii.gz", ".pt")
    if validation:
        save_path = config.processed_ct_dir_val / original_name
    else:
        save_path = config.processed_ct_dir_train / original_name

    clean_tensor = tensor_data.as_tensor().clone().detach()
    torch.save(clean_tensor, save_path)


def process_and_save_cts(
    config: MaisiTrainingConfig,
) -> None:
    """
    Load the dataset dictionary, initialize preprocessing transforms,
    and process both training and validation CT collections.

    Parameters
    ----------
    config : MaisiTrainingConfig
        Training configuration object containing data paths and directory settings.

    Raises
    ------
    FileNotFoundError
        If the data dictionary file specified in `config.data_dict_path` does not exist.

    json.JSONDecodeError
        If the data dictionary file contains invalid JSON syntax.

    KeyError
        If the loaded dictionary does not contain the expected structure
        (['0']['train'] and ['0']['val']).
    """

    if not config.data_dict_path.exists():
        raise FileNotFoundError(f"The data dictionary file was not found at: {config.data_dict_path}")

    # Handle potential JSON reading/parsing errors
    try:
        with open(config.data_dict_path) as file:
            ct_paths = cast(dict[str, Any], json.loads(file.read()))
    except json.JSONDecodeError as err:
        raise ValueError(f"Failed to parse the data dictionary JSON file at {config.data_dict_path}") from err

    # Validate the dictionary structure before starting long-running loops
    if "0" not in ct_paths or "train" not in ct_paths["0"] or "val" not in ct_paths["0"]:
        raise KeyError(
            f"The dataset dictionary at {config.data_dict_path} is missing the required nested structure "
            f"with keys: ['0']['train'] and ['0']['val']"
        )

    transform = get_ct_preprocessing_transform(config)

    for path in tqdm(ct_paths["0"]["train"], desc="Processing training data"):
        process_ct(Path(path), transform, config, False)

    for path in tqdm(ct_paths["0"]["val"], desc="Processing validation data"):
        process_ct(Path(path), transform, config, True)


def main() -> None:
    """
    The main entry point for processing CT scans from NIfTI format to VAE training ready .pt files.

    The only dependency this has is the presence of the full dataset in its directory
    along with the data split dictionary.

    The script should be run from the repository root using:

        python -m src.training.prepare_data
    """
    config = MaisiTrainingConfig()

    process_and_save_cts(config)


if __name__ == "__main__":
    main()
