import re
from pathlib import Path

import torch
from torch.utils.data import Dataset


class PatientFractionPairDataset(Dataset[dict[str, torch.Tensor]]):
    """
    A dataset class for loading pairs of patient fraction images from preprocessed PyTorch tensors.

    This dataset scans a directory for files following the pattern 'Patient_<id>_fraction_<id>_.pt',
    categorizes fraction 1 as the reference/condition image, and pairs it with all subsequent
    fractions (fraction X) for the same patient as target images.

    Parameters
    ----------
    data_dir : str | Path
        Path to the directory containing the preprocessed tensor files.

    Raises
    ------
    FileNotFoundError
        If the specified `data_dir` does not exist.

    ValueError
        If no valid patient fraction pairs are discovered in the directory.
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"The provided data directory does not exist: {self.data_dir}")

        self.pairs = []

        # Regex to capture patient_id and fraction_id
        filename_pattern = re.compile(r"Patient_(\d+)_fraction_(\d+)_\.pt")

        fraction_ones = {}
        other_fractions = []

        # Scan the directory and categorize the files
        for file_path in self.data_dir.glob("Patient_*_fraction_*_.pt"):
            match = filename_pattern.match(file_path.name)
            if match:
                patient_id, fraction_id = match.groups()

                if fraction_id == "1":
                    fraction_ones[patient_id] = file_path
                else:
                    other_fractions.append((patient_id, file_path))

        # Create pairs: (fraction_X, fraction_1)
        for patient_id, file_path in other_fractions:
            if patient_id in fraction_ones:
                # Store as strings since PyTorch DataLoader collates strings seamlessly
                self.pairs.append((str(file_path), str(fraction_ones[patient_id])))

        if not self.pairs:
            raise ValueError(f"No valid patient fraction pairs could be constructed from '{self.data_dir}'")

    def __len__(self) -> int:
        """
        Return the total number of paired fraction samples available.

        Returns
        -------
        int
            The number of pairs in the dataset.
        """

        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Fetch and load the target and condition tensor pair at the specified index.

        Parameters
        ----------
        idx : int
            The index of the pair to retrieve.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing:
            - "target": The tensor data for fraction X.
            - "condition": The tensor data for reference fraction 1.

        Raises
        ------
        IndexError
            If the index is out of bounds for the dataset size.

        RuntimeError
            If loading either tensor file fails due to internal file corruption or disk I/O errors.
        """

        try:
            target_path, condition_path = self.pairs[idx]
        except IndexError as err:
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self.pairs)}") from err

        try:
            target = torch.load(target_path, map_location="cpu", weights_only=True)
        except Exception as err:
            raise RuntimeError(f"Failed to load target tensor file from path: {target_path}") from err

        try:
            condition = torch.load(condition_path, map_location="cpu", weights_only=True)
        except Exception as err:
            raise RuntimeError(f"Failed to load condition tensor file from path: {condition_path}") from err

        return {
            "target": target,
            "condition": condition,
        }
