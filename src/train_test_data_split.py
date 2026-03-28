import glob
import json
import os

import numpy as np

# Config
SEED = 42

DATA_DIR = "DATA/CT/"
SAVE_PATH = "DATA/data_dict.json"

TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.2
FOLDS = 5

# Use the indexes in the CT folder as source of truth
data_dir = "DATA/CT/"
data_files = sorted(glob.glob(f"{data_dir}*/Patient_*"))
ids = [os.path.basename(i).split("_")[1] for i in data_files]

# The shuffling is always done with SEED=42 for reproducibility
SEED = 42
np.random.seed(SEED)
np.random.shuffle(ids)

# Split the dataset into train and test data
train_ids = ids[: int(len(ids) * TRAIN_FRACTION)]
test_ids = ids[int(len(ids) * TRAIN_FRACTION) :]

# Define size of validation data
val_num = int(len(train_ids) * VAL_FRACTION)

# Create cross-validation with 5 folds
data_dict: dict = {}
for fold in range(FOLDS):
    val_ids = train_ids[fold * val_num : min((fold + 1) * val_num, len(train_ids))]

    train_files: list[dict[str, str]] = []
    for i in train_ids:
        if i in val_ids:
            continue
        dname = f"{data_dir}/Patient_{i}/"
        fraction_names = [
            name
            for name in sorted(glob.glob(f"{dname}/*.nii.gz"))
            if "_fraction_1_.nii.gz" not in name
        ]
        first_fraction_name = f"{data_dir}/Patient_{i}/Patient_{i}_fraction_1_.nii.gz"
        data_dirs = [
            {"moving_image": first_fraction_name, "fixed_image": n}
            for n in fraction_names
        ]
        train_files.extend(data_dirs)

    val_files = []
    for i in val_ids:
        dname = f"{data_dir}/Patient_{i}/"
        fraction_names = [
            name
            for name in sorted(glob.glob(f"{dname}/*.nii.gz"))
            if "_fraction_1_.nii.gz" not in name
        ]
        first_fraction_name = f"{data_dir}/Patient_{i}/Patient_{i}_fraction_1_.nii.gz"
        data_dirs = [
            {"moving_image": first_fraction_name, "fixed_image": n}
            for n in fraction_names
        ]
        val_files.extend(data_dirs)

    data_dict[fold] = {}
    data_dict[fold]["train"] = train_files
    data_dict[fold]["val"] = val_files

# Extract the test data
test_files = []
for i in test_ids:
    dname = f"{data_dir}/Patient_{i}/"
    fraction_names = [
        name
        for name in sorted(glob.glob(f"{dname}/*.nii.gz"))
        if "_fraction_1_.nii.gz" not in name
    ]
    first_fraction_name = f"{data_dir}/Patient_{i}/Patient_{i}_fraction_1_.nii.gz"
    data_dirs = [
        {"moving_image": first_fraction_name, "fixed_image": n} for n in fraction_names
    ]
    test_files.extend(data_dirs)

data_dict["test"] = test_files

# Save the result as JSON
with open(SAVE_PATH, "w") as f:
    json.dump(data_dict, f, indent=4)
