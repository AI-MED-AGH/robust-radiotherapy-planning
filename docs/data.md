# DATA

## Data Structure

- **Input pairs**: `(first_fraction_CT, n-th_fraction_CT)` where n ranges from 2 to typically 30.
- **Conditional variable**: First fraction CT scan.
- **Target**: Learn probability distribution of anatomical variants matching real n-th fraction CTs.
- **Data creation**: Generated using `train_test_data_split.py` script.

## Data Organization

### Sample Data Notice

**Important**: The files in the `data/` folder are **downsized samples** from the original dataset due to GitHub size limitations. The sample data demonstrates the repository structure and format but may not be suitable for full model training.

### Data Format and Origin

All data files are stored in **NIfTI format** (`.nii` or `.nii.gz`) and have been converted from original DICOM files:
- **DICOM CT** → NIfTI 3D volumes,
- **RT DOSE** → NIfTI 3D dose maps,
- **RT STRUCT** → NIfTI 3D segmentation masks.

### `DATA/CT/` - Fraction CT Scans

Contains CT images from multiple radiation therapy fractions for each patient:
- **First fraction CTs**: Used as conditional input for generative models (planning CT).
- **N-th fraction CTs** (n = 2 to ~30): Real anatomical variants serving as ground truth.

### `DATA/DOSE/` - 3D Dose Maps

Contains calculated 3D dose distributions:
- **Planned dose map**: Available only for the first (planning) fraction. Voxel-wise radiation dose values in Gy.

### `DATA/STRUCTURES/` - Segmentation Masks

Contains 3D segmentation images for:
- **Organs at Risk (OARs)**: Critical structures to be spared from radiation.
- **Target volumes**: Tumor and planning target volumes (PTV, GTV, CTV).

## Data Splitting

Data splitting is performed through the use of the `train_test_data_split.py` script:

- **Train/Test split**: Per-patient assignment (prevents data leakage). The ratio of train to test is 80/20.
- **Cross-validation**: Train data further split into 5 folds (per-patient).
- **Split assignments**: Generated in `data_dict.json` file.
