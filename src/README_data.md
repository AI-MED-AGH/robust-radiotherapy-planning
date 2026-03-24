
---

## 💾 Data Organization

### Sample Data Notice

⚠️ **Important**: The files in the `data/` folder are **downsized samples** from the original dataset due to GitHub size limitations. The sample data demonstrates the repository structure and format but may not be suitable for full model training.

### Data Format and Origin

All data files are stored in **NIfTI format** (`.nii` or `.nii.gz`) and have been converted from original DICOM files:
- **DICOM CT** → NIfTI 3D volumes
- **RT DOSE** → NIfTI 3D dose maps
- **RT STRUCT** → NIfTI 3D segmentation masks

### `data/CT/` - Fraction CT Scans

Contains CT images from multiple radiation therapy fractions for each patient:
- **First fraction CTs**: Used as conditional input for generative models (planning CT)
- **N-th fraction CTs** (n = 2 to ~30): Real anatomical variants serving as ground truth
- Captures inter-fraction anatomical changes (e.g., tumor shrinkage, organ motion, weight loss)

### `data/DOSE/` - 3D Dose Maps

Contains calculated 3D dose distributions:
- **Planned dose map**: Available only for the first (planning) fraction
- Voxel-wise radiation dose values in Gy
- Converted from RT DOSE DICOM files
- Used to model probability distributions of dose under anatomical uncertainty
- Enables uncertainty quantification in treatment planning

### `data/STRUCTURES/` - Segmentation Masks

Contains 3D segmentation images for:
- **OARs (Organs at Risk)**: Critical structures to be spared from radiation
- **Target volumes**: Tumor and planning target volumes (PTV, GTV, CTV)
- Binary or multi-label masks aligned with corresponding CT scans
- Extracted from RT STRUCT DICOM files

---

## 🔄 DICOM to NIfTI Conversion

### Overview

The `prepareFractionCT.py` script converts DICOM files into NIfTI format for each patient and fraction. This preprocessing step is essential for working with the data in the repository.

### Conversion Pipeline

The script performs the following operations for each fraction:

#### 1. **CT Conversion**
- Reads series of DICOM CT slices
- Assembles slices into a single 3D volume
- Exports as NIfTI file (`.nii` or `.nii.gz`)
- Preserves voxel spacing and orientation metadata

#### 2. **Structure Extraction**
- Parses RT STRUCT DICOM files
- Extracts OARs of interest (e.g., bladder, rectum, femoral heads)
- Extracts target volumes (tumor, PTV, GTV, CTV)
- Creates corresponding 3D binary segmentation masks
- Saves each structure as separate NIfTI file

#### 3. **Target Naming Convention**
- **Challenge**: RT STRUCT files lack consistent naming conventions for target volumes
- **Solution**: Target names are read from an external reference file
- **Example**: `prostaty.txt` contains the specific target naming used in the database
- Ensures correct identification across different patients and institutions

#### 4. **Dose Map Conversion**
- Converts RT DOSE DICOM (internally 3D) to NIfTI format
- **Note**: Planned dose map available only for the **first (planning) fraction**
- Aligns dose grid with corresponding CT image geometry

### Usage

```bash
# Convert DICOM files to NIfTI format
python code/prepareFractionCT.py 
```

### Target Naming File Format

The target naming file (e.g., `prostaty.txt`) should contain patient ids and associated target name e.g.:
```
1 PTV
2 CTV
3 GTV_Primary
```

This ensures consistent identification of target structures across the dataset despite varying naming conventions in RT STRUCT files.

---
