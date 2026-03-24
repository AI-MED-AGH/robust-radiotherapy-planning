---

## 📋 Code Overview

The project focuses on generating **plausible anatomical variants** using CT scans from different fractions of radiation therapy. The models learn to predict realistic anatomical changes conditioned on the first fraction CT scan.

### Data Structure

- **Input pairs**: `(first_fraction_CT, n-th_fraction_CT)` where n ranges from 2 to typically 30
- **Conditional variable**: First fraction CT scan
- **Target**: Learn probability distribution of anatomical variants matching real n-th fraction CTs
- **Data creation**: Generated using `train_test_data_split.py` script

### Data Splitting

- ✅ **Train/Test split**: Per-patient assignment (prevents data leakage)
- ✅ **Cross-validation**: Train data further split into 5 folds (per-patient)
- 📄 **Split assignments**: Stored in `data_dict.json` file

---

## 🔬 Approaches

### 1. Dense Deformation Field (DDF) Approach 📁 `ddf/`

This approach directly models **dense deformation fields** that transform moving images into fixed images.

#### Training Pipeline
1. Calculate deformation fields between moving and fixed images using **SimpleITK**
2. Train a **U-Net with MC Dropout** where:
   - **Input**: First fraction CT
   - **Output**: Dense deformation field (DDF)

#### Inference Pipeline
1. Keep network in training mode to activate **MC Dropout**
2. Generate multiple DDF predictions for the same first fraction CT input
3. Apply DDFs to warp the first fraction CT → **plausible anatomical variants**
4. Use DDFs with corresponding variants to sample from the **estimated probability distribution of 3D dose maps**

**Key advantage**: Direct modeling of transformations with uncertainty quantification via MC Dropout

---

### 2. Stable Diffusion Approach 📁 `stable_diffusion/`

This approach leverages latent diffusion models for anatomical variant generation.

#### Training Pipeline
1. **VAE encoding**: Compress CTs into latent representations
2. Train **Conditional Denoising Diffusion (CDD)** model with first fraction CT as conditioning variable

#### Generation Pipeline
1. Encode first fraction CT to latent space using VAE
2. Use latent representation to condition the **CDD model**
3. Decode CDD output with VAE → **anatomical variant**
4. Calculate DDF between first fraction CT and generated variant
5. Use combined data to sample from **estimated probability distribution of 3D dose maps**

**Key advantage**: Powerful latent space modeling with state-of-the-art diffusion techniques

---

## 📂 Repository Structure

```
.
├── ddf/                          # Dense Deformation Field approach
├── stable_diffusion/             # Stable Diffusion approach  
├── train_test_data_split.py     # Data preparation script
├── data_dict.json                # Train/test/fold assignments
└── README.md                     # This file
```

---

## 🎯 Applications

Both approaches enable:
- ✨ Generation of **realistic anatomical variants** for radiation therapy planning
- 📊 Sampling from **probability distributions of 3D dose maps**
- 🔮 **Uncertainty quantification** in anatomical changes across treatment fractions

---

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- SimpleITK
- PyTorch

### Data Preparation
```bash
python train_test_data_split.py
```

This creates the paired CT data and generates `data_dict.json` with train/test/fold assignments.

---

