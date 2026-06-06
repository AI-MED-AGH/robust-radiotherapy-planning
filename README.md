# ROBUST-RADIOTHERAPY-PLANNING

![Python](https://img.shields.io/badge/Python-3.11.15-lightgray?style=flat&logo=python)
![Status](https://img.shields.io/badge/Status-In_Progress-blue)
![License](https://img.shields.io/badge/License-MIT-green)

This repository contains code for training and testing Flow Matching (FM) models that predict anatomical variations across radiation therapy fractions, along with their impact on 3D dose distributions. The models are conditioned on first fraction CT scans and generate plausible anatomical variants that match the distribution of real anatomical changes observed in subsequent treatment fractions.

## Contributors

This project is developed by a team of students:

| Name | GitHub | Role |
| :--- | :--- | :--- |
| **Wiktoria Arendarczyk** | [![GitHub](https://img.shields.io/badge/-MagicWiqqu-181717?style=flat&logo=github)](https://github.com/MagicWiqqu) | Lead Researcher|
| **Zuzanna Deszcz** | [![GitHub](https://img.shields.io/badge/-melliegrant-181717?style=flat&logo=github)](https://github.com/melliegrant) | Researcher |
| **Cezary Moskal** | [![GitHub](https://img.shields.io/badge/-Couch--bit-181717?style=flat&logo=github)](https://github.com/Couch-bit) | Researcher |

## Applications

This framework enables:

1. **Robust treatment planning**: Account for anatomical variations in dose optimization.
2. **Adaptive radiotherapy**: Predict anatomical changes for plan adaptation.
3. **Uncertainty quantification**: Quantify dose uncertainty due to anatomical variations.
4. **Quality assurance**: Validate delivered dose against predicted variations.
5. **Research**: Study inter-fraction anatomical changes and their dosimetric impact.

## Acknowledgments & Licensing

The source code in this repository is licensed under the [MIT License](LICENSE). 

This project utilizes model configurations and pre-trained weights derived from [NV-Generate-CT](https://huggingface.co/nvidia/NV-Generate-CT). To run some parts of the code, you must download these assets directly from the original source and place them in your local directory.

Please note that these external assets are licensed by NVIDIA Corporation under the **NVIDIA Open Model License**.
