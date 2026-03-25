# ROBUST-RADIOTHERAPY-PLANNING

![Python](https://img.shields.io/badge/Python-3.11-lightgray?style=flat&logo=python)
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
