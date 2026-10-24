# TRAINING

This file documents the scripts used for training, which are present in the `src/training` directory.

## Installing Dependencies

The cluster used for training did not support the use of `docker` or `uv`, so this part of the repository has a requirements file containing all the needed packages (`src/training/requirements.txt`).
The versions are picked to be as close to the `uv` environment as possible, but naturally reproducibility is lower. Keep in mind that the list assumes that Python is running on Linux. Here is the command to install the dependencies through a requirements file (make sure to do in a virtual environment from the project root):

```
pip install -r src/training/requirements.txt
```

Keep in mind that the training scripts (`train_vae.py` and `train_rflow.py`) will not work under the Windows `uv` environment due to lack of triton support, but the processing scripts (`prepare_data.py` and `encode_data.py`) run as expected.

## Description of Scripts

The `src/training` directory contains these scripts which should be run in this order:

1. Data preparation script (`prepare_data.py`) - processes CT scans from NIfTI format to VAE training ready .pt files. Can be run through this command (from project root):
```
python -m src.training.prepare_data
```

2. VAE training script (`train_vae.py`) - trains the VAE-GAN model. Can be run through this command (from project root, with at least 4 GPUs, trained on 4 A100 GPUs 40GB):
```
PYTHONPATH=$(pwd) torchrun --nproc_per_node=4 src/training/train_vae.py
```

3. CT encoding script (`encode_data.py`) - encodes data using the mean of the conditional latent distribution received from the trained VAE-GAN. Can be run through this command (from project root, GPU should have 16GB of VRAM):
```
python -m src.training.encode_data
```

4. Rectified Flow training script (`train_rflow.py`) - trains the Rectified Flow model. Can be run through this command (from project root, with at least 8 GPUs, trained on 8 A100 GPUs 40GB):
```
PYTHONPATH=$(pwd) torchrun --nproc_per_node=8 src/training/train_rflow.py
```

**IMPORTANT**: The Rectified Flow training pipeline does not perform any kind of early stopping or validation loss calculation. Instead weights are periodically saved. The best model was picked manually and is from epoch *TODO*.

## Logging

Monitoring of training runs is done through WandB. The scripts assume that the user has logged in using the following command before running training:

```
wandb login
```

## Additional Notes

This document provides an overview of the training scripts, but in order to get the full picture, the code documentation should be looked at. Special consideration should be given to the configuration file: `src/training/helpers/config.py`. In particular, make sure that the paths defined there match up with the actual data location. The scripts do not expect command line arguments, but parameters can be changed directly in the config if desired.
