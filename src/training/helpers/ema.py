import copy
from typing import cast

import torch


class EMA:
    """
    Exponential Moving Average (EMA) helper class for tracking model weights over time.

    This class maintains a copy of the model parameters and updates them using an
    exponential decay schedule during training. It provides utility methods to swap
    the active model parameters with the EMA shadow weights for validation/inference
    and restore them afterward.

    Parameters
    ----------
    model : torch.nn.Module
        The active training model whose parameters will be tracked.

    decay : float
        The decay rate for the moving average. Must be strictly between 0.0 and 1.0
        (typically close to 1.0, e.g., 0.999).

    Raises
    ------
    TypeError
        If the provided `model` is not an instance of `torch.nn.Module`.

    ValueError
        If `decay` is not within the range (0.0, 1.0).
    """

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Expected model to be a torch.nn.Module, but got {type(model).__name__}")

        if not (0.0 < decay < 1.0):
            raise ValueError(f"EMA decay must be strictly between 0.0 and 1.0, got: {decay}")

        # Strip DDP wrapper if present to keep the base architecture clean
        raw_model = model.module if hasattr(model, "module") else model

        # Create a deep copy of the un-wrapped base model
        self.ema_model = cast(torch.nn.Module, copy.deepcopy(raw_model))
        self.ema_model.eval()
        self.decay = decay

        # Ensure EMA weights don't track gradients
        for param in self.ema_model.parameters():
            param.requires_grad = False

        # Initialize backup as an empty dictionary
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """
        Update the internal EMA weights using a step of exponential moving average.

        Parameters
        ----------
        model : torch.nn.Module
            The active training model containing the newly updated optimization weights.

        Raises
        ------
        ValueError
            If the provided model architecture keys do not match the tracked EMA weights.
        """

        # Always extract the underlying raw model to strip any DDP 'module.' prefixes
        current_raw_model = cast(torch.nn.Module, model.module if hasattr(model, "module") else model)

        ema_state = self.ema_model.state_dict()
        model_state = current_raw_model.state_dict()

        if ema_state.keys() != model_state.keys():
            raise ValueError(
                "Cannot update EMA weights: The provided model's state_dict keys "
                "do not match the initialized EMA model state keys"
            )

        for key in ema_state.keys():
            if ema_state[key].is_floating_point():
                ema_state[key].mul_(self.decay).add_(model_state[key], alpha=1.0 - self.decay)
            else:
                ema_state[key].copy_(model_state[key])

    def apply_shadow(self, model: torch.nn.Module) -> None:
        """
        In-place overwrite the active model parameters with the internal EMA shadow weights.
        The original parameters are cached inside a backup repository.

        Parameters
        ----------
        model : torch.nn.Module
            The model to overwrite with shadow parameters.

        Raises
        ------
        RuntimeError
            If a shadow state has already been applied without being previously restored.

        KeyError
            If a parameter name in the active model cannot be found in the tracked EMA model.
        """

        if self.backup:
            raise RuntimeError(
                "A shadow weight backup already exists. You must call restore() before calling apply_shadow() again"
            )

        # Extract the runtime model
        current_raw_model = cast(torch.nn.Module, model.module if hasattr(model, "module") else model)
        ema_parameters = dict(self.ema_model.named_parameters())

        for name, param in current_raw_model.named_parameters():
            if param.requires_grad:
                if name not in ema_parameters:
                    raise KeyError(
                        f"Parameter '{name}' found in active model is missing from the internal EMA model parameters"
                    )

                # Store original weights
                self.backup[name] = param.data.clone()
                # Overwrite with EMA weights IN-PLACE
                param.data.copy_(ema_parameters[name].data)

    def restore(self, model: torch.nn.Module) -> None:
        """
        In-place restore the original model parameters from the cached backup registry.

        Parameters
        ----------
        model : torch.nn.Module
            The model whose parameters should be restored back to their pre-shadow state.

        Raises
        ------
        RuntimeError
            If no backup exists (i.e., apply_shadow was never called).

        KeyError
            If an active model parameter that requires gradients is missing from the backup registry.
        """
        if not self.backup:
            raise RuntimeError(
                "No parameter backup found. Cannot restore weights because apply_shadow() "
                "was not called or the backup has already been cleared"
            )

        current_raw_model = cast(torch.nn.Module, model.module if hasattr(model, "module") else model)
        for name, param in current_raw_model.named_parameters():
            if param.requires_grad:
                if name not in self.backup:
                    raise KeyError(
                        f"Expected parameter '{name}' to be in the weight backup registry, but it was missing"
                    )
                # Restore original weights IN-PLACE
                param.data.copy_(self.backup[name])

        # Clear backup dictionary to free up GPU memory after validation loop ends
        self.backup.clear()
