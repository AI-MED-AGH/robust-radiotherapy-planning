"""Write tensor snapshots in PyTorch's native .pt archive format."""

import copy
import pickle
from pathlib import Path
from typing import IO, Any

import torch
from torch.serialization import DEFAULT_PROTOCOL, _open_zipfile_writer, _save


def _snapshot(value: Any) -> Any:
    """
    Recursively copy tensors to independent CPU storage for serialization.

    Parameters
    ----------
    value : Any
        A tensor or checkpoint payload. Traversal follows dictionary values,
        list elements, and tuple elements.

    Returns
    -------
    Any
        A snapshot with each tensor replaced by its detached CPU clone.
        Dictionaries retain their keys and shallow-copied attributes, including
        model state dictionary metadata. Lists and tuples are rebuilt, and
        other leaves are returned by reference.
    """

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()

    if isinstance(value, dict):
        # Keep state_dict attributes, including module version metadata
        result = copy.copy(value)
        result.update((key, _snapshot(item)) for key, item in value.items())
        return result

    if isinstance(value, list):
        return [_snapshot(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)

    return value


def save_pt(value: Any, destination: str | Path | IO[bytes]) -> None:
    """
    Write a tensor or checkpoint to a native archive readable by `torch.load`.

    Each tensor is saved with `detach().cpu().clone()`. Detaching removes its
    connection to the autograd graph and leaves its accumulated gradient behind.
    Moving to CPU makes the snapshot loadable on CPU-only machines. Cloning
    gives the snapshot independent storage even when the input is already on
    CPU or is a view into a larger tensor.

    Parameters
    ----------
    value : Any
        A serializable payload, such as a CT tensor or a training checkpoint.
        Tensor values nested in dictionaries, lists, and tuples are saved as
        detached CPU clones.

    destination : str | Path | IO[bytes]
        Output file path (conventionally ending in `.pt`) or a writable binary
        stream. For a path, the parent directory must already exist and an
        existing file is overwritten.
    """

    snapshot = _snapshot(value)
    # Keep PyTorch's private native archive API isolated here
    if isinstance(destination, Path):
        destination = str(destination)

    with _open_zipfile_writer(destination) as archive:
        _save(snapshot, archive, pickle, DEFAULT_PROTOCOL, False)  # type: ignore[no-untyped-call]
