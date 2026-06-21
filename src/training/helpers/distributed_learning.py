import os

import torch
import torch.distributed as dist


def init_distributed() -> tuple[int, int, int, bool, torch.device]:
    """
    Initialize the PyTorch Distributed Data Parallel (DDP) environment.

    Looks for distributed setup parameters within the environment variables. If missing,
    configures execution gracefully for a standard local single-GPU/CPU fallback workspace.

    Returns
    -------
    tuple[int, int, int, bool, torch.device]
        A tuple containing:
        - The global process evaluation rank index.
        - Total world size count of distributed active workers.
        - The localized process hardware rank index on the current machine.
        - A boolean flag verifying if distributed mode is engaged.
        - The concrete targeted hardware allocation device object.

    Raises
    ------
    RuntimeError
        If process group initialization with the NCCL backend fails.
    """

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        # Initialize the process group (NCCL is the fastest backend for NVIDIA GPUs)
        try:
            dist.init_process_group(backend="nccl")
        except Exception as err:
            raise RuntimeError("Failed to initialize PyTorch distributed process group using NCCL backend") from err
        torch.cuda.set_device(local_rank)
        is_distributed = True
    else:
        # Fallback to single-GPU or CPU
        rank = 0
        world_size = 1
        local_rank = 0
        is_distributed = False
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, is_distributed, device


def synchronize_loss(losses: dict[str, torch.Tensor], is_distributed: bool, world_size: int) -> dict[str, float]:
    """
    Synchronize, sum, and average cross-node evaluation losses across distributed GPU devices.

    Parameters
    ----------
    losses : dict[str, torch.Tensor]
        A dictionary mapping loss identifiers to local PyTorch GPU tensor instances.

    is_distributed : bool
        Flag checking if DDP cluster reductions must execute.

    world_size : int
        Total node partition space count to average weights across.

    Returns
    -------
    dict[str, float]
        An aligned dictionary containing pure standard python floating point loss metrics.

    Raises
    ------
    RuntimeError
        If inter-node communication or tensor reduction fails in distributed execution.
    """

    if not losses:
        return {}

    new_loss_dict = {}
    if is_distributed:
        try:
            # Gather the losses into a single tensor on the GPU
            loss_tensor = torch.stack(list(losses.values()))

            # Sum the tensors across all available GPUs
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

            # Divide by world_size to get the true mathematical average
            loss_tensor /= world_size

            # Unpack back into the dictionary
            for i, key in enumerate(losses.keys()):
                new_loss_dict[key] = loss_tensor[i].item()
        except Exception as err:
            raise RuntimeError("Distributed loss communication reduction operation (all_reduce) failed.") from err
    else:
        # Safely extract floats for single-GPU runs
        for key in losses.keys():
            new_loss_dict[key] = losses[key].item()

    return new_loss_dict
