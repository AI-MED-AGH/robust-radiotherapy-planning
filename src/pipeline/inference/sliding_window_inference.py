import torch


def extract_roi(
    image: torch.Tensor,
    step: int,
    axis: int,
    chunk_size: int,
    halo_size: int,
    image_size: int,
    model_type: str,
    factor: int,
) -> tuple[torch.Tensor, int]:
    """
    Extract one image chunk with halo padding along a selected spatial axis.

    This helper is used by custom sliding-window inference. It extracts a
    chunk from a 5D tensor and includes an additional halo region around the
    core chunk to reduce boundary artifacts.

    Expected input shape:
        [B, C, H, W, D]

    Supported split axes:
    - axis=2: split along height
    - axis=3: split along width

    Parameters
    ----------
    image : torch.Tensor
        Input tensor with shape [B, C, H, W, D].

    step : int
        Index of the current chunk along the selected axis.

    axis : int
        Spatial axis along which the chunk is extracted.
        Must be 2 or 3.

    chunk_size : int
        Size of the core chunk before adding halo.

    halo_size : int
        Number of extra voxels added on each side of the chunk.

    image_size : int
        Full image size along the selected splitting axis.

    model_type : str
        Type of model applied to the chunk.
        Must be either "encoder" or "decoder".

    factor : int
        Spatial scaling factor between input and output.
        For encoder, output is smaller by this factor.
        For decoder, output is larger by this factor.

    Returns
    -------
    image_chunk : torch.Tensor
        Extracted image chunk with halo and padding.

    padding : int
        Offset used later to remove halo from the model output.

    Raises
    ------
    TypeError
        If `image` is not a torch.Tensor.

    ValueError
        If `image` is not 5-dimensional.
        If `axis` is not 2 or 3.
        If `model_type` is not "encoder" or "decoder".
        If `step`, `chunk_size`, `halo_size`, `image_size`, or `factor`
        have invalid values.
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"`image` must be a torch.Tensor. Got {type(image)}")

    if image.ndim != 5:
        raise ValueError(f"`image` must have shape [B, C, H, W, D]. Got {image.shape}")

    if axis not in [2, 3]:
        raise ValueError("`axis` must be either 2 or 3.")

    if model_type not in ["encoder", "decoder"]:
        raise ValueError("`model_type` must be either 'encoder' or 'decoder'")

    if step < 0:
        raise ValueError("`step` must be non-negative")

    if chunk_size <= 0:
        raise ValueError("`chunk_size` must be greater than 0")

    if halo_size < 0:
        raise ValueError("`halo_size` must be non-negative")

    if image_size <= 0:
        raise ValueError("`image_size` must be greater than 0")

    if factor <= 0:
        raise ValueError("`factor` must be greater than 0")

    start_idx = step * chunk_size
    end_idx = start_idx + chunk_size

    if start_idx >= image_size:
        raise ValueError(f"`step` is too large for image size. start_idx={start_idx}, image_size={image_size}")

    input_start = max(0, start_idx - halo_size)
    input_end = min(image_size, end_idx + halo_size)

    if axis == 2:
        image_chunk = image[:, :, input_start:input_end, :, :].contiguous()
    else:
        image_chunk = image[:, :, :, input_start:input_end, :].contiguous()

    current_size = image_chunk.shape[axis]
    target_size = chunk_size + 2 * halo_size
    pad_amount = target_size - current_size

    if pad_amount < 0:
        raise ValueError(
            f"`pad_amount` cannot be negative. Got {pad_amount}. Check chunk_size, halo_size, and image_size"
        )

    if model_type == "encoder":
        padding = (start_idx - input_start) // factor
    else:
        padding = (start_idx - input_start) * factor

    if axis == 2:
        image_chunk = torch.nn.functional.pad(
            image_chunk,
            (0, 0, 0, 0, 0, pad_amount),
        )
    else:
        image_chunk = torch.nn.functional.pad(
            image_chunk,
            (0, 0, 0, pad_amount, 0, 0),
        )

    return image_chunk, padding


def sliding_window_inference(
    image: torch.Tensor,
    chunk_size: int,
    halo_size: int,
    image_size: int,
    model,
    model_type: str,
    factor: int,
) -> torch.Tensor:
    """
    Perform custom sliding-window inference with halo overlap.

    The image is split into smaller chunks along the height and width axes.
    Each chunk is expanded with a halo region before being passed through the
    model. After inference, the halo region is removed from the model output,
    and clean chunks are stitched back together.

    Expected input shape:
        [B, C, H, W, D]

    Splitting axes:
    - axis 2: height
    - axis 3: width

    The depth axis is not split.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor with shape [B, C, H, W, D].

    chunk_size : int
        Size of the core chunk before adding halo.

    halo_size : int
        Number of halo voxels added around each chunk.

    image_size : int
        Full image size along height and width.
        This assumes height and width have the same size.

    model
        Callable model used for inference on each chunk.

    model_type : str
        Type of model used for inference.
        Must be either:
        - "encoder": output spatial size is smaller by `factor`
        - "decoder": output spatial size is larger by `factor`

    factor : int
        Spatial scaling factor between model input and output.

    Returns
    -------
    canvas : torch.Tensor
        Reconstructed output tensor after chunk inference and stitching.

    Raises
    ------
    TypeError
        If `image` is not a torch.Tensor.
        If `model` is not callable.

    ValueError
        If `image` is not 5-dimensional.
        If `model_type` is not "encoder" or "decoder".
        If `chunk_size`, `image_size`, or `factor` are not positive.
        If `halo_size` is negative.
        If `image_size` is not divisible by `chunk_size`.
        If encoder `chunk_size` is not divisible by `factor`.
        If input height or width is smaller than `image_size`.
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"`image` must be a torch.Tensor. Got {type(image)}")

    if image.ndim != 5:
        raise ValueError(f"`image` must have shape [B, C, H, W, D]. Got {image.shape}")

    if not callable(model):
        raise TypeError("`model` must be callable")

    if model_type not in ["encoder", "decoder"]:
        raise ValueError("`model_type` must be either 'encoder' or 'decoder'")

    if chunk_size <= 0:
        raise ValueError("`chunk_size` must be greater than 0")

    if halo_size < 0:
        raise ValueError("`halo_size` must be non-negative")

    if image_size <= 0:
        raise ValueError("`image_size` must be greater than 0")

    if factor <= 0:
        raise ValueError("`factor` must be greater than 0")

    if image_size % chunk_size != 0:
        raise ValueError(
            f"`image_size` must be divisible by `chunk_size`. Got image_size={image_size}, chunk_size={chunk_size}"
        )

    if model_type == "encoder" and chunk_size % factor != 0:
        raise ValueError(
            "For encoder inference, `chunk_size` must be divisible by "
            f"`factor`. Got chunk_size={chunk_size}, factor={factor}"
        )

    if image.shape[2] < image_size or image.shape[3] < image_size:
        raise ValueError(
            f"Input image height and width must be at least `image_size`. "
            f"Got image shape {image.shape} and image_size={image_size}"
        )

    num_steps = image_size // chunk_size
    output_chunks = []

    for step_h in range(num_steps):
        image_chunk, padding_h = extract_roi(
            image=image,
            step=step_h,
            axis=2,
            chunk_size=chunk_size,
            halo_size=halo_size,
            image_size=image_size,
            model_type=model_type,
            factor=factor,
        )

        output_chunk = []

        for step_w in range(num_steps):
            image_subchunk, padding_w = extract_roi(
                image=image_chunk,
                step=step_w,
                axis=3,
                chunk_size=chunk_size,
                halo_size=halo_size,
                image_size=image_size,
                model_type=model_type,
                factor=factor,
            )

            subchunk = model(image_subchunk)

            if model_type == "encoder":
                core_output = chunk_size // factor
            else:
                core_output = chunk_size * factor

            clean_subchunk = subchunk[
                :,
                :,
                padding_h : padding_h + core_output,
                padding_w : padding_w + core_output,
                :,
            ].contiguous()

            output_chunk.append(clean_subchunk)

        output_chunks.append(output_chunk)

    canvas = torch.cat(
        [torch.cat(output_chunk, dim=3) for output_chunk in output_chunks],
        dim=2,
    )

    return canvas
