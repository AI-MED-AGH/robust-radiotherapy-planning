import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F


def _create_normalized_grid(
    shape: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Generate a 3D identity sampling grid with coordinates normalized to [-1, 1]
    for PyTorch spatial transformations (`F.grid_sample`).

    Constructs a 5D coordinate grid tensor scaled across all spatial dimensions.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Spatial volumetric shape specified as (W, H, D) for width, height, and depth.

    device : torch.device
        PyTorch computation device allocated for tensor generation (e.g., CPU or CUDA).

    Returns
    -------
    torch.Tensor
        Normalized identity coordinate sampling grid of shape (1, W, H, D, 3)
        with vector components ordered as [x, y, z] in the range [-1.0, 1.0].
    """

    W, H, D = shape
    grid_w = torch.linspace(-1, 1, W, device=device)
    grid_h = torch.linspace(-1, 1, H, device=device)
    grid_d = torch.linspace(-1, 1, D, device=device)

    # meshgrid with indexing="ij" creates coordinate tensors of shape (W, H, D)
    mesh_w, mesh_h, mesh_d = torch.meshgrid(grid_w, grid_h, grid_d, indexing="ij")

    # Stack into (1, W, H, D, 3) with grid_sample ordering [x, y, z] -> [width, height, depth]
    grid = torch.stack([mesh_w, mesh_h, mesh_d], dim=-1).unsqueeze(0)
    return grid


def transport_dvf_to_atlas(
    dvf_intra: torch.Tensor,
    dvf_inter: torch.Tensor,
    voxel_spacing: tuple[float, float, float],
) -> torch.Tensor:
    """
    Transport an intra-patient DVF (fraction X -> fraction 1) into Atlas coordinate space
    using the inter-patient DVF (fraction 1 -> Atlas).

    Parameters
    ----------
    dvf_intra : torch.Tensor
        Intra-patient DVF of shape (3, W, H, D) in patient planning space.

    dvf_inter : torch.Tensor
        Inter-patient DVF of shape (3, W, H, D) mapping Atlas grid to patient planning grid.

    voxel_spacing : tuple[float, float, float]
        Physical voxel spacing in millimeters as (spacing_x, spacing_y, spacing_z).

    Returns
    -------
    torch.Tensor
        Transported DVF in Atlas coordinate space of shape (3, W, H, D).
    """

    device = dvf_intra.device
    _, W, H, D = dvf_intra.shape

    # Normalize inter-patient displacement vectors to grid coordinate space [-1, 1]
    # Spacing conversion: voxel_disp = mm_disp / spacing
    disp_x = (dvf_inter[0] / voxel_spacing[0]) * (2.0 / (W - 1))
    disp_y = (dvf_inter[1] / voxel_spacing[1]) * (2.0 / (H - 1))
    disp_z = (dvf_inter[2] / voxel_spacing[2]) * (2.0 / (D - 1))

    # Construct sampling coordinates: Identity_Grid + Normalized_Displacement
    norm_disp = torch.stack([disp_x, disp_y, disp_z], dim=-1).unsqueeze(0)
    identity_grid = _create_normalized_grid((W, H, D), device=device)
    sampling_grid = identity_grid + norm_disp

    # PyTorch grid_sample expects input layout (N, C, Depth, Height, Width) where grid
    # coords [x, y, z] map x -> Width (dim 4), y -> Height (dim 3), z -> Depth (dim 2).
    # Permute (1, 3, W, H, D) -> (1, 3, D, H, W) to align spatial axes with grid conventions.
    dvf_intra_5d = dvf_intra.unsqueeze(0).permute(0, 1, 4, 3, 2)

    dvf_atlas = F.grid_sample(
        dvf_intra_5d,
        sampling_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).squeeze(0)

    return dvf_atlas


def compute_bspline_dvf(
    fixed_tensor: torch.Tensor,
    moving_tensor: torch.Tensor,
    spacing: tuple[float, float, float],
    max_iterations: int = 30,
) -> torch.Tensor:
    """
    Register a moving 3D CT tensor to a fixed 3D CT tensor using non-rigid B-spline registration.

    Constructs a free-form B-spline deformation grid and optimizes spatial alignment using
    Mean Squared Error (MSE) and L-BFGS-B optimization. The registration is capped at a low
    number of iterations to enforce implicit regularization against image noise and artifacts.

    Parameters
    ----------
    fixed_tensor : torch.Tensor
        Target reference 3D volume tensor of shape `(W, H, D)`.

    moving_tensor : torch.Tensor
        Source 3D volume tensor of shape `(W, H, D)` to be deformed onto fixed domain.

    spacing : tuple[float, float, float]
        Physical voxel spacing in millimeters as `(spacing_x, spacing_y, spacing_z)`.

    max_iterations : int, default=30
        Maximum allowable L-BFGS-B optimizer iterations for implicit regularization.

    Returns
    -------
    torch.Tensor
        3D Deformation Vector Field tensor of shape `(3, W, H, D)` in physical units (millimeters),
        permuted to match PyTorch channel-first memory conventions.

    Raises
    ------
    ValueError
        If input tensors are not 3D or if shapes of fixed and moving tensors mismatch.
    """

    if fixed_tensor.ndim != 3 or moving_tensor.ndim != 3:
        raise ValueError(
            f"Expected 3D volumetric tensors, got shapes {tuple(fixed_tensor.shape)} and {tuple(moving_tensor.shape)}"
        )

    if fixed_tensor.shape != moving_tensor.shape:
        raise ValueError(
            f"Fixed tensor shape {tuple(fixed_tensor.shape)} does not match moving tensor {tuple(moving_tensor.shape)}"
        )

    # Convert PyTorch (W, H, D) -> NumPy (D, H, W) for SimpleITK image construction
    fixed_np = fixed_tensor.detach().cpu().numpy().astype(np.float32).transpose(2, 1, 0)
    moving_np = moving_tensor.detach().cpu().numpy().astype(np.float32).transpose(2, 1, 0)

    fixed_img = sitk.GetImageFromArray(fixed_np)
    moving_img = sitk.GetImageFromArray(moving_np)

    fixed_img.SetSpacing(spacing)  # type: ignore[no-untyped-call]
    moving_img.SetSpacing(spacing)  # type: ignore[no-untyped-call]

    transform_domain_mesh_size = [6, 6, 6]
    bspline_transform = sitk.BSplineTransformInitializer(
        fixed_img,
        transform_domain_mesh_size,
    )  # type: ignore[no-untyped-call]

    registration = sitk.ImageRegistrationMethod()  # type: ignore[no-untyped-call]
    registration.SetMetricAsMeanSquares()  # type: ignore[no-untyped-call]
    registration.SetInterpolator(sitk.sitkLinear)  # type: ignore[no-untyped-call]

    # Evaluate metric on only 10% of voxels selected randomly
    registration.SetMetricSamplingStrategy(registration.REGULAR)  # type: ignore[no-untyped-call]
    registration.SetMetricSamplingPercentage(0.05)  # type: ignore[no-untyped-call]

    registration.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=max_iterations,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1000,
    )  # type: ignore[no-untyped-call]

    registration.SetInitialTransform(bspline_transform, inPlace=True)  # type: ignore[no-untyped-call]
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])  # type: ignore[no-untyped-call]
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2.0, 1.0, 0.0])  # type: ignore[no-untyped-call]
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()  # type: ignore[no-untyped-call]

    final_transform = registration.Execute(fixed_img, moving_img)  # type: ignore[no-untyped-call]

    displacement_field_img = sitk.TransformToDisplacementField(
        final_transform,
        sitk.sitkVectorFloat32,
        size=fixed_img.GetSize(),  # type: ignore[no-untyped-call]
        outputOrigin=fixed_img.GetOrigin(),  # type: ignore[no-untyped-call]
        outputSpacing=fixed_img.GetSpacing(),  # type: ignore[no-untyped-call]
        outputDirection=fixed_img.GetDirection(),  # type: ignore[no-untyped-call]
    )

    # Reorder dimensions from (D, H, W, 3) back to PyTorch layout (3, W, H, D)
    dvf_array = sitk.GetArrayFromImage(displacement_field_img).astype(np.float32)
    dvf_tensor = torch.from_numpy(dvf_array).permute(3, 2, 1, 0).contiguous()

    return dvf_tensor
