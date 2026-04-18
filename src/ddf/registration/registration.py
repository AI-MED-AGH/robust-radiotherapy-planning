import json
import os
from pathlib import Path

import SimpleITK as sitk


def smooth_and_resample(image: sitk.Image, shrink_factor: float, smoothing_sigma: float) -> sitk.Image:
    """
    Smooth and resample the provided image.

    Assumptions:
    - The input `image` is a valid 2D or 3D SimpleITK image (sitk.Image).
    - The image has consistent spacing, origin, and direction metadata defined.
    - `shrink_factor` > 1 and results in non-zero dimensions after resampling.
    - `smoothing_sigma` is given in physical units (consistent with image spacing).
    - The image size is large enough so that `(new_sz - 1)` is not zero in spacing calculation.
    - Linear interpolation is sufficient for the intended downstream task (e.g., registration).

    Returns
    -------
        image : Image
            The result of smoothing the input and then
            resampling it using the given sigma and shrink factor.

    Parameters
    ----------
        image : Image
            The image to resample.
        shrink_factor : float
            A number greater than one, such that the new image's size is original_size/shrink_factor.
        smoothing_sigma : float
            Sigma for Gaussian smoothing, this is in physical (image spacing) units, not pixels.
    """
    smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_sigma)  # type: ignore

    original_spacing = image.GetSpacing()  # type: ignore
    original_size = image.GetSize()  # type: ignore
    new_size = [int(sz / shrink_factor + 0.5) for sz in original_size]
    new_spacing = [
        ((original_sz - 1) * original_spc) / (new_sz - 1)
        for original_sz, original_spc, new_sz in zip(original_size, original_spacing, new_size, strict=True)
    ]
    return sitk.Resample(
        smoothed_image,
        new_size,
        sitk.Transform(),  # type: ignore
        sitk.sitkLinear,
        image.GetOrigin(),  # type: ignore
        new_spacing,
        image.GetDirection(),  # type: ignore
        0.0,
        image.GetPixelID(),  # type: ignore
    )


def multiscale_demons(
    registration_algorithm: sitk.DemonsRegistrationFilter
    | sitk.DiffeomorphicDemonsRegistrationFilter
    | sitk.FastSymmetricForcesDemonsRegistrationFilter,
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    initial_transform: sitk.Transform | None = None,
    shrink_factors: list[float] | None = None,
    smoothing_sigmas: list[float] | None = None,
) -> sitk.DisplacementFieldTransform:
    """
    Run the given registration algorithm in a multiscale fashion. The original scale should not be given as input as the
    original images are implicitly incorporated as the base of the pyramid.

    Assumptions:
    - `fixed_image` and `moving_image` are valid SimpleITK images with matching dimensionality (2D or 3D).
    - Both images are roughly aligned in physical space (same orientation, similar spacing/origin).
    - `registration_algorithm` implements Execute(fixed, moving, displacement_field).
    - `shrink_factors` and `smoothing_sigmas` are of equal length and ordered from coarse to fine.
    - Values in `shrink_factors` are > 1 and produce valid image sizes at each pyramid level.
    - `smoothing_sigmas` are given in physical units (consistent with image spacing).
    - If provided, `initial_transform` is defined in the same spatial domain as `fixed_image`.
    - Displacement fields use `sitkVectorFloat64` as required by Demons-based filters.

    Returns
    -------
        DisplacementFieldTransform:
            The resulting transform.

    Parameters
    ----------
        registration_algorithm : DemonsRegistrationFilter
        | DiffeomorphicDemonsRegistrationFilter
        | FastSymmetricForcesDemonsRegistrationFilter:
            Any registration algorithm that has an
            Execute(fixed_image, moving_image, displacement_field_image) method.
        fixed_image : Image
            Resulting transformation maps points from
            this image's spatial domain to the moving image spatial domain.
        moving_image : Image
            Resulting transformation maps points from the fixed_image's spatial domain to
            this image's spatial domain.
        initial_transform : Transform | None
            Any SimpleITK transform, used to initialize the displacement field.
        shrink_factors : list[float] | None
            Shrink factors relative to the original image's size.
        smoothing_sigmas : list[float] | None
            Amount of smoothing which is done prior
            to resampling the image using the given shrink factor. These are in physical (image spacing) units.
    """
    # Create image pyramid
    fixed_images = [fixed_image]
    moving_images = [moving_image]
    if shrink_factors and smoothing_sigmas:
        for shrink_factor, smoothing_sigma in reversed(list(zip(shrink_factors, smoothing_sigmas, strict=True))):
            fixed_images.append(smooth_and_resample(fixed_images[0], shrink_factor, smoothing_sigma))
            moving_images.append(smooth_and_resample(moving_images[0], shrink_factor, smoothing_sigma))

    # Create initial displacement field at lowest resolution
    # The pixel type is required to be sitkVectorFloat64 because of a constraint imposed by the Demons filters
    if initial_transform:
        initial_displacement_field = sitk.TransformToDisplacementField(
            initial_transform,
            sitk.sitkVectorFloat64,
            fixed_images[-1].GetSize(),  # type: ignore
            fixed_images[-1].GetOrigin(),  # type: ignore
            fixed_images[-1].GetSpacing(),  # type: ignore
            fixed_images[-1].GetDirection(),  # type: ignore
        )
    else:
        initial_displacement_field = sitk.Image(
            fixed_images[-1].GetWidth(),  # type: ignore
            fixed_images[-1].GetHeight(),  # type: ignore
            fixed_images[-1].GetDepth(),  # type: ignore
            sitk.sitkVectorFloat64,
        )
        initial_displacement_field.CopyInformation(fixed_images[-1])  # type: ignore

    # Run the registration
    initial_displacement_field = registration_algorithm.Execute(
        fixed_images[-1], moving_images[-1], initial_displacement_field
    )  # type: ignore

    # Start at the top of the pyramid and work our way down
    for f_image, m_image in reversed(list(zip(fixed_images[0:-1], moving_images[0:-1], strict=True))):
        initial_displacement_field = sitk.Resample(initial_displacement_field, f_image)
        initial_displacement_field = registration_algorithm.Execute(f_image, m_image, initial_displacement_field)  # type: ignore

    return sitk.DisplacementFieldTransform(initial_displacement_field)  # type: ignore


def run_demons_registration_pipeline(
    data_dict_path: Path,
    save_transforms_dir: Path,
    save_transformed_images_dir: Path,
    completed_path: Path = Path("completed.json"),
    iterations: int = 120,
    shrink_factors: list[float] | None = None,
    smoothing_sigmas: list[float] | None = None,
) -> None:
    """
    Run multiscale diffeomorphic demons registration for all image pairs defined in the
    input data dictionary, save the resulting displacement fields and warped images, and
    track completed cases in a JSON file.

    Assumptions:
    - `data_dict_path` points to a valid JSON file with keys `"0"` and `"test"`.
    - The `"0"` entry contains `"train"` and `"val"` subsets with image-pair dictionaries.
    - Each item contains valid `"fixed_image"` and `"moving_image"` file paths.
    - Image filenames follow the pattern `Patient_<id>_fraction_<id>_.nii.gz`.
    - Fixed and moving images are valid 3D SimpleITK images with matching dimensionality.
    - `multiscale_demons(...)` is available and returns a valid displacement field transform.
    - `save_transforms_dir` and `save_transformed_images_dir` are writable locations.
    - Completed cases are stored as `[patient_id, fixed_id, moving_id]` lists in `completed_path`.

    Returns
    -------
        None:
            The function saves displacement fields, warped images, and updates the
            completed-cases JSON file in place.

    Parameters
    ----------
        data_dict_path : Path
            Path to the JSON file containing the train, validation, and test image pairs.
        save_transforms_dir : Path
            Directory where displacement field images will be saved.
        save_transformed_images_dir : Path
            Directory where transformed moving images will be saved.
        completed_path : Path
            Path to the JSON file used to track already processed image pairs.
        iterations : int
            Number of iterations used by the diffeomorphic demons filter at each scale.
        shrink_factors : list[float] | None
            Shrink factors relative to the original image size for the multiscale pyramid.
        smoothing_sigmas : list[float] | None
            Gaussian smoothing sigmas used before resampling at each pyramid level.
            These are given in physical (image spacing) units.
    """
    if shrink_factors is None:
        shrink_factors = [16, 8, 4, 2]

    if smoothing_sigmas is None:
        smoothing_sigmas = [16, 8, 4, 2]

    save_transforms_dir.mkdir(parents=True, exist_ok=True)
    save_transformed_images_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dict_path) as f:
        data_dict = json.load(f)

    all_data = data_dict["0"]["train"] + data_dict["0"]["val"] + data_dict["test"]

    if os.path.isfile(completed_path):
        with open(completed_path) as f:
            completed = json.load(f)
    else:
        completed = []

    for item in all_data:
        fixed_im_name = item["fixed_image"]
        moving_im_name = item["moving_image"]

        patient_id = os.path.basename(fixed_im_name).split("_")[1]
        fixed_id = os.path.basename(fixed_im_name).split("_")[3]
        moving_id = os.path.basename(moving_im_name).split("_")[3]

        case_id = [patient_id, fixed_id, moving_id]
        if case_id in completed:
            continue

        fixed = sitk.ReadImage(fixed_im_name, sitk.sitkFloat32)
        moving = sitk.ReadImage(moving_im_name, sitk.sitkFloat32)

        demons_filter = sitk.DiffeomorphicDemonsRegistrationFilter()  # type: ignore
        demons_filter.SetNumberOfIterations(iterations)  # type: ignore
        demons_filter.SetSmoothDisplacementField(True)  # type: ignore
        demons_filter.SetStandardDeviations(0.6)  # type: ignore

        initial_transform = sitk.CenteredTransformInitializer(
            fixed,
            moving,
            sitk.Euler3DTransform(),  # type: ignore
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

        print(f"Starting registration: patient {patient_id}, fixed {fixed_id}, moving {moving_id}")

        try:
            tfm = multiscale_demons(
                registration_algorithm=demons_filter,
                fixed_image=fixed,
                moving_image=moving,
                initial_transform=initial_transform,
                shrink_factors=shrink_factors,
                smoothing_sigmas=smoothing_sigmas,
            )
            print(f"Finished registration: patient {patient_id}, fixed {fixed_id}, moving {moving_id}")
        except Exception as e:
            print(f"Registration failed for patient {patient_id}, fixed {fixed_id}, moving {moving_id}: {e}")
            continue

        disp_field = sitk.TransformToDisplacementField(
            tfm,
            sitk.sitkVectorFloat64,
            moving.GetSize(),  # type: ignore
            moving.GetOrigin(),  # type: ignore
            moving.GetSpacing(),  # type: ignore
            moving.GetDirection(),  # type: ignore
        )

        resampler = sitk.ResampleImageFilter()  # type: ignore
        resampler.SetReferenceImage(fixed)  # type: ignore
        resampler.SetInterpolator(sitk.sitkLinear)  # type: ignore
        resampler.SetDefaultPixelValue(0)  # type: ignore
        resampler.SetTransform(tfm)  # type: ignore

        warped_image = resampler.Execute(moving)  # type: ignore

        transform_path = save_transforms_dir / (
            f"transform_patient_{patient_id}_fixed_{fixed_id}_moving_{moving_id}.nii.gz"
        )
        sitk.WriteImage(disp_field, str(transform_path))

        warped_image_path = save_transformed_images_dir / (
            f"transformed_image_patient_{patient_id}_fixed_{fixed_id}_moving_{moving_id}.nii.gz"
        )
        sitk.WriteImage(warped_image, str(warped_image_path))

        completed.append(case_id)

        with open(completed_path, "w") as f:
            json.dump(completed, f, indent=4)
