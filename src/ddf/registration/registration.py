# https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/tree/master/Python

# https://simpleitk.org/doxygen/latest/html/examples.html
# https://simpleitk.org/doxygen/latest/html/ImageRegistrationMethod1_2ImageRegistrationMethod1_8py-example.html
# https://simpleitk.org/doxygen/latest/html/ImageRegistrationMethod2_2ImageRegistrationMethod2_8py-example.html
# https://simpleitk.org/doxygen/latest/html/ImageRegistrationMethod3_2ImageRegistrationMethod3_8py-example.html
# https://simpleitk.org/doxygen/latest/html/ImageRegistrationMethod4_2ImageRegistrationMethod4_8py-example.html

# B_spline registration
# https://simpleitk.org/doxygen/latest/html/ImageRegistrationMethodBSpline1_2ImageRegistrationMethodBSpline1_8py-example.html
# https://simpleitk.org/doxygen/latest/html/ImageRegistrationMethodBSpline2_2ImageRegistrationMethodBSpline2_8py-example.html
# https://simpleitk.org/doxygen/latest/html/ImageRegistrationMethodBSpline3_2ImageRegistrationMethodBSpline3_8py-example.html
# http://simpleitk.org/SimpleITK-Notebooks/01_Image_Basics.html

import json
import os

import SimpleITK as sitk


def smooth_and_resample(image: sitk.Image, shrink_factor: float, smoothing_sigma: float) -> sitk.Image:
    """
    Smooth and resample the provided image.

    Args:
        image (Image): The image we want to resample.
        shrink_factor (float): A number greater than one, such that the new image's size is original_size/shrink_factor.
        smoothing_sigma (float): Sigma for Gaussian smoothing, this is in physical (image spacing) units, not pixels.

    Returns:
        Image: Image which is a result of smoothing the input and
            then resampling it using the given sigma and shrink factor.
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

    Args:
        registration_algorithm (DemonsRegistrationFilter
        | DiffeomorphicDemonsRegistrationFilter
        | FastSymmetricForcesDemonsRegistrationFilter): Any registration algorithm that has an
            Execute(fixed_image, moving_image, displacement_field_image) method.
        fixed_image (Image): Resulting transformation maps points from
            this image's spatial domain to the moving image spatial domain.
        moving_image (Image): Resulting transformation maps points from the fixed_image's spatial domain to
            this image's spatial domain.
        initial_transform (Transform | None): Any SimpleITK transform, used to initialize the displacement field.
        shrink_factors (list[float] | None): Shrink factors relative to the original image's size.
        smoothing_sigmas (list[float] | None): Amount of smoothing which is done prior
            to resampling the image using the given shrink factor. These are in physical (image spacing) units.

    Returns:
        DisplacementFieldTransform: The resulting transform.
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


# Config
DATA_DICT = "src/data_full/data_dict.json"

SAVE_TRANSFORMS_DIR = "RESULTS/TRANSFORMS/"
SAVE_TRANSFORMED_IMAGES_DIR = "RESULTS/TRANSFORMED_IMAGES/"

with open(DATA_DICT) as f:
    data_dict = json.load(f)

all_data = data_dict["0"]["train"] + data_dict["0"]["val"] + data_dict["test"]

if os.path.isfile("completed.json"):
    with open("completed.json") as f:
        completed = json.load(f)
else:
    completed = []

for item in all_data:
    fixedImName = item["fixed_image"]
    movingImName = item["moving_image"]

    patient_id = os.path.basename(fixedImName).split("_")[1]
    fixed_id = os.path.basename(fixedImName).split("_")[3]
    moving_id = os.path.basename(movingImName).split("_")[3]

    if [patient_id, fixed_id, moving_id] in completed:
        continue

    fixed = sitk.ReadImage(fixedImName, sitk.sitkFloat32)
    moving = sitk.ReadImage(movingImName, sitk.sitkFloat32)

    demons_filter = sitk.DiffeomorphicDemonsRegistrationFilter()  # type: ignore
    demons_filter.SetNumberOfIterations(120)  # type: ignore

    # Regularization (update field - viscous, total field - elastic)
    demons_filter.SetSmoothDisplacementField(True)  # type: ignore
    demons_filter.SetStandardDeviations(0.6)  # type: ignore

    # Create initial transform
    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),  # type: ignore
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    # Run the registration
    try:
        tfm = multiscale_demons(
            registration_algorithm=demons_filter,
            fixed_image=fixed,
            moving_image=moving,
            initial_transform=initial_transform,
            shrink_factors=[16, 8, 4, 2],
            smoothing_sigmas=[16, 8, 4, 2],
        )
    except Exception:
        continue

    displacement_transform = sitk.DisplacementFieldTransform(tfm)  # type: ignore
    disp_field = sitk.TransformToDisplacementField(
        displacement_transform,
        sitk.sitkVectorFloat64,
        moving.GetSize(),  # type: ignore
        moving.GetOrigin(),  # type: ignore
        moving.GetSpacing(),  # type: ignore
        moving.GetDirection(),  # type: ignore
    )

    # Apply using ResampleImageFilter
    resampler = sitk.ResampleImageFilter()  # type: ignore
    resampler.SetReferenceImage(fixed)  # type: ignore
    resampler.SetInterpolator(sitk.sitkLinear)  # type: ignore
    resampler.SetDefaultPixelValue(0)  # type: ignore
    resampler.SetTransform(displacement_transform)  # type: ignore

    warped_image = resampler.Execute(moving)  # type: ignore

    fname = SAVE_TRANSFORMS_DIR + f"transform_patient_{patient_id}_fixed_{fixed_id}_moving_{moving_id}.nii.gz"
    sitk.WriteImage(disp_field, fname)

    fname = (
        SAVE_TRANSFORMED_IMAGES_DIR
        + f"transformed_image_patient_{patient_id}_fixed_{fixed_id}_moving_{moving_id}.nii.gz"
    )
    sitk.WriteImage(warped_image, fname)

    completed.append((patient_id, fixed_id, moving_id))

    with open("completed.json", "w") as f:
        json.dump(completed, f, indent=4)
