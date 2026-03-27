import numpy as np
import glob
import os
import SimpleITK as sitk
import json

### Define Registration Method ###
def smooth_and_resample(image, shrink_factor, smoothing_sigma):
    """
    Args:
        image: The image we want to resample.
        shrink_factor: A number greater than one, such that the new image's size is original_size/shrink_factor.
        smoothing_sigma: Sigma for Gaussian smoothing, this is in physical (image spacing) units, not pixels.
    Return:
        Image which is a result of smoothing the input and then resampling it using the given sigma and shrink factor.
    """
    smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_sigma)
    
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(sz/float(shrink_factor) + 0.5) for sz in original_size]
    new_spacing = [((original_sz-1)*original_spc)/(new_sz-1) 
                   for original_sz, original_spc, new_sz in zip(original_size, original_spacing, new_size)]
    return sitk.Resample(smoothed_image, new_size, sitk.Transform(), 
                         sitk.sitkLinear, image.GetOrigin(),
                         new_spacing, image.GetDirection(), 0.0, 
                         image.GetPixelID())


    
def multiscale_demons(registration_algorithm,
                      fixed_image, moving_image, initial_transform = None, 
                      shrink_factors=None, smoothing_sigmas=None):
    """
    Run the given registration algorithm in a multiscale fashion. The original scale should not be given as input as the
    original images are implicitly incorporated as the base of the pyramid.
    Args:
        registration_algorithm: Any registration algorithm that has an Execute(fixed_image, moving_image, displacement_field_image)
                                method.
        fixed_image: Resulting transformation maps points from this image's spatial domain to the moving image spatial domain.
        moving_image: Resulting transformation maps points from the fixed_image's spatial domain to this image's spatial domain.
        initial_transform: Any SimpleITK transform, used to initialize the displacement field.
        shrink_factors: Shrink factors relative to the original image's size.
        smoothing_sigmas: Amount of smoothing which is done prior to resmapling the image using the given shrink factor. These
                          are in physical (image spacing) units.
    Returns: 
        SimpleITK.DisplacementFieldTransform
    """
    # Create image pyramid.
    fixed_images = [fixed_image]
    moving_images = [moving_image]
    if shrink_factors:
        for shrink_factor, smoothing_sigma in reversed(list(zip(shrink_factors, smoothing_sigmas))):
            fixed_images.append(smooth_and_resample(fixed_images[0], shrink_factor, smoothing_sigma))
            moving_images.append(smooth_and_resample(moving_images[0], shrink_factor, smoothing_sigma))
    
    # Create initial displacement field at lowest resolution. 
    # Currently, the pixel type is required to be sitkVectorFloat64 because of a constraint imposed by the Demons filters.
    if initial_transform:
        initial_displacement_field = sitk.TransformToDisplacementField(initial_transform, 
                                                                       sitk.sitkVectorFloat64,
                                                                       fixed_images[-1].GetSize(),
                                                                       fixed_images[-1].GetOrigin(),
                                                                       fixed_images[-1].GetSpacing(),
                                                                       fixed_images[-1].GetDirection())
    else:
        initial_displacement_field = sitk.Image(fixed_images[-1].GetWidth(), 
                                                fixed_images[-1].GetHeight(),
                                                fixed_images[-1].GetDepth(),
                                                sitk.sitkVectorFloat64)
        initial_displacement_field.CopyInformation(fixed_images[-1])
 
    # Run the registration.            
    initial_displacement_field = registration_algorithm.Execute(fixed_images[-1], 
                                                                moving_images[-1], 
                                                                initial_displacement_field)
    # Start at the top of the pyramid and work our way down.    
    for f_image, m_image in reversed(list(zip(fixed_images[0:-1], moving_images[0:-1]))):
            initial_displacement_field = sitk.Resample (initial_displacement_field, f_image)
            initial_displacement_field = registration_algorithm.Execute(f_image, m_image, initial_displacement_field)
    
    return sitk.DisplacementFieldTransform(initial_displacement_field)

##################################################################

with open('data_dict.json','r') as f:
    data_dict = json.load(f)

all_data = data_dict['0']['train'] + data_dict['0']['val'] + data_dict['test']

save_transforms_dir = './TRANSFORMS/'
save_transformed_images_dir = './TRANSFORMED_IMAGES/'


if os.path.isfile('completed.json'):
    with open('completed.json','r') as f:
        completed = json.load(f)
else:
    completed = []

for item in all_data:
    fixedImName = item['fixed_image']
    movingImName = item['moving_image']

    patient_id = os.path.basename(fixedImName).split('_')[1]
    fixed_id = os.path.basename(fixedImName).split('_')[3]
    moving_id = os.path.basename(movingImName).split('_')[3]

    if [patient_id, fixed_id, moving_id] in completed:
        continue

    fixed = sitk.ReadImage(fixedImName, sitk.sitkFloat32)
    moving = sitk.ReadImage(movingImName, sitk.sitkFloat32)

    demons_filter =  sitk.DiffeomorphicDemonsRegistrationFilter()
    demons_filter.SetNumberOfIterations(120)

    # Regularization (update field - viscous, total field - elastic).
    demons_filter.SetSmoothDisplacementField(True)
    demons_filter.SetStandardDeviations(0.6)

    # create initial transform
    initial_transform = sitk.CenteredTransformInitializer(fixed,
                                                          moving,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)


    # Run the registration
    try:
        tfm = multiscale_demons(registration_algorithm=demons_filter,
                                fixed_image = fixed,
                                moving_image = moving,
                                initial_transform = initial_transform,
                                shrink_factors = [16, 8, 4, 2],
                                smoothing_sigmas = [16, 8, 4, 2])
    except:
        continue


    displacement_transform = sitk.DisplacementFieldTransform(tfm)
    disp_field = sitk.TransformToDisplacementField(
        displacement_transform,
        sitk.sitkVectorFloat64,  # data type
        moving.GetSize(),
        moving.GetOrigin(),
        moving.GetSpacing(),
        moving.GetDirection()
    )

    # Apply using ResampleImageFilter
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)  # Use fixed image as reference
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)  # Background pixel value
    resampler.SetTransform(displacement_transform)

    warped_image = resampler.Execute(moving)


    fname = save_transforms_dir + f'transform_patient_{patient_id}_fixed_{fixed_id}_moving_{moving_id}.nii.gz' 
    sitk.WriteImage(disp_field, fname)

    fname = save_transformed_images_dir + f'transformed_image_patient_{patient_id}_fixed_{fixed_id}_moving_{moving_id}.nii.gz'
    sitk.WriteImage(warped_image, fname)
    
    completed.append((patient_id, fixed_id, moving_id))

    with open('completed.json','w') as f:
        json.dump(completed, f, indent = 4)

    #break

