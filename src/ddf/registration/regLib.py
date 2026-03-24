#https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/tree/master/Python

#https://simpleitk.org/doxygen/latest/html/examples.html
#https://simpleitk.org/doxygen/latest/html/ImageRegistrationMethod1_2ImageRegistrationMethod1_8py-example.html
#https://simpleitk.org/doxygen/latest/html/ImageRegistrationMethod2_2ImageRegistrationMethod2_8py-example.html
#https://simpleitk.org/doxygen/latest/html/ImageRegistrationMethod3_2ImageRegistrationMethod3_8py-example.html
#https://simpleitk.org/doxygen/latest/html/ImageRegistrationMethod4_2ImageRegistrationMethod4_8py-example.html

#B_spline registration
#https://simpleitk.org/doxygen/latest/html/ImageRegistrationMethodBSpline1_2ImageRegistrationMethodBSpline1_8py-example.html
#https://simpleitk.org/doxygen/latest/html/ImageRegistrationMethodBSpline2_2ImageRegistrationMethodBSpline2_8py-example.html
#https://simpleitk.org/doxygen/latest/html/ImageRegistrationMethodBSpline3_2ImageRegistrationMethodBSpline3_8py-example.html
#http://simpleitk.org/SimpleITK-Notebooks/01_Image_Basics.html

import numpy as np
import glob
#import pydicom
import os
#from pycimg import CImg
import nibabel as nib
#from scipy.interpolate import interpn
#from skimage.filters import gaussian
import SimpleITK as sitk
import sys
#from skimage.transform import resize

#from skimage.measure import marching_cubes
#import pyvista as pv

###############################################################################    
###############################################################################    
###############################################################################    

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

###############################################################################    
###############################################################################    
###############################################################################    
    
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

###############################################################################    
###############################################################################    
###############################################################################    

def command_iteration(method):
     print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():10.5f} : {method.GetOptimizerPosition()}")

###############################################################################    
###############################################################################    
###############################################################################    

def three_step_registration(fixed,moving,shrink_factors = [8, 4, 2],smoothing_sigmas = [8, 4, 2]):

###############################################################################    
#  First affine registration moving->fixed
    
    R = sitk.ImageRegistrationMethod()

    R.SetMetricAsCorrelation()
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0,
                                            minStep=1e-4,
                                            numberOfIterations=10,
                                            gradientMagnitudeTolerance=1e-8)
    R.SetOptimizerScalesFromIndexShift()
    tx = sitk.CenteredTransformInitializer(fixed, moving,
                                        sitk.Similarity3DTransform())
    R.SetInitialTransform(tx)
    R.SetInterpolator(sitk.sitkLinear)

    # Uncomment line below to print processing progress
    #R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    outTx = R.Execute(fixed, moving)

    # Apply computed transform to "moving image"

    #print("-------")
    #print(outTx)
    #print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    #print(f" Iteration: {R.GetOptimizerIteration()}")
    #print(f" Metric value: {R.GetMetricValue()}")

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)

    # SimpleITK moving image registered to fixed image
    out = resampler.Execute(moving)

###############################################################################    
# Next step - bspline elastic registration outTx(moving)->fixed

    transformDomainMeshSize = [5] * moving.GetDimension()
    tx1 = sitk.BSplineTransformInitializer(fixed,transformDomainMeshSize)

    #print("Initial Parameters:")
    #print(tx.GetParameters())

    R1 = sitk.ImageRegistrationMethod()
    R1.SetMetricAsCorrelation()

    R1.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
                        numberOfIterations=10,
                        maximumNumberOfCorrections=5,
                        maximumNumberOfFunctionEvaluations=1000,
                        costFunctionConvergenceFactor=1e+7)
    R1.SetInitialTransform(tx1, True)
    R1.SetInterpolator(sitk.sitkLinear)

    # tUncomment line below to print processing progress
    #R1.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R1))

    outTx1 = R1.Execute(fixed, out)

    # Apply computed transform to outTx(moving)

    #print("-------")
    #print(outTx1)
    #print(f"Optimizer stop condition: {R1.GetOptimizerStopConditionDescription()}")
    #print(f" Iteration: {R1.GetOptimizerIteration()}")
    #print(f" Metric value: {R1.GetMetricValue()}")


    resampler1 = sitk.ResampleImageFilter()
    resampler1.SetReferenceImage(fixed)
    resampler1.SetInterpolator(sitk.sitkLinear)
    resampler1.SetDefaultPixelValue(0)
    resampler1.SetTransform(outTx1)

    out1 = resampler1.Execute(out)

###############################################################################    
# Final step - diffeomorphic demons registration

    # Select a Demons filter and configure it.
    demons_filter =  sitk.DiffeomorphicDemonsRegistrationFilter()
    demons_filter.SetNumberOfIterations(60)
    # Regularization (update field - viscous, total field - elastic).
    demons_filter.SetSmoothDisplacementField(True)
    demons_filter.SetStandardDeviations(0.6)

    # create initial transform
    initial_transform = sitk.CenteredTransformInitializer(fixed, 
                                                          out1, 
                                                          sitk.Euler3DTransform(), 
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)



    # Run the registration.
    tfm = multiscale_demons(registration_algorithm=demons_filter, 
                            fixed_image = fixed,
                            moving_image = out1,
                            initial_transform = initial_transform,
                            shrink_factors = shrink_factors,
                            smoothing_sigmas = smoothing_sigmas)

    #print(f"Calculated transform: {tfm}")

    # Apply computed transform to outTx1(outTx(moving))
    out2 = sitk.Resample(out1, fixed, tfm, sitk.sitkLinear, 0, fixed.GetPixelID())  
    
    return out,out1,out2,outTx,outTx1,tfm 

###############################################################################    
###############################################################################    
###############################################################################    

def one_step_registration(fixed,moving):

###############################################################################    
# Final step - diffeomorphic demons registration

    # Select a Demons filter and configure it.
    demons_filter =  sitk.DiffeomorphicDemonsRegistrationFilter()
    demons_filter.SetNumberOfIterations(60)
    # Regularization (update field - viscous, total field - elastic).
    demons_filter.SetSmoothDisplacementField(True)
    demons_filter.SetStandardDeviations(0.6)

    # create initial transform
    initial_transform = sitk.CenteredTransformInitializer(fixed, 
                                                          moving, 
                                                          sitk.Euler3DTransform(), 
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)



    # Run the registration.
    tfm = multiscale_demons(registration_algorithm=demons_filter, 
                            fixed_image = fixed,
                            moving_image = moving,
                            initial_transform = initial_transform,
                            shrink_factors = [8, 4, 2],
                            smoothing_sigmas = [8, 4, 2])

    #print(f"Calculated transform: {tfm}")

    # Apply computed transform to outTx1(outTx(moving))
    out = sitk.Resample(moving, fixed, tfm, sitk.sitkLinear, 0, fixed.GetPixelID())  
    
    return out,tfm 

###############################################################################    
###############################################################################    
############################################################################### 

def decimateMesh(im,DECIMATION_FACTOR):

    landmarks,faces,_,_ = marching_cubes(im,0.5)

    sizes = np.array([3]*faces.shape[0],dtype=np.int32).reshape((-1,1))
    facesForMesh = np.concatenate((sizes,faces),axis = 1)
    mesh = pv.PolyData(landmarks, facesForMesh)

    # decimate mesh
    newMesh = mesh.decimate(DECIMATION_FACTOR)

    landmarks = np.asarray(newMesh.points)
 
    return landmarks

###############################################################################    
###############################################################################    
###############################################################################   

def resample(moving_img, fixed_img, transform):
    # output image Origin, Spacing, Size, Direction are taken from the fixed image in this call to Resample
    interpolator = sitk.sitkLinear
    default_value = 0.0
    return sitk.Resample(moving_img, fixed_img, transform,
                         interpolator, default_value)

###############################################################################    
###############################################################################    
###############################################################################   

def generate_inverse_transform(I_fixed, I_moving, tx_org):
    
    # Use TransformToDisplacementField to populate the displacement field
    displacement_field = sitk.TransformToDisplacementField(tx_org, sitk.sitkVectorFloat64, I_fixed.GetSize(), I_fixed.GetOrigin(), I_fixed.GetSpacing(), I_fixed.GetDirection())
    inverted_displacement_field = sitk.InvertDisplacementField(displacement_field)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(I_moving)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    new_displacement_field = resampler.Execute(inverted_displacement_field)
    inverse_transform = sitk.DisplacementFieldTransform(new_displacement_field)

    return inverse_transform

###############################################################################    
###############################################################################    
###############################################################################   

 

