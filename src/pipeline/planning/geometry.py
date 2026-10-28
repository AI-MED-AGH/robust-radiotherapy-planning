from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from numpy.typing import NDArray


def make_nifti(values: Any, affine: Any) -> Any:
    """
    Wrap voxel values and their world-coordinate transform in a NIfTI image.

    Parameters
    ----------
    values : array_like
        Voxel array, normally with three spatial dimensions. Values are passed
        to NiBabel without explicit dtype conversion or resampling.

    affine : array_like
        4-by-4 transform mapping voxel indices to NIfTI RAS+ world coordinates
        in millimetres for the pipeline's medical images.

    Returns
    -------
    image : nibabel.Nifti1Image
        In-memory image containing the supplied values and affine. No source
        header is copied and no file is written.
    """

    return nib.Nifti1Image(values, affine)  # type: ignore[no-untyped-call]


def canonical_image(image: Any) -> Any:
    """
    Reorder voxel axes into the closest canonical RAS+ orientation.

    Axes are permuted or flipped and the affine is updated to preserve each
    voxel's world position. This does not interpolate values or remove
    obliquity or shear from the grid.

    Parameters
    ----------
    image : nibabel.spatialimages.SpatialImage
        Image with an affine mapping voxel indices to world coordinates.

    Returns
    -------
    canonical : nibabel.spatialimages.SpatialImage
        Image whose voxel axes are ordered as close as possible to right,
        anterior, and superior. An already canonical image may be returned
        unchanged.
    """

    return nib.as_closest_canonical(image)  # type: ignore[no-untyped-call]


def resample_to_grid(image: Any, reference: Any, *, order: int) -> Any:
    """
    Resample an image onto a reference voxel grid using their world affines.

    Parameters
    ----------
    image : nibabel.spatialimages.SpatialImage
        Source image to sample. Its world-coordinate system and units must
        match those of ``reference``.

    reference : nibabel.spatialimages.SpatialImage
        Image supplying the output shape and voxel-to-world affine.

    order : int
        Spline interpolation order from 0 to 5. Use 0 for categorical labels
        or masks and 1 for linear interpolation of continuous CT values.

    Returns
    -------
    aligned : nibabel.spatialimages.SpatialImage
        Resampled image on the reference grid. Locations outside the source
        field of view are filled with zero. This aligns grids using existing
        affines; it does not estimate an anatomical registration.
    """

    return resample_from_to(image, (reference.shape, reference.affine), order=order)  # type: ignore[no-untyped-call]


def load_ct(path: Path) -> Any:
    """
    Load a finite 3D CT and require an axis-aligned canonical grid.

    Values are read as float32 for validation, then the image is reoriented
    without interpolation. No HU clipping or tissue-range check is performed
    here; the input is expected to contain CT values in HU after NiBabel's
    normal image scaling.

    Parameters
    ----------
    path : Path
        CT image file readable by NiBabel.

    Returns
    -------
    canonical : nibabel.spatialimages.SpatialImage
        CT in canonical RAS+ voxel orientation with positive diagonal spacing
        to within the affine check's tolerance. The returned image retains
        its image representation rather than the validation array.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.

    ValueError
        If CT values are not finite and three-dimensional, or the canonical
        affine contains obliquity or shear unsupported by this OpenTPS bridge.
    """

    image: Any = nib.load(str(path))
    values = image.get_fdata(dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError(f"CT must contain finite 3D HU values: {path}")
    # OpenTPS/MCsquare cannot represent an arbitrary oblique NIfTI grid
    canonical = canonical_image(image)
    if not np.allclose(canonical.affine[:3, :3], np.diag(canonical.header.get_zooms()[:3]), atol=1e-4):
        raise ValueError(f"Oblique/sheared CT must be resampled to an axis-aligned grid first: {path}")
    return canonical


def pipeline_grid(planning_image: Any, shape: tuple[int, int, int]) -> Any:
    """
    Reconstruct the world affine of MAISI's padded and centre-cropped tensor.

    The offset follows symmetric spatial padding and centre cropping, including
    their integer rounding for odd sizes. Voxel spacing and axis directions
    stay unchanged; only the world position of voxel index zero is shifted.

    Parameters
    ----------
    planning_image : nibabel.spatialimages.SpatialImage
        Canonical RAS+ 3D planning CT before pipeline padding and cropping,
        normally returned by ``load_ct``.

    shape : tuple[int, int, int]
        Positive target spatial dimensions used by the tensor preprocessing.

    Returns
    -------
    grid : nibabel.Nifti1Image
        Zero-filled uint8 image with the tensor's shape and reconstructed
        affine. This image describes geometry only; it contains no CT values.
    """

    original_shape = np.asarray(planning_image.shape)
    target_shape = np.asarray(shape)
    padding_before = np.maximum(target_shape - original_shape, 0) // 2
    padded_shape = np.maximum(original_shape, target_shape)
    crop_before = padded_shape // 2 - target_shape // 2
    affine = planning_image.affine.copy()
    affine[:3, 3] += affine[:3, :3] @ (crop_before - padding_before)
    return make_nifti(np.zeros(shape, dtype=np.uint8), affine)


def labels_on_grid(path: Path, reference: Any) -> NDArray[np.int16]:
    """
    Load a structure label map and resample it onto a reference grid.

    Parameters
    ----------
    path : Path
        File containing finite, integer-valued 3D structure labels. Label
        values are expected to fit in int16; their range is not checked here.

    reference : nibabel.spatialimages.SpatialImage
        Image defining the output shape and affine in the same world frame.

    Returns
    -------
    labels : numpy.ndarray
        Int16 label array with the reference shape. Nearest-neighbour sampling
        preserves categorical labels, and out-of-bounds locations receive
        background label zero. The array itself does not retain an affine.

    Raises
    ------
    FileNotFoundError
        If the label-map file does not exist.

    ValueError
        If the label map is not three-dimensional, contains non-finite values,
        or its values are not close to integers under ``numpy.allclose``.
    """

    image: Any = nib.load(str(path))
    values = image.get_fdata()
    if values.ndim != 3 or not np.isfinite(values).all() or not np.allclose(values, np.rint(values)):
        raise ValueError(f"Structure map must contain finite integer 3D labels: {path}")
    aligned = resample_to_grid(image, reference, order=0)
    return np.asarray(aligned.dataobj).round().astype(np.int16)


def to_lps(image: Any) -> Any:
    """
    Reorient voxel axes toward left, posterior, and superior.

    The voxel array is permuted or flipped without interpolation, and its
    affine is updated to preserve world positions. The affine still expresses
    NIfTI RAS+ world coordinates; converting the world-coordinate basis to
    LPS is a separate step performed by ``opentps_image``.

    Parameters
    ----------
    image : nibabel.spatialimages.SpatialImage
        Spatial image with a valid voxel-to-world affine.

    Returns
    -------
    lps : nibabel.spatialimages.SpatialImage
        Image with LPS voxel-axis orientation. Any obliquity or shear remains.
    """

    orientation = nib.orientations.ornt_transform(  # type: ignore[no-untyped-call]
        nib.orientations.io_orientation(image.affine),  # type: ignore[no-untyped-call]
        nib.orientations.axcodes2ornt(("L", "P", "S")),  # type: ignore[no-untyped-call]
    )
    return image.as_reoriented(orientation)


def opentps_image(image: Any, image_class: Any, *, name: str) -> Any:
    """
    Convert an axis-aligned NIfTI image into an OpenTPS image in LPS space.

    First reorient voxel axes to LPS, then negate the RAS world x and y
    coordinates to obtain an LPS origin and positive voxel spacing. The voxel
    values are copied so the OpenTPS image owns a separate array.

    Parameters
    ----------
    image : nibabel.spatialimages.SpatialImage
        Axis-aligned 3D image with an affine in RAS+ millimetres. Callers must
        ensure axis alignment, for example with ``load_ct``; off-diagonal
        affine terms are not represented or validated by this conversion.

    image_class : type
        OpenTPS image class accepting ``imageArray``, ``name``, ``origin``,
        and ``spacing``, such as ``CTImage`` or ``ROIMask``.

    name : str
        Name assigned to the new OpenTPS image.

    Returns
    -------
    converted : object
        Instance of ``image_class`` containing the reoriented values and LPS
        geometry. Intensities and physical units are unchanged.
    """

    lps = to_lps(image)
    affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps.affine
    return image_class(
        imageArray=np.asarray(lps.dataobj).copy(),
        name=name,
        origin=affine[:3, 3],
        spacing=np.diag(affine[:3, :3]),
    )


def dose_to_nifti(dose: Any, path: Path) -> None:
    """
    Validate an OpenTPS dose grid and save it with a NIfTI RAS+ world affine.

    The affine is built from the LPS spacing and origin, then its world x and
    y coordinates are negated. Voxel order is retained, so the saved image
    normally has LPS voxel-axis orientation despite its RAS+ world affine.

    Parameters
    ----------
    dose : opentps.core.data.images.DoseImage
        Dose object supplying ``imageArray``, ``spacing``, ``origin``, and
        ``angles``. Values are expected in Gy and are converted to float32
        without dose normalization or unit conversion.

    path : Path
        Output NIfTI path, normally ending in ``.nii`` or ``.nii.gz``. Its
        parent directory must already exist; an existing file is overwritten.

    Returns
    -------
    None
        The dose array and its world-coordinate affine are written to disk.

    Raises
    ------
    ValueError
        If float32 dose values are not finite and three-dimensional, contain
        negative values, or contain no positive value; also if dose grid
        angles are not close to zero under ``numpy.allclose``.

    OSError
        If the output file cannot be written.
    """

    values = np.asarray(dose.imageArray, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all() or (values < 0).any() or not (values > 0).any():
        raise ValueError("OpenTPS returned an invalid, negative or empty dose distribution")
    if not np.allclose(dose.angles, 0):
        raise ValueError("OpenTPS returned an unsupported rotated dose grid")
    affine = np.eye(4)
    affine[:3, :3] = np.diag(dose.spacing)
    affine[:3, 3] = dose.origin
    affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine
    nib.save(make_nifti(values, affine), str(path))
