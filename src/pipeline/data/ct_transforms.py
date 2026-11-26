from monai.transforms import CenterSpatialCropd, SpatialPadd  # type: ignore[attr-defined]


def ct_pad_and_crop(shape: tuple[int, int, int]) -> tuple[SpatialPadd, CenterSpatialCropd]:
    """
    Create the padding and cropping transforms for a channel-first CT.

    CT preprocessing and planning grid reconstruction use the same transforms
    so that MONAI places voxel zero at the same world position in both grids,
    including when padding or cropping splits unevenly across an axis.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Target voxel counts in the order of the CT's three spatial axes.

    Returns
    -------
    transforms : tuple[SpatialPadd, CenterSpatialCropd]
        Apply these transforms in order to the dictionary entry ``"image"``.
        The first pads undersized axes with zeros on either side. The second
        center-crops oversized axes to the target dimensions.
    """

    return (
        SpatialPadd(keys="image", spatial_size=shape, mode="constant", constant_values=0),
        CenterSpatialCropd(keys="image", roi_size=shape),
    )
