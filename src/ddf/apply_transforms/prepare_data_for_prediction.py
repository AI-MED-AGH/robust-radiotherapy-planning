from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
from skimage.transform import resize


def prepare_moving_images_for_inference(
    data_file: Path,
    data_dir: Path,
    save_dir: Path,
    dshape: tuple[int, int] = (256, 256),
    factor: float = 2.0,
    num_channels: int = 4,
) -> None:
    """
    Prepare resized moving CT images for inference and save them as multiple channel-specific
    NIfTI files per patient.

    Assumptions:
    - `data_file` points to a valid JSON file containing a `"test"` split.
    - Test items contain `"moving_image"` paths following the pattern `Patient_<id>_fraction_<id>_.nii.gz`.
    - Moving CT images are stored in `data_dir / Patient_<id> / Patient_<id>_fraction_1_.nii.gz`.
    - Input CT images are 3D NIfTI volumes.
    - `dshape` defines the resized in-plane shape `(height, width)`.
    - `factor` matches the in-plane resizing ratio used to update the affine matrix.
    - The output directory is writable.

    Returns
    -------
        None:
            The function saves resized moving images to `save_dir` as
            `image_<pid>_0000.nii.gz`, ..., `image_<pid>_<channel>.nii.gz`.

    Parameters
    ----------
        data_file : Path
            Path to the JSON file containing the test split definition.
        data_dir : Path
            Directory containing patient CT subdirectories.
        save_dir : Path
            Output directory where resized moving images will be saved.
        dshape : tuple[int, int]
            Target in-plane shape `(height, width)` used during resizing.
        factor : float
            Scaling factor applied to the first two affine diagonal elements after resizing.
        num_channels : int
            Number of identical channel-specific output files to save per patient.
    """
    if not data_file.is_file():
        raise FileNotFoundError(f"Data split file does not exist: {data_file}")

    if not data_dir.is_dir():
        raise FileNotFoundError(f"CT data directory does not exist: {data_dir}")

    save_dir.mkdir(parents=True, exist_ok=True)

    with open(data_file) as f:
        data_dict = json.load(f)

    if "test" not in data_dict:
        raise ValueError(f'Missing required key "test" in JSON file: {data_file}')

    if not isinstance(data_dict["test"], list):
        raise ValueError(f'"test" must be a list in JSON file: {data_file}')

    for idx, item in enumerate(data_dict["test"]):
        if "moving_image" not in item:
            raise ValueError(f'Missing "moving_image" in test item at index {idx}')

    ids = sorted({Path(item["moving_image"]).name.split("_")[1] for item in data_dict["test"]})

    failed_patients: list[tuple[str, str]] = []

    for pid in ids:
        try:
            print(f"Processing patient {pid}")

            image_path = data_dir / f"Patient_{pid}" / f"Patient_{pid}_fraction_1_.nii.gz"
            if not image_path.is_file():
                raise FileNotFoundError(f"Moving image does not exist for patient {pid}: {image_path}")

            nifti = nib.load(str(image_path))
            fixed_img = nifti.get_fdata()  # type: ignore
            aff = nifti.affine.copy()  # type: ignore

            if fixed_img.ndim != 3:
                raise ValueError(
                    f"Expected a 3D image for patient {pid}, but got shape {fixed_img.shape}"
                )

            resized_fixed_img = resize(
                fixed_img,
                dshape + (fixed_img.shape[2],),
                anti_aliasing=True,
                preserve_range=True,
            )  # type: ignore

            aff[0, 0] *= factor
            aff[1, 1] *= factor

            nifti_image = nib.Nifti1Image(resized_fixed_img, affine=aff)  # type: ignore

            for channel_idx in range(num_channels):
                output_path = save_dir / f"image_{pid}_{channel_idx:04d}.nii.gz"
                nib.save(nifti_image, str(output_path))

            print(f"Finished patient {pid}")

        except Exception as e:
            print(f"Failed patient {pid}: {e}")
            failed_patients.append((pid, str(e)))
            continue

    if failed_patients:
        print("\nFailed patients summary:")
        for pid, error_msg in failed_patients:
            print(f"- Patient {pid}: {error_msg}")
    else:
        print("\nAll patients processed successfully.")