import mmap
from collections.abc import Iterator, Sequence
from pathlib import Path
from struct import Struct
from tempfile import TemporaryFile
from typing import Any

import numpy as np
from scipy.sparse import csc_matrix

# Each spot begins with the voxel count, beam ID, layer ID, and XY position
_SPOT_HEADER = Struct("<IIIff")
_RUN_HEADER = Struct("<II")
_DOSE_DTYPE = np.dtype("<f4")


def _column_runs(source: mmap.mmap, n_voxels: int, n_spots: int) -> Iterator[np.ndarray[Any, Any]]:
    """
    Parse the voxel runs for each spot in a mapped MCsquare dose file.

    Each spot header gives the total number of stored dose values. Each
    subsequent run gives the count and first voxel index, followed by that
    many little-endian float32 values.

    Parameters
    ----------
    source : mmap.mmap
        Mapped contents of the MCsquare sparse beamlet binary.

    n_voxels : int
        Number of voxels on the full scoring grid.

    n_spots : int
        Number of spot records expected in the file.

    Yields
    ------
    numpy.ndarray
        An int64 array with three columns for each spot. The columns give the
        inclusive start voxel, exclusive end voxel, and byte offset of the
        first dose value. An empty spot yields an array with zero rows.

    Raises
    ------
    ValueError
        If a spot or run is truncated, the runs overlap or fall outside the
        grid, the run counts are invalid, or the file has trailing bytes.
    """

    offset = 0
    size = len(source)
    for column in range(n_spots):
        if offset + _SPOT_HEADER.size > size:
            raise ValueError(f"MCsquare sparse dose file is missing spot {column + 1}/{n_spots}")
        n_values, *_ = _SPOT_HEADER.unpack_from(source, offset)
        offset += _SPOT_HEADER.size
        runs = []
        remaining = n_values
        previous_end = 0
        while remaining:
            if offset + _RUN_HEADER.size > size:
                raise ValueError(f"Truncated MCsquare sparse dose run in spot {column + 1}")
            count, start = _RUN_HEADER.unpack_from(source, offset)
            dose_offset = offset + _RUN_HEADER.size
            end = start + count
            offset = dose_offset + count * _DOSE_DTYPE.itemsize
            if not count or count > remaining or start < previous_end or end > n_voxels or offset > size:
                raise ValueError(f"Invalid MCsquare sparse dose run in spot {column + 1}")
            runs.append((start, end, dose_offset))
            remaining -= count
            previous_end = end
        yield np.asarray(runs, dtype=np.int64).reshape(-1, 3)
    if offset != size:
        raise ValueError("MCsquare sparse dose file contains data after the last expected spot")


def read_objective_matrix(
    path: Path,
    n_voxels: int,
    rows: np.ndarray[Any, Any],
    scaling: Sequence[float],
    *,
    scratch_dir: Path | None = None,
) -> csc_matrix:
    """
    Build a compact float32 CSC matrix for the selected scoring-grid rows.

    Parameters
    ----------
    path : Path
        Path to the MCsquare sparse beamlet binary.
    n_voxels : int
        Number of voxels on the full scoring grid.
    rows : numpy.ndarray
        One-dimensional array of unique voxel indices in ascending MCsquare
        order. Each index must be within the full scoring grid.
    scaling : Sequence[float]
        One factor per spot, in file order, to convert the stored coefficients
        to dose per monitor unit. Its length is the expected spot count.
    scratch_dir : Path or None
        Directory for the temporary matrix files. Defaults to the input file's
        directory.

    Returns
    -------
    scipy.sparse.csc_matrix
        Float32 matrix with rows in the order of ``rows`` and one column per
        spot, including spots with no dose in the selected voxels.

    Raises
    ------
    OSError
        If the binary file cannot be opened or mapped, or scratch writes fail.
    ValueError
        If the binary file is empty or contains invalid spot, run, or dose data.

    Notes
    -----
    The first pass validates the file layout and counts retained coefficients.
    The second writes one spot at a time and maps the finished arrays read-only.
    Multiple scenario matrices can be used together while the OS reclaims pages
    from their temporary files as needed. The mappings own the backing storage.
    """

    scales = np.asarray(scaling, dtype=np.float32)
    n_spots = len(scales)
    scratch_dir = scratch_dir if scratch_dir is not None else path.parent
    with (
        path.open("rb") as stream,
        TemporaryFile(dir=scratch_dir) as index_file,
        TemporaryFile(dir=scratch_dir) as data_file,
    ):
        if path.stat().st_size == 0:
            raise ValueError(f"MCsquare sparse dose file is empty: {path}")
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as source:
            # Count intersections first to choose the CSC index width and column offsets
            indptr = np.zeros(n_spots + 1, dtype=np.int64)
            for column, runs in enumerate(_column_runs(source, n_voxels, n_spots)):
                bounds = np.searchsorted(rows, runs[:, :2])
                indptr[column + 1] = (bounds[:, 1] - bounds[:, 0]).sum()
            np.cumsum(indptr, out=indptr)
            if max(int(indptr[-1]), len(rows), n_spots) <= np.iinfo(np.int32).max:
                indptr = indptr.astype(np.int32)
            for column, runs in enumerate(_column_runs(source, n_voxels, n_spots)):
                n_values = int(indptr[column + 1] - indptr[column])
                if not n_values:
                    continue
                indices = np.empty(n_values, dtype=indptr.dtype)
                data = np.empty(n_values, dtype=np.float32)
                output_start = 0
                bounds = np.searchsorted(rows, runs[:, :2])
                active = bounds[:, 0] < bounds[:, 1]
                for (start, end), (run_start, run_end, dose_offset) in zip(bounds[active], runs[active], strict=True):
                    # Selected CT rows need not be contiguous within the dose run
                    values = np.frombuffer(
                        source, dtype=_DOSE_DTYPE, count=int(run_end - run_start), offset=int(dose_offset)
                    )
                    output_end = output_start + end - start
                    try:
                        data[output_start:output_end] = values[rows[start:end] - run_start] * scales[column]
                    finally:
                        del values
                    indices[output_start:output_end] = np.arange(start, end, dtype=indices.dtype)
                    output_start = output_end
                if not np.isfinite(data).all() or (data < 0).any():
                    raise ValueError(f"Invalid MCsquare dose coefficients in {path}, spot {column + 1}")
                indices.tofile(index_file)
                data.tofile(data_file)

        nnz = int(indptr[-1])
        if not nnz:
            return csc_matrix((len(rows), n_spots), dtype=np.float32)
        indices = np.memmap(index_file, mode="r", dtype=indptr.dtype, shape=(nnz,))
        data = np.memmap(data_file, mode="r", dtype=np.float32, shape=(nnz,))
    return csc_matrix((data, indices, indptr), shape=(len(rows), n_spots), copy=False)
