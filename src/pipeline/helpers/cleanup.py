import shutil
from pathlib import Path


def clear_directory_contents(directory: Path) -> None:
    """
    Delete every item inside an output directory while preserving the directory.

    Pipeline stages use this helper at the beginning of their public entry
    points so repeated runs do not mix new outputs with leftovers from earlier
    runs. The directory itself is retained because downstream code writes to
    the configured path directly.

    Symbolic links are treated as files and unlinked rather than traversed.
    Real subdirectories are removed recursively.

    Parameters
    ----------
    directory : Path
        Directory whose contents should be removed. The path is created if it
        does not already exist.

    Raises
    ------
    OSError
        If the directory cannot be created, or if one of its contents cannot be
        removed because of permissions, locks, or another filesystem error.
    """

    directory.mkdir(parents=True, exist_ok=True)

    for item in directory.iterdir():
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        else:
            item.unlink()
