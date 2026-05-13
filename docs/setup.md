# SETUP

This file serves as a short guide on how to set up the environment for using this repository and for contribution.

## Installing Dependencies

This project uses `uv` to manage dependencies. In order to properly use the project and contribute, it should be installed through Windows PowerShell:

```
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Once `uv` is installed it can be used to sync the `.venv` or a Python installation with `uv.lock`. The command used depends on where the sync happens:

1. Syncing to a `.venv`:
```
uv sync --python python
```

2. Syncing directly to Python:
```
uv pip sync pyproject.toml --python python --system
```

**IMPORTANT**: NEVER sync directly to Python when using `conda` as that can completely break the `conda` environment (`uv` will remove conda-specific packages).

This project uses PyTorch. The default environment uses CPU, but for heavier code it is preferred to use GPU. The project directly manages a CUDA 12.8 PyTorch installation. This may not work for everyone, so adding more versions can be discussed in the future. In order to install this version of PyTorch instead, one should add the flag `--extra gpu`:

1. Syncing to a `.venv`:
```
uv sync --python python --extra gpu
```

2. Syncing directly to Python:
```
uv pip sync pyproject.toml --python python --system --extra gpu
```

**IMPORTANT**: Make sure to always sync if there is a change to the environment to not use outdated packages.

## Adding Dependencies

Sometimes additional dependencies may have to be added. While first writing code it is best to use `pip` or `uv pip` to find versions of packages which work well. Once those versions are found, the packages can be added to `uv` through a command:
```
uv add "<package_name>=<lower_bound>,<<upper_bound>"
```

- <package_name> - the package to be added.
- <lower_bound> - the version which is found to work well.
- <upper_bound> - the next major version (those often introduce breaking changes, this should be specified even if there is no new major version).

Here is an example with NumPy:
```
uv add "numpy>=2.4.4,<3.0.0"
```

## Code Quality Tools

This project uses Ruff and Mypy in order to manage code quality. The following commands are useful:

1. Checking formatting:
```
uv run ruff format --check
```

2. Automatic formatting:
```
uv run ruff format
```

3. Checking code quality with Ruff:
```
uv run ruff check
```

4. Fixing auto-fixable issues with Ruff:
```
uv run ruff check --fix
```

5. Checking code quality with Mypy:
```
uv run mypy .
```

It is also advised to use Visual Studio Code Extensions for these 2 tools as they simplify development greatly. They should be configured to inherit the configuration from `pyproject.toml` rather than the editor.
