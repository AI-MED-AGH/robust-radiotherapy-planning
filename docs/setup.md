# SETUP

This file serves as a short guide on how to set up the environment for using this repository and for contribution.

## Installing Dependencies

In order to install Python dependencies in the conda/virtual environment of choice, the following command should be used (it should be run at the repository root):

```
pip install -e .
```

**IMPORTANT**: `-e` should not be omitted as it installs the repository as an editable package rather than a fixed one.

## Code Quality Tools

This project uses Ruff and Mypy in order to manage code quality. The following commands are useful:

1. Checking formatting:
```
ruff format --check
```

2. Automatic formatting:
```
ruff format
```

3. Checking code quality with Ruff:
```
ruff check
```

4. Fixing auto-fixable issues with Ruff:
```
ruff check --fix
```

5. Checking code quality with Mypy:
```
mypy .
```

It is also advised to use Visual Studio Code Extensions for these 2 tools as they simplify development greatly. They should be configured to inherit the configuration from `pyproject.toml` rather than the editor.
