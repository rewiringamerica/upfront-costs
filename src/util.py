"""Utility functions that can be used across the upfront costs repo."""

from pathlib import Path

import tomllib


def get_release_version() -> str:
    """
    Retrieve the version string from the repo's root pyproject.toml file.

    Returns
    -------
    str
        The version defined under the `[tool.poetry]` table in pyproject.toml.
        Returns an empty string if the file or version key is not found.
    """
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject_path.is_file():
        return ""

    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    if "tool" in data and isinstance(data["tool"], dict):
        poetry = data["tool"].get("poetry", {})
        version = poetry.get("version")
        if version:
            return f"v{str(version)}"

    return ""
