"""Regression tests for Python package metadata and build layout."""

from __future__ import annotations

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]


def test_distribution_maps_to_the_compatible_module() -> None:
    """Ensure rebranding does not break uv package builds."""
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)

    assert project["project"]["name"] == "crimson-vc-studio"
    assert project["tool"]["uv"]["build-backend"]["module-name"] == "ultimate_rvc"
    assert (PROJECT_ROOT / "src" / "ultimate_rvc" / "__init__.py").is_file()