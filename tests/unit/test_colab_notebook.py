"""Structural and syntax tests for the primary Colab notebook."""

from __future__ import annotations

from typing import Any

import ast
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "CrimsonVC_Colab.ipynb"


def _load_notebook() -> dict[str, Any]:
    """Load the primary Colab notebook as JSON."""
    with NOTEBOOK_PATH.open(encoding="utf-8") as notebook_file:
        return json.load(notebook_file)


class TestColabNotebook:
    """Test the primary Colab notebook without requiring a GPU."""

    def test_notebook_structure(self) -> None:
        """Test notebook format, cell identifiers, and clean outputs."""
        notebook = _load_notebook()
        cells = notebook["cells"]
        cell_ids = [cell["metadata"]["id"] for cell in cells]

        assert notebook["nbformat"] == 4
        assert notebook["nbformat_minor"] == 5
        assert len(cells) == 8
        assert len(cell_ids) == len(set(cell_ids))

        for cell in cells:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []

    def test_all_code_cells_are_valid_python(self) -> None:
        """Test that every code cell parses without notebook magics."""
        notebook = _load_notebook()
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                ast.parse(source, filename=f"{NOTEBOOK_PATH.name}:cell_{index}")

    def test_repository_uses_the_published_url(self) -> None:
        """Test that the notebook clones the published repository by default."""
        notebook = _load_notebook()
        repository_cell = next(
            cell for cell in notebook["cells"] if cell["metadata"]["id"] == "repository"
        )
        source = "".join(repository_cell["source"])

        assert "https://github.com/DDME36/CrimsonVC-Studio.git" in source
        assert "YOUR_USERNAME" not in source
        assert 'Path("/content/CrimsonVC")' in source
        assert "subprocess.Popen" in source
        assert "stdout=subprocess.PIPE" in source
        assert "stderr=subprocess.STDOUT" in source

    def test_runtime_only_storage_is_the_default(self) -> None:
        """Test that inference does not require mounting Google Drive."""
        notebook = _load_notebook()
        storage_cell = next(
            cell for cell in notebook["cells"] if cell["metadata"]["id"] == "storage"
        )
        source = "".join(storage_cell["source"])

        assert 'storage_mode = "runtime_only"' in source
        assert 'if storage_mode == "google_drive"' in source
        assert 'Path("/content/CrimsonVC-data")' in source

    def test_install_uses_cuda_extra_and_lock_when_available(self) -> None:
        """Test that the install cell follows the project CUDA setup."""
        notebook = _load_notebook()
        install_cell = next(
            cell for cell in notebook["cells"] if cell["metadata"]["id"] == "install"
        )
        source = "".join(install_cell["source"])

        assert '"--extra"' in source
        assert '"cuda"' in source
        assert "uv.lock" in source
        assert '"--locked"' in source
        assert "URVC_DOWNLOAD_ALL_EMBEDDERS" in source

    def test_optional_benchmark_cell_uses_project_runner(self) -> None:
        """Test that model checks are available without running by default."""
        notebook = _load_notebook()
        benchmark_cell = next(
            cell for cell in notebook["cells"] if cell["metadata"]["id"] == "benchmark"
        )
        source = "".join(benchmark_cell["source"])

        assert "run_model_benchmark = False" in source
        assert "scripts/benchmark_models.py" in source
        assert 'benchmark_suite = "inventory"' in source

    def test_launch_uses_colab_entry_point(self) -> None:
        """Test that the launch cell uses the authenticated entry point."""
        notebook = _load_notebook()
        launch_cell = next(
            cell for cell in notebook["cells"] if cell["metadata"]["id"] == "launch"
        )
        source = "".join(launch_cell["source"])

        assert "getpass" in source
        assert "URVC_AUTH_USERNAME" in source
        assert "URVC_AUTH_PASSWORD" in source
        assert 'WORKSPACE / ".venv" / "bin" / "python"' in source
        assert 'WORKSPACE / "src"' in source
        assert 'launch_env["PYTHONPATH"]' in source
        assert 'launch_env["MPLBACKEND"] = "Agg"' in source
        assert '"-u"' in source
        assert '[uv, "run"' not in source
        assert "ultimate_rvc.web.colab" in source
