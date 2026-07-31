"""Tests for the Colab-first model stack configuration."""

import subprocess
import sys
from pathlib import Path

import pytest

from ultimate_rvc.rvc.lib.tools import prerequisites_download
from ultimate_rvc.typing_extra import SeparationModel


def test_current_roformer_candidates_are_available() -> None:
    """Keep the registry-backed separator choices exposed to the UI."""
    assert (
        SeparationModel.BS_ROFORMER_VIPERX_1297.value
        == "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    )
    assert (
        SeparationModel.BS_ROFORMER_VIPERX_1296.value
        == "model_bs_roformer_ep_368_sdr_12.9628.ckpt"
    )
    assert (
        SeparationModel.MEL_BAND_ROFORMER_KIMBERLEY.value
        == "vocals_mel_band_roformer.ckpt"
    )


def test_colab_setup_prefetches_only_contentvec_by_default() -> None:
    """Optional language and Spin models should remain lazy downloads."""
    default_folders = {
        folder for folder, _ in prerequisites_download.default_embedders_list
    }
    optional_folders = {
        folder for folder, _ in prerequisites_download.optional_embedders_list
    }

    assert default_folders == {"embedders/contentvec/"}
    assert "embedders/spin/" in optional_folders
    assert "embedders/spin-v2/" in optional_folders
    assert default_folders.isdisjoint(optional_folders)


def test_missing_asset_detection_does_not_depend_on_head_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing Content-Length must not make setup skip the real download."""
    remote_folder = "embedders/test/"
    monkeypatch.setitem(
        prerequisites_download.folder_mapping_list,
        remote_folder,
        str(tmp_path),
    )
    mapping = [(remote_folder, ["model.bin"])]

    assert prerequisites_download.has_missing_files(mapping) is True
    (tmp_path / "model.bin").write_bytes(b"ready")
    assert prerequisites_download.has_missing_files(mapping) is False


def test_disabled_pretrained_download_has_zero_size() -> None:
    """Disabling both model groups should not schedule hidden downloads."""
    assert prerequisites_download.calculate_total_size([], models=False) == 0


def test_standard_synthesizer_import_does_not_require_triton() -> None:
    """HiFi-GAN models must stay usable when optional Triton is unavailable."""
    script = """
import builtins

original_import = builtins.__import__


def import_without_triton(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "triton" or name.startswith("triton."):
        raise ModuleNotFoundError("No module named 'triton'")
    return original_import(name, globals, locals, fromlist, level)


builtins.__import__ = import_without_triton

from ultimate_rvc.rvc.lib.algorithm.synthesizers import Synthesizer

assert Synthesizer.__name__ == "Synthesizer"
"""
    subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
