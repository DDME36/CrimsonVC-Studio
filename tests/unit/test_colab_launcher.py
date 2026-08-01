"""Tests for the isolated Colab Web UI launcher."""

from __future__ import annotations

import os

from ultimate_rvc.web.colab import _configure_headless_matplotlib


def test_colab_replaces_notebook_only_matplotlib_backend(monkeypatch) -> None:
    """The Gradio process must not inherit Colab's unavailable inline backend."""
    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")

    _configure_headless_matplotlib()

    assert os.environ["MPLBACKEND"] == "Agg"
