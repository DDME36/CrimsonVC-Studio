"""Tests for selecting the full or compact CrimsonVC interface."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from ultimate_rvc.web.ui_mode import UIMode, is_cover_mode, resolve_ui_mode


def test_ui_mode_defaults_to_full_studio(monkeypatch) -> None:
    """The local and full Colab launchers should retain the complete UI."""
    monkeypatch.delenv("URVC_UI_MODE", raising=False)

    assert resolve_ui_mode() is UIMode.STUDIO
    assert not is_cover_mode()


def test_ui_mode_reads_cover_environment(monkeypatch) -> None:
    """The Lite notebook should select the compact cover workflow."""
    monkeypatch.setenv("URVC_UI_MODE", " COVER ")

    assert resolve_ui_mode() is UIMode.COVER
    assert is_cover_mode()


def test_explicit_ui_mode_overrides_environment(monkeypatch) -> None:
    """An explicit render mode should be deterministic in tests and callers."""
    monkeypatch.setenv("URVC_UI_MODE", "cover")

    assert resolve_ui_mode(UIMode.STUDIO) is UIMode.STUDIO


@pytest.mark.parametrize("invalid_mode", ["", "lite", "training", "all"])
def test_invalid_ui_mode_has_actionable_error(invalid_mode: str) -> None:
    """Invalid modes should fail before Gradio begins rendering."""
    with pytest.raises(ValueError, match="Choose one of: studio, cover"):
        resolve_ui_mode(invalid_mode)


def test_cover_hero_describes_only_the_compact_workflow() -> None:
    """Cover mode branding should not advertise hidden training controls."""
    from ultimate_rvc.web.presentation import hero_html  # noqa: PLC0415

    cover_hero = hero_html(cover_only=True)

    assert "AI cover" in cover_hero
    assert "Train" not in cover_hero
    assert "convert speech" not in cover_hero


def test_cover_events_only_target_browser_rendered_components() -> None:
    """Lite event chains must not return updates for omitted Studio controls."""
    environment = os.environ.copy()
    environment.update({"MPLBACKEND": "Agg", "URVC_UI_MODE": "cover"})
    script = """
from ultimate_rvc.web.main import app

config = app.get_config_file()
rendered_ids = {component["id"] for component in config["components"]}
invalid_outputs = [
    (dependency.get("id"), output_id)
    for dependency in config["dependencies"]
    for output_id in dependency.get("outputs", [])
    if output_id not in rendered_ids
]
print(f"invalid_outputs={invalid_outputs}")
raise SystemExit(bool(invalid_outputs))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "invalid_outputs=[]" in result.stdout
