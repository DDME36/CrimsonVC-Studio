"""Presentation helpers for the CrimsonVC Studio Gradio shell."""

from __future__ import annotations

from html import escape
import os
from pathlib import Path

from ultimate_rvc.branding import APP_NAME, RELEASE_LABEL, TAGLINE


def load_app_styles() -> str:
    """Load the app stylesheet using UTF-8."""
    return (Path(__file__).parent / "config/styles.css").read_text(encoding="utf-8")


def hero_html() -> str:
    """Return the static product header."""
    return f"""
    <section class="crimson-hero" aria-labelledby="crimson-title">
      <div class="crimson-brand-row">
        <div class="crimson-logo">
          <span class="crimson-logo-mark" aria-hidden="true">C</span>
          <span>{APP_NAME}</span>
        </div>
        <span class="crimson-release">{RELEASE_LABEL}</span>
      </div>
      <h1 id="crimson-title">Shape a voice.<br>Keep the performance.</h1>
      <p>{TAGLINE}. Create song covers, convert speech, and train a custom
      voice model from one focused workspace.</p>
      <div class="crimson-workflows" aria-label="Primary workflows">
        <div class="crimson-workflow">
          <strong>01 / Create</strong>
          <span>Generate a song cover or convert speech with a voice model.</span>
        </div>
        <div class="crimson-workflow">
          <strong>02 / Train</strong>
          <span>Start with a safe preset, then tune advanced controls if needed.</span>
        </div>
        <div class="crimson-workflow">
          <strong>03 / Manage</strong>
          <span>Keep models, datasets, and generated audio organized.</span>
        </div>
      </div>
    </section>
    """


def runtime_status_html() -> str:
    """Return a compact runtime, accelerator, and storage summary."""
    runtime_name = (
        "Google Colab"
        if os.environ.get("COLAB_RELEASE_TAG")
        or os.environ.get("COLAB_BACKEND_VERSION")
        else "Local runtime"
    )
    storage_name = (
        "Google Drive connected"
        if Path("/content/drive/MyDrive").is_dir()
        else "Runtime / local disk"
    )
    accelerator_name = "CPU"
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            vram_gb = properties.total_memory / 1024**3
            accelerator_name = (
                f"{torch.cuda.get_device_name(0)} - {vram_gb:.1f} GB VRAM"
            )
    except (ImportError, RuntimeError):
        accelerator_name = "Accelerator unavailable"

    cards = [
        ("Runtime", runtime_name),
        ("Accelerator", accelerator_name),
        ("Storage", storage_name),
    ]
    rendered_cards = "".join(
        (
            '<div class="crimson-status-card">'
            f"<span>{escape(label)}</span><strong>{escape(value)}</strong>"
            "</div>"
        )
        for label, value in cards
    )
    return (
        '<section class="crimson-runtime" aria-label="Runtime status">'
        f"{rendered_cards}</section>"
    )
