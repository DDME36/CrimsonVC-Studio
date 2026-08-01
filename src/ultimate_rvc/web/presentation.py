"""Presentation helpers for the CrimsonVC Studio Gradio shell."""

from __future__ import annotations

import os
from html import escape
from pathlib import Path

from ultimate_rvc.branding import APP_NAME, RELEASE_LABEL, TAGLINE


def load_app_styles() -> str:
    """Load the app stylesheet using UTF-8."""
    return (Path(__file__).parent / "config/styles.css").read_text(encoding="utf-8")


def hero_html(*, cover_only: bool = False) -> str:
    """Return the product header for either the full or compact workflow."""
    if cover_only:
        heading = "Make an AI cover.<br>Keep the performance."
        description = (
            "A focused RVC Cover workspace for loading a voice model, "
            "converting a song, and downloading the result."
        )
        workflows = (
            ("01 / Voice model", "Download or upload a compatible RVC model."),
            ("02 / AI Cover", "Separate, convert, and mix the song in one click."),
            ("03 / Export", "Preview the finished cover and download the audio."),
        )
    else:
        heading = "Shape a voice.<br>Keep the performance."
        description = (
            f"{TAGLINE}. Create song covers, convert speech, and train a custom "
            "voice model from one focused workspace."
        )
        workflows = (
            (
                "01 / Create",
                "Generate a song cover or convert speech with a voice model.",
            ),
            ("02 / Train", "Start with a safe preset, then tune advanced controls."),
            ("03 / Manage", "Keep models, datasets, and generated audio organized."),
        )

    workflow_html = "".join(
        (
            '<div class="crimson-workflow">'
            f"<strong>{label}</strong><span>{summary}</span></div>"
        )
        for label, summary in workflows
    )
    return f"""
    <section class="crimson-hero" aria-labelledby="crimson-title">
      <div class="crimson-brand-row">
        <div class="crimson-logo">
          <span class="crimson-logo-mark" aria-hidden="true">C</span>
          <span>{APP_NAME}</span>
        </div>
        <span class="crimson-release">{RELEASE_LABEL}</span>
      </div>
      <h1 id="crimson-title">{heading}</h1>
      <p>{description}</p>
      <div class="crimson-workflows" aria-label="Primary workflows">
        {workflow_html}
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
