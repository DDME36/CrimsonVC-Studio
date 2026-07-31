"""Guided training controls for CrimsonVC Studio."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dataclasses import dataclass
from pathlib import Path

import gradio as gr

from ultimate_rvc.common import TRAINING_MODELS_DIR
from ultimate_rvc.training_presets import (
    GOAL_LABELS,
    TrainingGoal,
    format_preset_summary,
    get_training_preset,
)
from ultimate_rvc.typing_extra import DeviceType

if TYPE_CHECKING:
    from ultimate_rvc.web.config.main import MultiStepTrainingConfig


@dataclass(frozen=True)
class GuidedTrainingControls:
    """Components created before the detailed training workflow."""

    goal: gr.Radio
    apply_button: gr.Button
    preset_summary: gr.Markdown
    health_button: gr.Button
    health_report: gr.Markdown


def render_guided_controls() -> GuidedTrainingControls:
    """Render the beginner-friendly training entry point."""
    gr.HTML(
        """
        <div class="training-steps">
          <strong>Training flow</strong><br>
          1. Prepare dataset &nbsp;&rarr;&nbsp; 2. Extract voice features
          &nbsp;&rarr;&nbsp; 3. Train and export
        </div>
        """,
    )
    with gr.Group(elem_classes=["quick-start-panel"]):
        gr.Markdown(
            """
            ## Quick Train

            Choose a goal and apply a safe starting preset. This only fills the
            controls; it never starts training automatically. Detailed controls
            remain available in each **Advanced options** section.
            """,
        )
        with gr.Row(equal_height=True):
            goal = gr.Radio(
                choices=[
                    (label, training_goal.value)
                    for training_goal, label in GOAL_LABELS.items()
                ],
                value=TrainingGoal.SPEECH.value,
                label="Training goal",
                scale=3,
            )
            apply_button = gr.Button(
                "Apply recommended settings",
                variant="primary",
                scale=2,
            )
        preset_summary = gr.Markdown(
            "Select a goal, then apply the preset. Hardware-aware batch size and "
            "precision will be chosen for the current runtime.",
        )
        with gr.Row(equal_height=True):
            health_button = gr.Button(
                "Analyze preprocessed dataset",
                variant="secondary",
            )
            gr.Markdown(
                "Run this after Step 1 to check clip count and usable duration.",
            )
        health_report = gr.Markdown(
            "Dataset report will appear here.",
            elem_classes=["dataset-health"],
        )
    return GuidedTrainingControls(
        goal=goal,
        apply_button=apply_button,
        preset_summary=preset_summary,
        health_button=health_button,
        health_report=health_report,
    )


def bind_guided_training(
    controls: GuidedTrainingControls,
    tab_config: MultiStepTrainingConfig,
) -> None:
    """Bind preset and dataset-report events after all controls exist."""
    controls.apply_button.click(
        _apply_preset,
        inputs=controls.goal,
        outputs=[
            tab_config.sample_rate.instance,
            tab_config.f0_method.instance,
            tab_config.embedder_model.instance,
            tab_config.num_epochs.instance,
            tab_config.batch_size.instance,
            tab_config.detect_overtraining.instance,
            tab_config.overtraining_threshold.instance,
            tab_config.vocoder.instance,
            tab_config.index_algorithm.instance,
            tab_config.pretrained_type.instance,
            tab_config.save_interval.instance,
            tab_config.training_acceleration.instance,
            tab_config.precision.instance,
            tab_config.preload_dataset.instance,
            tab_config.reduce_memory_usage.instance,
            controls.preset_summary,
        ],
        show_progress="hidden",
    )
    controls.health_button.click(
        analyze_preprocessed_dataset,
        inputs=tab_config.preprocess_model.instance,
        outputs=controls.health_report,
        show_progress="minimal",
    )


def _apply_preset(goal: str) -> list[object]:
    """Apply the selected preset and return values in UI output order."""
    cuda_available, bf16_supported, vram_gb, gpu_name = _hardware_profile()
    preset = get_training_preset(
        goal,
        cuda_available=cuda_available,
        bf16_supported=bf16_supported,
        vram_gb=vram_gb,
    )
    summary = format_preset_summary(
        preset,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
    )
    return [
        preset.sample_rate,
        preset.f0_method,
        preset.embedder_model,
        preset.num_epochs,
        preset.batch_size,
        preset.detect_overtraining,
        preset.overtraining_threshold,
        preset.vocoder,
        preset.index_algorithm,
        preset.pretrained_type,
        preset.save_interval,
        DeviceType.AUTOMATIC,
        preset.precision,
        preset.preload_dataset,
        preset.reduce_memory_usage,
        summary,
    ]


def _hardware_profile() -> tuple[bool, bool, float | None, str]:
    """Inspect the first CUDA device without making CUDA mandatory."""
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            return (
                True,
                torch.cuda.is_bf16_supported(),
                properties.total_memory / 1024**3,
                torch.cuda.get_device_name(0),
            )
    except (ImportError, RuntimeError):
        pass
    return False, False, None, "CPU"


def analyze_preprocessed_dataset(model_name: str | None) -> str:
    """Summarize the clips produced by dataset preprocessing."""
    if not model_name or not model_name.strip():
        return (
            "**Dataset check:** Enter or select a model name in Step 1, "
            "preprocess it, then run this check."
        )
    sliced_dir = TRAINING_MODELS_DIR / model_name.strip() / "sliced_audios"
    if not sliced_dir.is_dir():
        return (
            f"**Dataset check:** No preprocessed clips found for `{model_name}`. "
            "Complete Step 1 first."
        )

    duration_seconds, readable, unreadable = _scan_audio_duration(sliced_dir)
    duration_minutes = duration_seconds / 60
    if readable == 0:
        return (
            "**Dataset check:** No readable WAV clips were found. Inspect the "
            "source files and run preprocessing again."
        )
    if duration_minutes < 2:
        status = "Needs more data"
        advice = (
            "This is only suitable for a pipeline test. Add more clean, isolated "
            "voice audio before a long training run."
        )
    elif duration_minutes < 10:
        status = "Usable draft"
        advice = (
            "You can test a short run, but additional clean audio will usually "
            "improve stability and generalization."
        )
    else:
        status = "Ready to evaluate"
        advice = (
            "The duration is a reasonable starting point. Listen to several "
            "preview clips for noise, reverb, clipping, and instrument bleed."
        )

    unreadable_note = (
        f" · `{unreadable}` unreadable file(s)" if unreadable else ""
    )
    return (
        f"**Dataset check: {status}**  \n"
        f"`{readable}` clips · `{duration_minutes:.1f}` minutes"
        f"{unreadable_note}  \n\n{advice}"
    )


def _scan_audio_duration(directory: Path) -> tuple[float, int, int]:
    """Return total duration, readable count, and unreadable count."""
    import soundfile as sf  # noqa: PLC0415

    duration_seconds = 0.0
    readable = 0
    unreadable = 0
    for audio_file in directory.glob("*.wav"):
        try:
            duration_seconds += sf.info(str(audio_file)).duration
            readable += 1
        except (OSError, RuntimeError):
            unreadable += 1
    return duration_seconds, readable, unreadable
