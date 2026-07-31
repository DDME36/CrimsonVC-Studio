"""Beginner-friendly presets for CrimsonVC voice-model training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ultimate_rvc.typing_extra import (
    EmbedderModel,
    F0Method,
    IndexAlgorithm,
    PrecisionType,
    PretrainedType,
    TrainingSampleRate,
    Vocoder,
)


class TrainingGoal(StrEnum):
    """Training goals exposed by the guided training UI."""

    SPEECH = "speech"
    SINGING = "singing"
    FAST_DRAFT = "fast-draft"


@dataclass(frozen=True)
class TrainingPreset:
    """A complete set of safe starting values for guided training."""

    label: str
    description: str
    sample_rate: TrainingSampleRate
    f0_method: F0Method
    embedder_model: EmbedderModel
    num_epochs: int
    batch_size: int
    vocoder: Vocoder
    precision: PrecisionType
    reduce_memory_usage: bool
    detect_overtraining: bool = True
    overtraining_threshold: int = 50
    index_algorithm: IndexAlgorithm = IndexAlgorithm.AUTO
    pretrained_type: PretrainedType = PretrainedType.DEFAULT
    save_interval: int = 10
    preload_dataset: bool = False


GOAL_LABELS: dict[TrainingGoal, str] = {
    TrainingGoal.SPEECH: "Speech - balanced",
    TrainingGoal.SINGING: "Singing - high fidelity",
    TrainingGoal.FAST_DRAFT: "Fast draft - test the dataset",
}


def get_training_preset(
    goal: TrainingGoal | str,
    *,
    cuda_available: bool,
    bf16_supported: bool,
    vram_gb: float | None,
) -> TrainingPreset:
    """Build a preset and adapt memory settings to the current hardware."""
    selected_goal = TrainingGoal(goal)
    precision = _recommended_precision(cuda_available, bf16_supported)

    match selected_goal:
        case TrainingGoal.SPEECH:
            label = GOAL_LABELS[selected_goal]
            description = (
                "Compatibility-first settings for spoken voice and general use."
            )
            sample_rate = TrainingSampleRate.HZ_40K
            f0_method = F0Method.RMVPE
            embedder_model = EmbedderModel.CONTENTVEC
            num_epochs = 300
            base_batch_size = 8
        case TrainingGoal.SINGING:
            label = GOAL_LABELS[selected_goal]
            description = (
                "Higher sample rate for singing while keeping a stable pretrained "
                "HiFi-GAN path."
            )
            sample_rate = TrainingSampleRate.HZ_48K
            f0_method = F0Method.RMVPE
            embedder_model = EmbedderModel.CONTENTVEC
            num_epochs = 500
            base_batch_size = 8
        case TrainingGoal.FAST_DRAFT:
            label = GOAL_LABELS[selected_goal]
            description = (
                "A short validation run. Use it to catch dataset problems before a "
                "long training session."
            )
            sample_rate = TrainingSampleRate.HZ_40K
            f0_method = F0Method.RMVPE
            embedder_model = EmbedderModel.CONTENTVEC
            num_epochs = 120
            base_batch_size = 8

    batch_size = _recommended_batch_size(base_batch_size, cuda_available, vram_gb)
    reduce_memory_usage = bool(
        cuda_available and vram_gb is not None and vram_gb < 8
    )
    return TrainingPreset(
        label=label,
        description=description,
        sample_rate=sample_rate,
        f0_method=f0_method,
        embedder_model=embedder_model,
        num_epochs=num_epochs,
        batch_size=batch_size,
        vocoder=Vocoder.HIFI_GAN,
        precision=precision,
        reduce_memory_usage=reduce_memory_usage,
    )


def format_preset_summary(
    preset: TrainingPreset,
    *,
    gpu_name: str,
    vram_gb: float | None,
) -> str:
    """Format a concise confirmation shown after applying a preset."""
    memory = f"{vram_gb:.1f} GB VRAM" if vram_gb is not None else "VRAM unavailable"
    return (
        f"**Applied: {preset.label}**  \n"
        f"{preset.description}  \n\n"
        f"`{preset.sample_rate.value} Hz` · `{preset.num_epochs} epochs` · "
        f"`batch {preset.batch_size}` · `{preset.precision.value}` · "
        f"`{preset.vocoder.value}`  \n"
        f"Runtime: **{gpu_name}** ({memory}). Review the advanced options if you "
        "are resuming a custom experiment."
    )


def _recommended_precision(
    cuda_available: bool,
    bf16_supported: bool,
) -> PrecisionType:
    if not cuda_available:
        return PrecisionType.FP32
    return PrecisionType.BF16 if bf16_supported else PrecisionType.FP16


def _recommended_batch_size(
    base_batch_size: int,
    cuda_available: bool,
    vram_gb: float | None,
) -> int:
    if not cuda_available:
        return 2
    if vram_gb is None:
        return min(base_batch_size, 4)
    if vram_gb < 6:
        return 2
    if vram_gb < 10:
        return min(base_batch_size, 4)
    if vram_gb < 14:
        return min(base_batch_size, 6)
    return base_batch_size
