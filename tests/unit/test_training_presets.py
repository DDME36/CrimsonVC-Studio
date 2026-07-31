"""Tests for hardware-aware guided training presets."""

from ultimate_rvc.typing_extra import PrecisionType, TrainingSampleRate, Vocoder
from ultimate_rvc.training_presets import TrainingGoal, get_training_preset


def test_t4_speech_preset_uses_fp16_and_safe_batch() -> None:
    """A 16 GB GPU should use FP16 unless BF16 support is reported."""
    preset = get_training_preset(
        TrainingGoal.SPEECH,
        cuda_available=True,
        bf16_supported=False,
        vram_gb=15.0,
    )

    assert preset.precision is PrecisionType.FP16
    assert preset.batch_size == 8
    assert preset.vocoder is Vocoder.HIFI_GAN


def test_low_memory_gpu_reduces_batch_and_enables_checkpointing() -> None:
    """Small GPUs should receive conservative memory defaults."""
    preset = get_training_preset(
        TrainingGoal.SINGING,
        cuda_available=True,
        bf16_supported=False,
        vram_gb=5.5,
    )

    assert preset.batch_size == 2
    assert preset.reduce_memory_usage is True
    assert preset.sample_rate is TrainingSampleRate.HZ_48K


def test_bf16_gpu_uses_bf16() -> None:
    """Modern GPUs may use BF16 when PyTorch confirms support."""
    preset = get_training_preset(
        TrainingGoal.FAST_DRAFT,
        cuda_available=True,
        bf16_supported=True,
        vram_gb=22.0,
    )

    assert preset.precision is PrecisionType.BF16
    assert preset.num_epochs == 120


def test_cpu_preset_uses_fp32_and_small_batch() -> None:
    """CPU fallback must avoid mixed precision assumptions."""
    preset = get_training_preset(
        TrainingGoal.SPEECH,
        cuda_available=False,
        bf16_supported=False,
        vram_gb=None,
    )

    assert preset.precision is PrecisionType.FP32
    assert preset.batch_size == 2
