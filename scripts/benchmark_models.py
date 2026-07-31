"""
Compatibility and speed benchmarks for CrimsonVC Studio model choices.

Run this inside the installed project environment. Quality comparisons require
representative recordings; the synthetic F0 test is only a repeatable smoke
test and must not be treated as a listening-test result.
"""

from __future__ import annotations

from typing import Any

import argparse
import gc
import json
import math
import re
import time
from pathlib import Path

import numpy as np

import torch

from ultimate_rvc.common import SEPARATOR_MODELS_DIR, TEMP_DIR
from ultimate_rvc.typing_extra import EmbedderModel, F0Method, SeparationModel


def _device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _nested_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _nested_strings(key)
            yield from _nested_strings(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _nested_strings(item)


def benchmark_inventory() -> list[dict[str, Any]]:
    """Verify that every UI separator filename exists in the installed registry."""
    from audio_separator.separator import Separator

    SEPARATOR_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    separator = Separator(
        model_file_dir=str(SEPARATOR_MODELS_DIR),
        info_only=True,
    )
    registry = separator.list_supported_model_files()
    known_values = set(_nested_strings(registry))
    rows = []
    for model in SeparationModel:
        rows.append(
            {
                "suite": "inventory",
                "model": model.name,
                "filename": model.value,
                "supported": model.value in known_values,
            },
        )
    return rows


def _synthetic_pitch_sweep(
    sample_rate: int = 16000,
    duration: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260731)
    time_axis = np.arange(round(sample_rate * duration)) / sample_rate
    target_f0 = 110.0 * 2.0 ** (2.0 * time_axis / duration)
    phase = np.cumsum(2.0 * np.pi * target_f0 / sample_rate)
    audio = 0.30 * np.sin(phase) + 0.06 * np.sin(2.0 * phase)
    audio += rng.normal(0.0, 0.002, size=audio.shape)
    return audio.astype(np.float32), target_f0


def benchmark_f0(device: str) -> list[dict[str, Any]]:
    """Run every bundled F0 backend on a deterministic monophonic sweep."""
    from ultimate_rvc.rvc.lib.predictors.f0 import CREPE, FCPE, RMVPE

    audio, target = _synthetic_pitch_sweep()
    duration = len(audio) / 16000
    methods = {
        F0Method.RMVPE: lambda: RMVPE(device=device),
        F0Method.CREPE: lambda: CREPE(device=device),
        F0Method.CREPE_TINY: lambda: CREPE(device=device),
        F0Method.FCPE: lambda: FCPE(device=device),
    }
    rows = []

    for method, factory in methods.items():
        _release_cuda()
        started = time.perf_counter()
        model = factory()
        if method == F0Method.CREPE:
            predicted = model.get_f0(audio, model="full")
        elif method == F0Method.CREPE_TINY:
            predicted = model.get_f0(audio, model="tiny")
        else:
            predicted = model.get_f0(audio)
        _sync(device)
        elapsed = time.perf_counter() - started

        predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
        frame_positions = np.linspace(0, len(target) - 1, len(predicted))
        expected = np.interp(frame_positions, np.arange(len(target)), target)
        voiced = np.isfinite(predicted) & (predicted > 0)
        cents = np.full_like(predicted, np.nan)
        cents[voiced] = 1200.0 * np.abs(
            np.log2(predicted[voiced] / expected[voiced]),
        )
        rows.append(
            {
                "suite": "f0",
                "model": method.value,
                "device": device,
                "seconds": round(elapsed, 3),
                "real_time_factor": round(elapsed / duration, 4),
                "voiced_percent": round(100.0 * float(voiced.mean()), 2),
                "median_cents_error": (
                    round(float(np.nanmedian(cents)), 2) if np.any(voiced) else None
                ),
                "frames": len(predicted),
            },
        )
        del model
    _release_cuda()
    return rows


def benchmark_embedders(
    device: str,
    include_language_models: bool,
) -> list[dict[str, Any]]:
    """Load compatible HuBERT-style embedders and validate their output."""
    from ultimate_rvc.rvc.lib.utils import load_embedding

    audio, _ = _synthetic_pitch_sweep(duration=3.0)
    waveform = torch.from_numpy(audio).unsqueeze(0).to(device)
    models = [
        EmbedderModel.CONTENTVEC,
        EmbedderModel.SPIN,
        EmbedderModel.SPIN_V2,
    ]
    if include_language_models:
        models.extend(
            [
                EmbedderModel.CHINESE_HUBERT_BASE,
                EmbedderModel.JAPANESE_HUBERT_BASE,
                EmbedderModel.KOREAN_HUBERT_BASE,
            ],
        )

    rows = []
    for model_name in models:
        _release_cuda()
        started = time.perf_counter()
        model = load_embedding(model_name.value).eval().to(device)
        with torch.inference_mode():
            features = model(waveform)["last_hidden_state"]
        _sync(device)
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "suite": "embedder",
                "model": model_name.value,
                "device": device,
                "seconds": round(elapsed, 3),
                "shape": list(features.shape),
                "finite": bool(torch.isfinite(features).all().item()),
            },
        )
        del features, model
    _release_cuda()
    return rows


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    length = min(len(estimate), len(reference))
    estimate = estimate[:length].astype(np.float64)
    reference = reference[:length].astype(np.float64)
    estimate -= estimate.mean()
    reference -= reference.mean()
    scale = np.dot(estimate, reference) / (
        np.dot(reference, reference) + np.finfo(np.float64).eps
    )
    target = scale * reference
    noise = estimate - target
    return 10.0 * math.log10(
        (np.dot(target, target) + np.finfo(np.float64).eps)
        / (np.dot(noise, noise) + np.finfo(np.float64).eps),
    )


def benchmark_separation(
    audio_path: Path,
    reference_vocals: Path | None,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Run every bundled separator; compute SI-SDR when a reference is supplied."""
    import librosa
    import static_ffmpeg
    from audio_separator.separator import Separator

    static_ffmpeg.add_paths(weak=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_duration = librosa.get_duration(path=audio_path)
    rows = []

    for model_name in SeparationModel:
        slug = _safe_slug(model_name.name)
        separator = Separator(
            model_file_dir=str(SEPARATOR_MODELS_DIR),
            output_dir=str(output_dir),
            output_format="WAV",
            use_autocast=False,
        )
        started = time.perf_counter()
        separator.load_model(model_name.value)
        primary_stem = separator.model_instance.primary_stem_name
        secondary_stem = separator.model_instance.secondary_stem_name
        output_files = separator.separate(
            str(audio_path),
            custom_output_names={
                primary_stem: f"{slug}-primary",
                secondary_stem: f"{slug}-secondary",
            },
        )
        _sync("cuda:0" if torch.cuda.is_available() else "cpu")
        elapsed = time.perf_counter() - started
        primary_path = next(
            Path(path) for path in output_files if Path(path).stem == f"{slug}-primary"
        )

        score = None
        if reference_vocals is not None and primary_stem.lower() == "vocals":
            estimate, sample_rate = librosa.load(primary_path, sr=None, mono=True)
            reference, _ = librosa.load(
                reference_vocals,
                sr=sample_rate,
                mono=True,
            )
            score = round(_si_sdr(estimate, reference), 3)

        rows.append(
            {
                "suite": "separation",
                "model": model_name.value,
                "primary_stem": primary_stem,
                "seconds": round(elapsed, 3),
                "real_time_factor": round(elapsed / input_duration, 4),
                "si_sdr_db": score,
                "outputs": [str(path) for path in output_files],
            },
        )
        del separator
        _release_cuda()
    return rows


def _print_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=["inventory", "f0", "embedders", "separation", "all"],
        default="inventory",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--reference-vocals", type=Path)
    parser.add_argument(
        "--include-language-embedders",
        action="store_true",
        help="Also download/test Chinese, Japanese and Korean HuBERT models.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TEMP_DIR / "model-benchmark",
    )
    parser.add_argument("--json", type=Path, dest="json_path")
    args = parser.parse_args()

    device = _device(args.device)
    rows: list[dict[str, Any]] = []
    if args.suite in {"inventory", "all"}:
        rows.extend(benchmark_inventory())
    if args.suite in {"f0", "all"}:
        rows.extend(benchmark_f0(device))
    if args.suite in {"embedders", "all"}:
        rows.extend(
            benchmark_embedders(device, args.include_language_embedders),
        )
    if args.suite in {"separation", "all"}:
        if args.audio is None:
            if args.suite == "separation":
                parser.error("--audio is required for the separation suite")
            print("Skipping separation: pass --audio to include it.")
        else:
            rows.extend(
                benchmark_separation(
                    args.audio,
                    args.reference_vocals,
                    args.output_dir,
                ),
            )

    _print_rows(rows)
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    failed = [
        row
        for row in rows
        if row.get("supported") is False or row.get("finite") is False
    ]
    if failed:
        raise SystemExit(f"{len(failed)} compatibility check(s) failed")


if __name__ == "__main__":
    main()
