<div align="center">

# CrimsonVC Studio

**Colab-first voice conversion and model training studio**

Create song covers, convert speech, and train RVC voice models from a focused
Gradio interface.

[![Release](https://img.shields.io/badge/status-alpha-e11d48)](#project-status)
[![Python](https://img.shields.io/badge/python-3.12-3776ab)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/gradio-5.50.0-f97316)](https://www.gradio.app/)
[![License](https://img.shields.io/badge/license-MIT-22c55e)](LICENSE)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DDME36/CrimsonVC-Studio/blob/main/CrimsonVC_Colab.ipynb)

</div>

> [!IMPORTANT]
> The first public release should be treated as an alpha until a fresh Colab T4
> run completes end to end.

## What it does

CrimsonVC Studio is a product-focused distribution of
[Ultimate RVC](https://github.com/JackismyShephard/ultimate-rvc). It keeps the
compatible `ultimate_rvc` Python package internally while presenting a clearer
workflow for creation, model training, and asset management.

- Generate a song cover with source separation, voice conversion, and mixing.
- Convert spoken audio or synthesize speech before applying an RVC voice.
- Train, resume, export, upload, and manage custom voice models.
- Start training from hardware-aware Speech, Singing, or Fast Draft presets.
- Inspect preprocessed dataset duration and clip count before a long run.
- Persist models, datasets, output audio, and configuration in Google Drive.
- Protect temporary Colab share links with username/password authentication.

The new interface uses progressive disclosure: recommended settings are
available first, while the original detailed controls remain under
**Advanced options**.

## Project status

**Current release target: `0.1.0-alpha`**

The CPU-safe local audit passes 286 tests, including notebook syntax, guided
training presets, download behavior, and model-stack regression checks. A real
GPU smoke test is still required after publishing:

1. Start a clean Colab runtime with a T4 or better NVIDIA GPU.
2. Run every notebook cell without reusing an old workspace.
3. Launch the authenticated Gradio URL.
4. Download or upload a test voice model.
5. Generate a short output.
6. Run a short Fast Draft training job and confirm that the model survives a
   runtime restart when Google Drive storage is enabled.

Do not call the release stable until that checklist passes.

## Google Colab quick start

The primary notebook is
[`CrimsonVC_Colab.ipynb`](CrimsonVC_Colab.ipynb).

### Run it

1. Open the Colab badge above.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Keep the default `runtime_only` mode for RVC Cover. Select `google_drive`
   only when models and outputs should persist between sessions.
4. Select **Runtime > Run all**. Drive authorization appears only if you selected
   `google_drive`.
5. Cell 6 asks for a Web UI password. Type at least eight characters into the
   hidden input and press Enter.
6. Wait for `Running on public URL`, open the `gradio.live` link, and sign in
   with username `crimsonvc` and the password from step 5.

The first installation can take several minutes. The final launch cell keeps
running while the UI is online; stop that cell to close the public URL.

#### Use RVC without training

1. Open **Models & Train > Upload > Voice models**.
2. Upload the model `.pth` and its optional `.index`, enter a unique model name,
   and click **Upload**.
3. For an audio file or song, open **Create > Song cover > One-click**, choose
   the local-file source, upload audio, select the voice model, and click
   **Generate**.
4. For text-to-speech followed by RVC, use **Create > Speech > One-click**.

The notebook clones the official
[`DDME36/CrimsonVC-Studio`](https://github.com/DDME36/CrimsonVC-Studio)
repository by default. You can still change `repository_url` to test a fork.

### Storage modes

| Mode | Persists after runtime deletion | Training speed | Best for |
|---|---:|---:|---|
| `runtime_only` | No | Usually faster | RVC Cover and disposable sessions |
| `google_drive` | Yes | Can be slower | Persistent models, outputs, and training |

Google Drive is optional. In the default `runtime_only` mode, the complete RVC
Cover pipeline runs on Colab's local disk, but uploaded models and generated audio
disappear when the runtime is deleted. Select `google_drive` to store them under
`MyDrive/CrimsonVC`.

Gradio share URLs are reachable from the internet. Authentication is enabled
by default, but the URL is still intended for temporary personal sessions.

## Guided model training

Open **Models & Train > Train > Guided training**.

1. Choose a goal and click **Apply recommended settings**.
2. Add or select a dataset, enter a model name, and preprocess it.
3. Click **Analyze preprocessed dataset** and listen to sliced previews.
4. Extract pitch and content features.
5. Train the model and download the exported `.pth` and `.index` files.

### Preset starting points

| Goal | Sample rate | Epoch ceiling | Stable vocoder | Intended use |
|---|---:|---:|---|---|
| Speech | 40 kHz | 300 | HiFi-GAN | Spoken voice and general conversion |
| Singing | 48 kHz | 500 | HiFi-GAN | Singing and higher-frequency detail |
| Fast Draft | 40 kHz | 120 | HiFi-GAN | Validate a dataset and pipeline quickly |

Batch size and precision are adapted to the detected hardware. For example, a
typical T4 uses FP16, a GPU confirmed by PyTorch to support BF16 may use BF16,
and CPU fallback uses FP32. These presets are safe starting points, not a
guarantee of model quality.

### Dataset quality matters most

- Use only recordings you own or have permission to process.
- Prefer a clean, isolated voice with minimal reverb and background music.
- Remove clipping, long silence, duplicate takes, and inconsistent volume.
- Cover the pitch, pronunciation, and vocal styles expected at inference time.
- Use Fast Draft before committing to a long training run.
- More audio is not automatically better; clean and representative audio wins.

The dataset report checks readable clip count and total duration. It does not
yet calculate perceptual quality, SNR, clipping rate, or speaker consistency.

## Model stack and benchmark guidance

The compatibility defaults stay conservative, while current alternatives are
available for A/B tests on your own recordings.

| Stage | Included choices | Recommended starting point | What to test next |
|---|---|---|---|
| Vocal separation | BS-RoFormer 1297/1296, Mel-Band RoFormer, MDX23C and UVR utility models | BS-RoFormer Viperx 1297 | Mel-Band on dense mixes; an ensemble only when extra runtime is acceptable |
| Pitch extraction | RMVPE, FCPE, CREPE full/tiny | RMVPE | FCPE for speed; CREPE when comparing difficult high notes |
| Content representation | ContentVec, Spin, Spin-v2, Chinese/Japanese/Korean HuBERT | ContentVec | Spin/Spin-v2 when source-speaker leakage is audible |
| RVC vocoder | HiFi-GAN family, RefineGAN, RingFormer, APEX-GAN | HiFi-GAN pretrained path | Experimental vocoders only with matching pretrained weights |
| Retrieval index | Auto, FAISS, KMeans | Auto | Tune index rate before changing the index algorithm |

The UI stays on Gradio 5.50.0, the final stable v5 release, because Gradio 6 is
a major-version migration and this interface uses a large set of component/event
APIs. Upgrade to Gradio 6 only in a dedicated compatibility pass with a full UI
smoke test.

The separator runtime is pinned to
[`audio-separator` 0.44.5](https://github.com/nomadkaraoke/python-audio-separator).
Its published registry still ranks the included BS-RoFormer Viperx 1297
checkpoint at the top of the listed vocal models, so CrimsonVC keeps it as the
single-model default. The newer package also offers a `vocal_rvc` ensemble
preset. That preset is a promising quality mode, but it runs two models and is
not the default until it is measured against representative speech and singing
sets.

ContentVec remains the safest RVC-compatible default. Spin and Spin-v2 keep the
same HuBERT-style 768-dimensional interface used by this codebase and are useful
experiments for reducing source-speaker information. Newer research such as
[R-Spin](https://aclanthology.org/2024.naacl-long.36/) and
[DC-Spin](https://arxiv.org/abs/2410.24177) is interesting, but those checkpoints
are not automatically drop-in replacements for an RVC model trained with this
feature layout.

RMVPE remains the compatibility-first F0 default. FCPE is the included fast
alternative. [PESTO](https://github.com/SonyCSLParis/pesto) and
[SwiftF0](https://github.com/lars76/swift-f0) are benchmark candidates rather
than bundled defaults because frame alignment, voicing behavior, and conversion
quality must be validated inside the RVC pipeline, not only on a pitch dataset.

### Run the model checks in Colab

The optional notebook cell can run the installed-model checks, or use the CLI:

```console
# Confirm that every separator shown by the UI exists in the installed registry
uv run --no-sync python scripts/benchmark_models.py --suite inventory

# Repeatable compatibility/speed test for RMVPE, FCPE and both CREPE sizes
uv run --no-sync python scripts/benchmark_models.py --suite f0

# Load ContentVec, Spin and Spin-v2 and validate HuBERT output tensors
uv run --no-sync python scripts/benchmark_models.py --suite embedders

# Real separator timing; add a clean vocal reference to calculate SI-SDR
uv run --no-sync python scripts/benchmark_models.py --suite separation \
  --audio /content/test-mixture.wav \
  --reference-vocals /content/test-vocals.wav \
  --json /content/separator-results.json
```

The synthetic F0 sweep is a smoke test, not a listening score. A useful model
comparison needs the same held-out recordings, exact runtime/GPU details,
objective metrics where a reference exists, and blind listening for artifacts,
speaker leakage, consonant clarity, and pitch stability.

RVC remains the primary engine because it is practical for singing conversion,
pitch preservation, and per-voice fine-tuning on Colab-class hardware. Other
systems solve different problems:

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) targets few-shot speech
  cloning and text-to-speech.
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) targets multilingual and
  streaming text-to-speech.
- [Seed-VC](https://github.com/Plachtaa/seed-vc) demonstrates zero-shot speech
  and singing conversion, but its official repository is archived.

These are roadmap candidates, not bundled backends. Test them in separate
notebooks/environments first to avoid dependency conflicts and unexpected VRAM
pressure.

## Local setup

### Requirements

- Git
- Windows or Ubuntu 22.04/24.04
- NVIDIA CUDA GPU for the default accelerator path
- Sufficient free disk space for PyTorch, separator assets, and model data

### Clone

```console
git clone https://github.com/DDME36/CrimsonVC-Studio.git
cd CrimsonVC-Studio
```

### Windows

```powershell
.\urvc.ps1 install
.\urvc.ps1 run
```

### Ubuntu

```console
chmod +x ./urvc
./urvc install
./urvc run
```

Open the local URL printed by Gradio, normally
`http://127.0.0.1:7860`.

The legacy `urvc` launcher and `ultimate_rvc` module names are intentionally
retained to avoid breaking upstream-compatible imports, saved configuration,
and scripts.

## Main environment variables

| Variable | Purpose | Default |
|---|---|---|
| `URVC_MODELS_DIR` | Model storage | `./models` |
| `URVC_AUDIO_DIR` | Dataset and generated audio storage | `./audio` |
| `URVC_CONFIG_DIR` | Saved UI configuration | `./config` |
| `URVC_TEMP_DIR` | Temporary files | `./temp` |
| `URVC_CONFIG` | Configuration name loaded at startup | Default UI values |
| `URVC_ACCELERATOR` | `cuda` or `rocm` dependency group | `cuda` |
| `YT_COOKIEFILE` | Optional cookies used by yt-dlp | Not set |
| `URVC_AUTH_USERNAME` | Colab Gradio username | Not set |
| `URVC_AUTH_PASSWORD` | Colab Gradio password | Not set |
| `URVC_DOWNLOAD_ALL_EMBEDDERS` | Prefetch every optional HuBERT/Spin model (`1`) | `0` |

## Repository layout

```text
CrimsonVC_Colab.ipynb       Primary Google Colab launcher
src/ultimate_rvc/           Compatible application package
  web/                      Gradio UI, guided training, and Colab launcher
  core/                     Application workflows
  rvc/                      RVC inference and training implementation
scripts/benchmark_models.py Optional model compatibility/quality benchmark
tests/                      CPU-safe structural and unit tests
```

## Responsible use

Voice conversion can be used for creative work, accessibility, localization,
and research, but it can also be abused.

- Obtain consent for voices and recordings you use.
- Clearly label synthetic or converted media when confusion is possible.
- Do not impersonate someone for fraud, harassment, political deception, or
  identity theft.
- Follow applicable copyright, publicity-right, privacy, platform, and Colab
  policies.

You are responsible for how you train models and publish generated media.

## Credits and license

CrimsonVC Studio builds on
[Ultimate RVC](https://github.com/JackismyShephard/ultimate-rvc) and the wider
open-source RVC ecosystem. Individual model assets and third-party packages may
have their own terms; review them before redistribution.

The project code is distributed under the [MIT License](LICENSE). Preserve the
upstream copyright notice and attribution when publishing a fork.
