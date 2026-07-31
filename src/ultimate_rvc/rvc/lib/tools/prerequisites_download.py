"""Download the model assets required by CrimsonVC Studio."""

from __future__ import annotations

from typing import TYPE_CHECKING

import lazy_loader as lazy

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ultimate_rvc.common import (
    EMBEDDER_MODELS_DIR,
    PRETRAINED_MODELS_DIR,
    RVC_MODELS_DIR,
)

if TYPE_CHECKING:
    import requests

    import tqdm

else:
    requests = lazy.load("requests")
    tqdm = lazy.load("tqdm")


URL_BASE = "https://huggingface.co/JackismyShephard/ultimate-rvc/resolve/main/Resources"

pretraineds_hifigan_list = [
    (
        "pretrained_v2/",
        [
            "f0D32k.pth",
            "f0D40k.pth",
            "f0D48k.pth",
            "f0G32k.pth",
            "f0G40k.pth",
            "f0G48k.pth",
        ],
    ),
]
pretraineds_refinegan_list = [("refinegan/", ["f0D32k.pth", "f0G32k.pth"])]
models_list = [("predictors/", ["rmvpe.pt", "fcpe.pt"])]

# ContentVec is the compatibility-first default. The other embedders are
# available through load_embedding(), which downloads them when first selected.
default_embedders_list = [
    ("embedders/contentvec/", ["pytorch_model.bin", "config.json"]),
]
optional_embedders_list = [
    ("embedders/chinese_hubert_base/", ["pytorch_model.bin", "config.json"]),
    ("embedders/japanese_hubert_base/", ["pytorch_model.bin", "config.json"]),
    ("embedders/korean_hubert_base/", ["pytorch_model.bin", "config.json"]),
    ("embedders/spin/", ["pytorch_model.bin", "config.json"]),
    ("embedders/spin-v2/", ["pytorch_model.bin", "config.json"]),
]
embedders_list = default_embedders_list + optional_embedders_list

executables_list = [("", ["ffmpeg.exe", "ffprobe.exe"])]

folder_mapping_list = {
    "pretrained_v2/": str(PRETRAINED_MODELS_DIR / "hifi-gan/"),
    "refinegan/": str(PRETRAINED_MODELS_DIR / "refinegan/"),
    "embedders/contentvec/": str(EMBEDDER_MODELS_DIR / "contentvec/"),
    "embedders/chinese_hubert_base/": str(
        EMBEDDER_MODELS_DIR / "chinese_hubert_base/",
    ),
    "embedders/japanese_hubert_base/": str(
        EMBEDDER_MODELS_DIR / "japanese_hubert_base/",
    ),
    "embedders/korean_hubert_base/": str(
        EMBEDDER_MODELS_DIR / "korean_hubert_base/",
    ),
    "embedders/spin/": str(EMBEDDER_MODELS_DIR / "spin/"),
    "embedders/spin-v2/": str(EMBEDDER_MODELS_DIR / "spin-v2/"),
    "predictors/": str(RVC_MODELS_DIR / "predictors/"),
    "formant/": str(RVC_MODELS_DIR / "formant/"),
}


def has_missing_files(file_list) -> bool:
    """Return whether at least one mapped file is absent locally."""
    return any(
        not (Path(folder_mapping_list.get(folder, "")) / file_name).exists()
        for folder, files in file_list
        for file_name in files
    )


def get_file_size_if_missing(file_list) -> int:
    """Return the known remote byte size of files not present locally."""
    total_size = 0
    for remote_folder, files in file_list:
        local_folder = folder_mapping_list.get(remote_folder, "")
        for file_name in files:
            destination_path = Path(local_folder) / file_name
            if destination_path.exists():
                continue
            url = f"{URL_BASE}/{remote_folder}{file_name}"
            try:
                response = requests.head(
                    url,
                    allow_redirects=True,
                    timeout=30,
                )
                response.raise_for_status()
            except requests.RequestException:
                # A missing Content-Length must not prevent the real GET below.
                continue
            total_size += int(response.headers.get("content-length", 0))
    return total_size


def download_file(url: str, destination_path: str, global_bar) -> None:
    """Download one file atomically while updating the shared progress bar."""
    destination = Path(destination_path)
    destination.parent.mkdir(exist_ok=True, parents=True)
    partial = destination.with_name(f"{destination.name}.part")

    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with partial.open("wb") as file:
                for data in response.iter_content(1024 * 1024):
                    if data:
                        file.write(data)
                        global_bar.update(len(data))
        partial.replace(destination)
    finally:
        if partial.exists():
            partial.unlink()


def download_mapping_files(file_mapping_list, global_bar) -> None:
    """Download missing files in parallel."""
    with ThreadPoolExecutor() as executor:
        futures = []
        for remote_folder, file_list in file_mapping_list:
            local_folder = folder_mapping_list.get(remote_folder, "")
            for file_name in file_list:
                destination_path = str(Path(local_folder) / file_name)
                if Path(destination_path).exists():
                    continue
                url = f"{URL_BASE}/{remote_folder}{file_name}"
                futures.append(
                    executor.submit(
                        download_file,
                        url,
                        destination_path,
                        global_bar,
                    ),
                )
        for future in futures:
            future.result()


def split_pretraineds(pretrained_list):
    """Split pretrained filenames into F0 and non-F0 groups."""
    f0_list = []
    non_f0_list = []
    for folder, files in pretrained_list:
        f0_files = [file for file in files if file.startswith("f0")]
        non_f0_files = [file for file in files if not file.startswith("f0")]
        if f0_files:
            f0_list.append((folder, f0_files))
        if non_f0_files:
            non_f0_list.append((folder, non_f0_files))
    return f0_list, non_f0_list


pretraineds_hifigan_list, _ = split_pretraineds(pretraineds_hifigan_list)


def calculate_total_size(
    pretraineds_hifigan,
    models: bool,
    all_embedders: bool = False,
) -> int:
    """Calculate the known size of assets that are still missing."""
    total_size = 0
    if models:
        selected_embedders = embedders_list if all_embedders else default_embedders_list
        total_size += get_file_size_if_missing(models_list)
        total_size += get_file_size_if_missing(selected_embedders)
    if pretraineds_hifigan:
        total_size += get_file_size_if_missing(pretraineds_hifigan)
        total_size += get_file_size_if_missing(pretraineds_refinegan_list)
    return total_size


def prequisites_download_pipeline(
    pretraineds_hifigan: bool = True,
    models: bool = True,
    exe: bool = True,
    all_embedders: bool = False,
) -> None:
    """Download required model files, with optional embedder prefetching."""
    selected_pretraineds = pretraineds_hifigan_list if pretraineds_hifigan else []
    selected_embedders = embedders_list if all_embedders else default_embedders_list
    total_size = calculate_total_size(
        selected_pretraineds,
        models,
        all_embedders,
    )
    if exe and os.name == "nt":
        total_size += get_file_size_if_missing(executables_list)

    download_groups = []
    if models:
        download_groups.extend([models_list, selected_embedders])
    if exe and os.name == "nt":
        download_groups.append(executables_list)
    if pretraineds_hifigan:
        download_groups.extend(
            [pretraineds_hifigan_list, pretraineds_refinegan_list],
        )
    if not any(has_missing_files(group) for group in download_groups):
        return

    with tqdm.tqdm(
        total=total_size or None,
        unit="iB",
        unit_scale=True,
        desc="Downloading required files",
    ) as global_bar:
        if models:
            download_mapping_files(models_list, global_bar)
            download_mapping_files(selected_embedders, global_bar)
        if exe and os.name == "nt":
            download_mapping_files(executables_list, global_bar)
        if pretraineds_hifigan:
            download_mapping_files(pretraineds_hifigan_list, global_bar)
            download_mapping_files(pretraineds_refinegan_list, global_bar)
