"""Initialize the CrimsonVC Studio core runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

import lazy_loader as lazy

import os

from ultimate_rvc.core.common import FLAG_FILE
from ultimate_rvc.rvc.lib.tools.prerequisites_download import (
    prequisites_download_pipeline,
)

if TYPE_CHECKING:
    import static_sox

else:
    static_sox = lazy.load("static_sox")


def initialize() -> None:
    """
    Download required assets and initialize bundled audio tools.

    ContentVec is the compatibility-first embedder and is downloaded during
    setup. Optional embedders are downloaded lazily when selected. Set
    ``URVC_DOWNLOAD_ALL_EMBEDDERS=1`` to prefetch every bundled embedder.
    """
    download_all_embedders = os.environ.get("URVC_DOWNLOAD_ALL_EMBEDDERS") == "1"
    prequisites_download_pipeline(
        exe=False,
        all_embedders=download_all_embedders,
    )

    if not FLAG_FILE.is_file():
        # add_paths also downloads the bundled SoX binaries when required.
        static_sox.add_paths(weak=True)
        FLAG_FILE.touch()


if __name__ == "__main__":
    initialize()
