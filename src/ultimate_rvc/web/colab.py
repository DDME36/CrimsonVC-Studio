"""
Colab-specific launcher for the CrimsonVC Studio Gradio application.

The regular launcher remains suitable for local use. This module adds
optional password authentication for a temporary public Colab share
link without putting credentials in process arguments.
"""

from __future__ import annotations

import os
import platform
import sys
import traceback
from pathlib import Path


def _configure_headless_matplotlib() -> None:
    """Use a backend available inside the isolated Colab environment.

    Colab exports its notebook-only inline backend to child processes. The
    project virtual environment intentionally does not depend on IPython's
    ``matplotlib-inline`` package, so a Gradio server must use ``Agg`` instead.
    """
    os.environ["MPLBACKEND"] = "Agg"


def _get_auth_credentials() -> list[tuple[str, str]] | None:
    """
    Get optional Gradio credentials from environment variables.

    Returns
    -------
    list[tuple[str, str]] or None
        A list of username/password pairs when authentication is configured,
        otherwise ``None``.

    Raises
    ------
    RuntimeError
        If only one required authentication variable is set.

    """
    username = os.environ.get("URVC_AUTH_USERNAME")
    password = os.environ.get("URVC_AUTH_PASSWORD")
    if not username and not password:
        return None
    if not username or not password:
        msg = (
            "URVC_AUTH_USERNAME and URVC_AUTH_PASSWORD must either both be set "
            "or both be omitted."
        )
        raise RuntimeError(msg)
    return [(username, password)]


def main() -> None:
    """Launch the CrimsonVC Studio application for a Colab runtime."""
    _configure_headless_matplotlib()
    print("Matplotlib backend: Agg (headless Gradio server)", flush=True)
    print(f"Python {platform.python_version()} at {sys.executable}", flush=True)
    print(f"Launcher source: {Path(__file__).resolve()}", flush=True)
    print("Importing Gradio...", flush=True)
    import gradio as gr  # noqa: PLC0415

    print(f"Loading CrimsonVC Studio with Gradio {gr.__version__}...", flush=True)
    print("Resolving CrimsonVC storage paths...", flush=True)
    from ultimate_rvc.common import (  # noqa: PLC0415
        AUDIO_DIR,
        MODELS_DIR,
        TEMP_DIR,
    )

    print("Rendering the Web UI...", flush=True)
    from ultimate_rvc.web.main import app  # noqa: PLC0415

    print(f"Web UI rendered successfully ({len(app.blocks)} components).", flush=True)
    os.environ["GRADIO_TEMP_DIR"] = str(TEMP_DIR)
    gr.set_static_paths([MODELS_DIR, AUDIO_DIR, TEMP_DIR])
    app.queue()
    print("Creating the temporary gradio.live tunnel...", flush=True)
    app.launch(
        share=True,
        inline=False,
        server_name="0.0.0.0",  # noqa: S104
        auth=_get_auth_credentials(),
        show_error=True,
        debug=True,
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as err:  # noqa: BLE001
        print(f"CrimsonVC Studio failed to start: {err}", file=sys.stdout, flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        sys.exit(1)
