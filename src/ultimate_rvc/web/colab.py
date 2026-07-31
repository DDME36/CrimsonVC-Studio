"""
Colab-specific launcher for the CrimsonVC Studio Gradio application.

The regular launcher remains suitable for local use. This module adds
optional password authentication for a temporary public Colab share
link without putting credentials in process arguments.
"""

from __future__ import annotations

import os

import gradio as gr

from ultimate_rvc.common import AUDIO_DIR, MODELS_DIR, TEMP_DIR
from ultimate_rvc.web.main import app


def _get_auth_credentials() -> tuple[str, str] | None:
    """
    Get optional Gradio credentials from environment variables.

    Returns
    -------
    tuple[str, str] or None
        A username/password pair when authentication is configured,
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
    return username, password


def main() -> None:
    """Launch the CrimsonVC Studio application for a Colab runtime."""
    os.environ["GRADIO_TEMP_DIR"] = str(TEMP_DIR)
    gr.set_static_paths([MODELS_DIR, AUDIO_DIR])
    app.queue()
    app.launch(
        share=True,
        server_name="127.0.0.1",
        auth=_get_auth_credentials(),
    )


if __name__ == "__main__":
    main()
