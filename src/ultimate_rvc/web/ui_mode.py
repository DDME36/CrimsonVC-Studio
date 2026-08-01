"""Select the amount of functionality rendered by the Web UI."""

from __future__ import annotations

import os
from enum import StrEnum


class UIMode(StrEnum):
    """Supported CrimsonVC Web UI modes."""

    STUDIO = "studio"
    COVER = "cover"


def resolve_ui_mode(mode: UIMode | str | None = None) -> UIMode:
    """Resolve an explicit mode or the ``URVC_UI_MODE`` environment variable."""
    raw_mode = mode if mode is not None else os.environ.get("URVC_UI_MODE", "studio")
    normalized = str(raw_mode).strip().lower()
    try:
        return UIMode(normalized)
    except ValueError as error:
        supported = ", ".join(item.value for item in UIMode)
        msg = (
            f"Unsupported CrimsonVC UI mode: {raw_mode!r}. Choose one of: {supported}."
        )
        raise ValueError(msg) from error


def is_cover_mode(mode: UIMode | str | None = None) -> bool:
    """Return whether the compact AI Cover interface is selected."""
    return resolve_ui_mode(mode) is UIMode.COVER
