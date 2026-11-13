"""Paste-a-screenshot box (display input only): click it, press Ctrl+V, and the clipboard image comes back as bytes.

A minimal bidirectional Streamlit component in plain HTML/JS (components/paste_image/index.html), so no package or
build step is needed. Streamlit's file uploader accepts drag-and-drop and file pickers but not the clipboard.
"""
from __future__ import annotations

import base64
from pathlib import Path

import streamlit.components.v1 as components

_DIR = Path(__file__).resolve().parent / "components" / "paste_image"
_component = components.declare_component("nexus_paste_image", path=str(_DIR))
ALLOWED = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif", "image/bmp"}
MAX_BYTES = 8 * 1024 * 1024            # the browser side refuses larger images too (index.html); this is the server check


def decode(value) -> tuple[bytes, str] | None:
    """The component's value -> (image bytes, mime type), or None. Rejects anything that is not an image data URL, and
    anything over MAX_BYTES before decoding it (A-102: the browser check alone can be bypassed by a crafted message)."""
    if not value or not isinstance(value, dict):
        return None
    raw_data = value.get("data", "")
    if not isinstance(raw_data, str) or len(raw_data) > MAX_BYTES * 4 // 3 + 128:
        return None
    data, mime = raw_data, str(value.get("mime", "")).lower()
    if mime not in ALLOWED or not data.startswith(f"data:{mime};base64,"):
        return None
    try:
        raw = base64.b64decode(data.split(",", 1)[1], validate=True)
    except ValueError:
        return None
    return (raw, mime) if raw and len(raw) <= MAX_BYTES else None


def paste_image(label: str, key: str) -> tuple[bytes, str] | None:
    return decode(_component(label=label, key=key, default=None))
