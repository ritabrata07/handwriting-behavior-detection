"""Application configuration utilities."""

from __future__ import annotations

import os
from typing import Any

DEFAULT_MODEL_ASSET_DIR = "/Users/ritabratamaity/Handwriting Project/Project/model_files"
DEFAULT_ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}


def build_runtime_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a runtime config dictionary from environment variables and overrides."""
    config: dict[str, Any] = {
        "MODEL_ASSET_DIR": os.getenv("MODEL_ASSET_DIR", DEFAULT_MODEL_ASSET_DIR),
        "FLASK_ENV": os.getenv("FLASK_ENV", "production"),
        "PORT": int(os.getenv("PORT", "5000")),
        "MAX_UPLOAD_MB": int(os.getenv("MAX_UPLOAD_MB", "8")),
        "ALLOWED_EXTENSIONS": set(DEFAULT_ALLOWED_EXTENSIONS),
        "TESTING": False,
    }

    if overrides:
        config.update(overrides)

    if "MAX_CONTENT_LENGTH" in config:
        config["MAX_CONTENT_LENGTH"] = int(config["MAX_CONTENT_LENGTH"])
    else:
        config["MAX_CONTENT_LENGTH"] = int(config["MAX_UPLOAD_MB"]) * 1024 * 1024
    return config
