from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_image_bytes() -> bytes:
    image = np.full((300, 600), 255, dtype=np.uint8)
    cv2.putText(
        image,
        "final year",
        (30, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (20,),
        5,
        cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()
