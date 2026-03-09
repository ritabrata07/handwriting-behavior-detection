from __future__ import annotations

import numpy as np

from backend.inference import InferenceService


def test_preprocess_pipeline_shape_and_range(sample_image_bytes: bytes) -> None:
    tensor = InferenceService.preprocess_image_bytes(sample_image_bytes)

    assert tensor.shape == (1, 256, 256, 1)
    assert tensor.dtype == np.float32
    assert 0.0 <= float(tensor.min()) <= 1.0
    assert 0.0 <= float(tensor.max()) <= 1.0


def test_preprocess_is_deterministic(sample_image_bytes: bytes) -> None:
    first = InferenceService.preprocess_image_bytes(sample_image_bytes)
    second = InferenceService.preprocess_image_bytes(sample_image_bytes)

    np.testing.assert_allclose(first, second)


def test_build_prediction_payload_maps_classes() -> None:
    service = InferenceService(model_asset_dir="/tmp/unused")
    probabilities = np.array([0.15, 0.10, 0.60, 0.15], dtype=np.float32)
    classes = ["CLASS_0", "CLASS_1", "CLASS_2", "CLASS_3"]

    payload = service.build_prediction_payload(probabilities, classes)

    assert payload["predicted_class"] == "CLASS_2"
    assert payload["predicted_emotion"] == "Angry"
    assert payload["happy_status"] == "pending_not_trained"
    assert set(payload["probabilities"].keys()) == {"Calm", "Stressed", "Angry", "Focused"}
    assert abs(payload["probabilities"]["Angry"] - 0.60) < 1e-6
