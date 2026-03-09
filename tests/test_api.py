from __future__ import annotations

import io

from backend.app import create_app


class LoadedDummyService:
    model_loaded = True
    model_error = None

    def predict(self, _: bytes) -> dict:
        return {
            "predicted_emotion": "Calm",
            "predicted_class": "CLASS_0",
            "confidence": 0.93,
            "probabilities": {
                "Calm": 0.93,
                "Stressed": 0.03,
                "Angry": 0.02,
                "Focused": 0.02,
            },
            "happy_status": "pending_not_trained",
        }


class UnavailableDummyService:
    model_loaded = False
    model_error = "weights not loaded"

    def predict(self, _: bytes) -> dict:
        raise AssertionError("predict() should not be called when model_loaded=False")


def test_health_endpoint_with_loaded_model() -> None:
    app = create_app(config_overrides={"TESTING": True}, inference_service=LoadedDummyService())
    client = app.test_client()

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "model_loaded": True}


def test_predict_success(sample_image_bytes: bytes) -> None:
    app = create_app(config_overrides={"TESTING": True}, inference_service=LoadedDummyService())
    client = app.test_client()

    response = client.post(
        "/api/predict",
        data={"file": (io.BytesIO(sample_image_bytes), "sample.jpg")},
        content_type="multipart/form-data",
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["predicted_emotion"] == "Calm"
    assert payload["predicted_class"] == "CLASS_0"
    assert payload["happy_status"] == "pending_not_trained"
    assert set(payload["probabilities"].keys()) == {"Calm", "Stressed", "Angry", "Focused"}


def test_predict_missing_file() -> None:
    app = create_app(config_overrides={"TESTING": True}, inference_service=LoadedDummyService())
    client = app.test_client()

    response = client.post("/api/predict", data={}, content_type="multipart/form-data")

    payload = response.get_json()
    assert response.status_code == 400
    assert payload["error_code"] == "NO_FILE"


def test_predict_invalid_extension(sample_image_bytes: bytes) -> None:
    app = create_app(config_overrides={"TESTING": True}, inference_service=LoadedDummyService())
    client = app.test_client()

    response = client.post(
        "/api/predict",
        data={"file": (io.BytesIO(sample_image_bytes), "sample.txt")},
        content_type="multipart/form-data",
    )

    payload = response.get_json()
    assert response.status_code == 400
    assert payload["error_code"] == "INVALID_FILE_TYPE"


def test_predict_model_unavailable_from_service(sample_image_bytes: bytes) -> None:
    app = create_app(config_overrides={"TESTING": True}, inference_service=UnavailableDummyService())
    client = app.test_client()

    response = client.post(
        "/api/predict",
        data={"file": (io.BytesIO(sample_image_bytes), "sample.jpg")},
        content_type="multipart/form-data",
    )

    payload = response.get_json()
    assert response.status_code == 503
    assert payload["error_code"] == "MODEL_UNAVAILABLE"


def test_predict_model_not_found_from_config(sample_image_bytes: bytes, tmp_path) -> None:
    app = create_app(
        config_overrides={
            "TESTING": True,
            "MODEL_ASSET_DIR": str(tmp_path),
        }
    )
    client = app.test_client()

    response = client.post(
        "/api/predict",
        data={"file": (io.BytesIO(sample_image_bytes), "sample.jpg")},
        content_type="multipart/form-data",
    )

    payload = response.get_json()
    assert response.status_code == 503
    assert payload["error_code"] == "MODEL_UNAVAILABLE"


def test_predict_payload_too_large() -> None:
    app = create_app(
        config_overrides={
            "TESTING": True,
            "MAX_UPLOAD_MB": 1,
            "MAX_CONTENT_LENGTH": 128,
        },
        inference_service=LoadedDummyService(),
    )
    client = app.test_client()

    large_bytes = b"x" * 5000
    response = client.post(
        "/api/predict",
        data={"file": (io.BytesIO(large_bytes), "sample.jpg")},
        content_type="multipart/form-data",
    )

    payload = response.get_json()
    assert response.status_code == 413
    assert payload["error_code"] == "PAYLOAD_TOO_LARGE"
