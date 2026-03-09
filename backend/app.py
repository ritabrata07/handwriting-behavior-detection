"""Flask application for handwriting emotion recognition."""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Any

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from .config import build_runtime_config
from .inference import InferenceService, ModelUnavailableError, PredictionInputError

LOGGER = logging.getLogger(__name__)


def create_app(
    config_overrides: dict[str, Any] | None = None,
    inference_service: InferenceService | None = None,
) -> Flask:
    """Application factory."""
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    runtime_config = build_runtime_config(config_overrides)
    app.config.update(runtime_config)

    service = inference_service or InferenceService(app.config["MODEL_ASSET_DIR"])
    if inference_service is None:
        service.initialize()

    if not service.model_loaded:
        LOGGER.warning("Model failed to load at startup: %s", service.model_error)

    app.inference_service = service  # type: ignore[attr-defined]

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/api/health")
    def health() -> Any:
        return jsonify({"status": "ok", "model_loaded": app.inference_service.model_loaded})

    @app.post("/api/predict")
    def predict() -> Any:
        if not app.inference_service.model_loaded:
            details = app.inference_service.model_error or "Model not loaded."
            return _error_response(
                "MODEL_UNAVAILABLE",
                f"Prediction model is unavailable: {details}",
                HTTPStatus.SERVICE_UNAVAILABLE,
            )

        if "file" not in request.files:
            return _error_response(
                "NO_FILE",
                "No file uploaded. Send multipart form-data with the 'file' field.",
                HTTPStatus.BAD_REQUEST,
            )

        upload = request.files["file"]
        if not upload.filename:
            return _error_response(
                "EMPTY_FILENAME",
                "Uploaded file has no filename.",
                HTTPStatus.BAD_REQUEST,
            )

        filename = secure_filename(upload.filename)
        if not _allowed_file(filename, app.config["ALLOWED_EXTENSIONS"]):
            return _error_response(
                "INVALID_FILE_TYPE",
                "Unsupported file type. Use png, jpg, jpeg, bmp, tif, or tiff.",
                HTTPStatus.BAD_REQUEST,
            )

        try:
            image_bytes = upload.read()
            if not image_bytes:
                return _error_response(
                    "EMPTY_FILE",
                    "Uploaded file is empty.",
                    HTTPStatus.BAD_REQUEST,
                )

            result = app.inference_service.predict(image_bytes)
            return jsonify(result)
        except ModelUnavailableError as exc:
            return _error_response(
                "MODEL_UNAVAILABLE",
                f"Prediction model is unavailable: {exc}",
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
        except PredictionInputError as exc:
            return _error_response(
                "INVALID_IMAGE",
                str(exc),
                HTTPStatus.BAD_REQUEST,
            )
        except Exception as exc:  # noqa: BLE001 - API boundary should not leak stacktraces.
            LOGGER.exception("Unexpected prediction failure")
            return _error_response(
                "PREDICTION_FAILED",
                f"Unexpected prediction failure: {exc}",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    @app.errorhandler(RequestEntityTooLarge)
    def handle_too_large(_: RequestEntityTooLarge) -> Any:
        return _error_response(
            "PAYLOAD_TOO_LARGE",
            f"File exceeds maximum size of {app.config['MAX_UPLOAD_MB']} MB.",
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        )

    return app


def _allowed_file(filename: str, allowed_extensions: set[str]) -> bool:
    if "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in allowed_extensions


def _error_response(error_code: str, message: str, status: HTTPStatus) -> Any:
    response = jsonify({"error_code": error_code, "message": message})
    response.status_code = status
    return response


if __name__ == "__main__":
    application = create_app()
    is_dev = application.config.get("FLASK_ENV") == "development"
    application.run(host="0.0.0.0", port=application.config["PORT"], debug=is_dev)
