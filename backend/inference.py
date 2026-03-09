"""Model loading and handwriting emotion inference."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


class ModelUnavailableError(RuntimeError):
    """Raised when the model is unavailable at prediction time."""


class PredictionInputError(ValueError):
    """Raised when an uploaded image cannot be processed."""


@dataclass(frozen=True)
class ModelAssetPaths:
    model_json: str
    model_weights: str
    label_encoder: str


class InferenceService:
    """Inference service that mirrors the training-time preprocessing pipeline."""

    CLASS_TO_EMOTION = {
        "CLASS_0": "Calm",
        "CLASS_1": "Stressed",
        "CLASS_2": "Angry",
        "CLASS_3": "Focused",
    }
    SUPPORTED_EMOTIONS = ["Calm", "Stressed", "Angry", "Focused"]

    def __init__(self, model_asset_dir: str) -> None:
        self.model_asset_dir = model_asset_dir
        self.model: Any | None = None
        self.label_encoder: Any | None = None
        self.model_loaded = False
        self.model_error: str | None = None

    def initialize(self) -> None:
        """Load model architecture, weights, and label encoder."""
        try:
            paths = self._get_asset_paths()
            self._validate_assets(paths)
            self.model = self._load_model(paths)
            self.label_encoder = self._load_label_encoder(paths)
            self.model_loaded = True
            self.model_error = None
        except Exception as exc:  # noqa: BLE001 - keep error details for API diagnostics.
            self.model_loaded = False
            self.model_error = str(exc)

    def _get_asset_paths(self) -> ModelAssetPaths:
        return ModelAssetPaths(
            model_json=os.path.join(self.model_asset_dir, "model.json"),
            model_weights=os.path.join(self.model_asset_dir, "model.weights.h5"),
            label_encoder=os.path.join(self.model_asset_dir, "labels.sav"),
        )

    @staticmethod
    def _validate_assets(paths: ModelAssetPaths) -> None:
        missing = [
            path
            for path in [paths.model_json, paths.model_weights, paths.label_encoder]
            if not os.path.isfile(path)
        ]
        if missing:
            raise FileNotFoundError(
                "Missing model assets: " + ", ".join(missing)
            )

    @staticmethod
    def _load_model(paths: ModelAssetPaths) -> Any:
        from tensorflow.keras.models import model_from_json

        with open(paths.model_json, "r", encoding="utf-8") as model_json_file:
            model_json = model_json_file.read()

        model = model_from_json(model_json)
        model.load_weights(paths.model_weights)
        return model

    @staticmethod
    def _load_label_encoder(paths: ModelAssetPaths) -> Any:
        with open(paths.label_encoder, "rb") as label_file:
            return pickle.load(label_file)

    @staticmethod
    def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
        """Apply the same pipeline used in training for grayscale handwriting images."""
        np_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_bytes, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise PredictionInputError("Unable to decode uploaded image.")

        image = cv2.resize(image, (256, 256))
        _, image = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY)

        struct = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        image = cv2.dilate(~image, struct, iterations=1)
        image = ~image

        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        return image

    def build_prediction_payload(
        self,
        probabilities: np.ndarray,
        classes: list[str],
    ) -> dict[str, Any]:
        if probabilities.ndim != 1:
            raise ValueError("Probabilities must be a 1D array.")

        if not classes or len(classes) != len(probabilities):
            classes = [f"CLASS_{idx}" for idx in range(len(probabilities))]

        top_idx = int(np.argmax(probabilities))
        predicted_class = classes[top_idx]
        predicted_emotion = self.CLASS_TO_EMOTION.get(predicted_class, predicted_class)

        emotion_probabilities = {emotion: 0.0 for emotion in self.SUPPORTED_EMOTIONS}
        for class_name, score in zip(classes, probabilities):
            mapped = self.CLASS_TO_EMOTION.get(class_name)
            if mapped in emotion_probabilities:
                emotion_probabilities[mapped] = float(score)

        return {
            "predicted_emotion": predicted_emotion,
            "predicted_class": predicted_class,
            "confidence": float(probabilities[top_idx]),
            "probabilities": emotion_probabilities,
            "happy_status": "pending_not_trained",
        }

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        if not self.model_loaded or self.model is None or self.label_encoder is None:
            details = self.model_error or "Model not loaded."
            raise ModelUnavailableError(details)

        processed = self.preprocess_image_bytes(image_bytes)
        outputs = self.model.predict(processed, verbose=0)

        probabilities = np.asarray(outputs[0], dtype=np.float32)
        classes = [str(item) for item in getattr(self.label_encoder, "classes_", [])]
        return self.build_prediction_payload(probabilities, classes)
