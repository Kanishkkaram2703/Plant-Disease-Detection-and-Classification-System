"""Plant Doctor model and knowledge service.

This module owns the legacy 39-class model integration.  It intentionally
does not import torchvision: the original application pins an old torchvision
build that is not importable in the current CPU environment, while the model's
training transform can be reproduced with Pillow and torch directly.
"""

from __future__ import annotations

import csv
import io
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile, UnidentifiedImageError

try:
    from .CNN import CNN, idx_to_classes
except ImportError:  # pragma: no cover - supports running app.py directly
    from CNN import CNN, idx_to_classes


ImageFile.LOAD_TRUNCATED_IMAGES = False

MODEL_VERSION = "plant_disease_model_1_latest.pt"
INPUT_SIZE = 224
RESIZE_SHORT_SIDE = 255
CLASS_COUNT = 39
MAX_IMAGE_BYTES = 8 * 1024 * 1024
DEFAULT_UNCERTAINTY_THRESHOLD = 0.90
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class PlantDoctorError(Exception):
    """Base error for safe, user-facing prediction failures."""


class InvalidImageError(PlantDoctorError):
    """Raised when an upload is not a supported, decodable image."""


class ModelUnavailableError(PlantDoctorError):
    """Raised when the model cannot be used for inference."""


def _clean_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _humanize(value: str) -> str:
    value = value.replace("_", " ").replace(",", ", ")
    return " ".join(value.split()).strip().title()


def split_class_name(class_name: str) -> Tuple[str, str]:
    """Return farmer-facing plant and condition labels from the verified key."""

    if class_name == "Background_without_leaves":
        return "Leaf image", "No leaf detected"
    if "___" not in class_name:
        return _humanize(class_name), ""
    plant, condition = class_name.split("___", 1)
    return _humanize(plant), _humanize(condition)


def load_catalog(app_dir: Path) -> List[Dict[str, Any]]:
    """Load only the verified disease and supplement CSV fields."""

    disease_path = app_dir / "disease_info.csv"
    supplement_path = app_dir / "supplement_info.csv"
    with disease_path.open("r", encoding="cp1252", newline="") as handle:
        disease_rows = list(csv.DictReader(handle))
    with supplement_path.open("r", encoding="cp1252", newline="") as handle:
        supplement_rows = list(csv.DictReader(handle))

    supplements = {
        _clean_value(row.get("disease_name")): row
        for row in supplement_rows
        if _clean_value(row.get("disease_name"))
    }
    catalog: List[Dict[str, Any]] = []
    for row in disease_rows:
        index = int(row["index"])
        class_key = idx_to_classes[index]
        plant, condition = split_class_name(class_key)
        supplement = supplements.get(class_key, {})
        catalog.append(
            {
                "index": index,
                "class_key": class_key,
                "plant": plant,
                "condition": condition,
                "display_name": f"{plant} — {condition}",
                "is_healthy": class_key.endswith("___healthy"),
                "is_background": class_key == "Background_without_leaves",
                "description": _clean_value(row.get("description")),
                "management": _clean_value(row.get("Possible Steps")),
                "source_image_url": _clean_value(row.get("image_url")),
                "supplement_name": _clean_value(supplement.get("supplement name")),
                "supplement_image_url": _clean_value(supplement.get("supplement image")),
                "supplement_buy_url": _clean_value(supplement.get("buy link")),
            }
        )
    return sorted(catalog, key=lambda item: item["index"])


def _resize_short_side(image: Image.Image, short_side: int) -> Image.Image:
    """Reproduce torchvision.transforms.Resize(int) for the training pipeline."""

    width, height = image.size
    if width <= 0 or height <= 0:
        raise InvalidImageError("The image has invalid dimensions.")
    if width < height:
        new_width = short_side
        new_height = int(short_side * height / width)
    else:
        new_height = short_side
        new_width = int(short_side * width / height)
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return image.resize((new_width, new_height), resampling)


def training_preprocess(image: Image.Image) -> torch.Tensor:
    """Apply Resize(255) -> CenterCrop(224) -> ToTensor exactly."""

    image = image.convert("RGB")
    image = _resize_short_side(image, RESIZE_SHORT_SIDE)
    width, height = image.size
    left = max(0, (width - INPUT_SIZE) // 2)
    top = max(0, (height - INPUT_SIZE) // 2)
    image = image.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))
    # PIL/torchvision ToTensor converts uint8 RGB pixels to CHW float32 / 255.
    tensor = torch.from_numpy(np.asarray(image, dtype="float32"))
    tensor = tensor.permute(2, 0, 1).contiguous().div(255.0)
    return tensor.unsqueeze(0)


class PlantDoctorService:
    """Persistent CPU model wrapper with verified class/knowledge mapping."""

    def __init__(self, app_dir: Path, uncertainty_threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD):
        self.app_dir = Path(app_dir)
        self.model_path = self.app_dir.parent / "Model" / MODEL_VERSION
        self.upload_dir = self.app_dir / "static" / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.catalog = load_catalog(self.app_dir)
        self.catalog_by_index = {item["index"]: item for item in self.catalog}
        self.catalog_by_class = {item["class_key"]: item for item in self.catalog}
        self.class_names = [idx_to_classes[index] for index in range(CLASS_COUNT)]
        self.model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        if not self.model_path.is_file():
            raise ModelUnavailableError("The configured plant disease model is missing.")
        model = CNN(CLASS_COUNT)
        try:
            state_dict = torch.load(self.model_path, map_location="cpu", weights_only=True)
        except TypeError:  # torch < 2.0 compatibility
            state_dict = torch.load(self.model_path, map_location="cpu")
        try:
            model.load_state_dict(state_dict, strict=True)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ModelUnavailableError("The plant disease model could not be loaded.") from exc
        model.eval()
        return model

    @staticmethod
    def _open_image(data: bytes) -> Image.Image:
        if not data:
            raise InvalidImageError("Choose an image before analyzing.")
        if len(data) > MAX_IMAGE_BYTES:
            raise InvalidImageError("Images must be 8 MB or smaller.")
        try:
            with Image.open(io.BytesIO(data)) as source:
                source.verify()
            with Image.open(io.BytesIO(data)) as source:
                image_format = (source.format or "").lower()
                if image_format not in {"jpeg", "png", "webp"}:
                    raise InvalidImageError("Please upload a JPEG, PNG, or WebP image.")
                image = source.convert("RGB")
                image.load()
                return image
        except InvalidImageError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageError("That file is not a supported, readable image.") from exc

    def predict_bytes(self, data: bytes) -> Dict[str, Any]:
        """Predict from upload bytes using the same canonical path for upload/camera."""

        if self.model is None:
            raise ModelUnavailableError("The plant disease model is unavailable.")
        image = self._open_image(data)
        tensor = training_preprocess(image)
        try:
            with torch.no_grad():
                logits = self.model(tensor)
                probabilities = torch.softmax(logits, dim=1)[0]
                top_values, top_indices = torch.topk(probabilities, k=3)
        except (RuntimeError, ValueError) as exc:
            raise ModelUnavailableError("The image could not be analyzed right now.") from exc

        top_predictions = []
        for probability, class_index in zip(top_values.tolist(), top_indices.tolist()):
            top_predictions.append(self._prediction_item(int(class_index), float(probability)))
        primary = top_predictions[0]
        return {
            "model_version": MODEL_VERSION,
            "input_size": f"{INPUT_SIZE} × {INPUT_SIZE}",
            "probability": primary["probability"],
            "is_uncertain": primary["probability"] < self.uncertainty_threshold,
            "uncertainty_threshold": self.uncertainty_threshold,
            "threshold_note": "Screening threshold based on the audit sample; softmax scores are not calibrated probabilities.",
            "prediction": primary,
            "top_predictions": top_predictions,
            "knowledge": primary["knowledge"],
        }

    def _prediction_item(self, class_index: int, probability: float) -> Dict[str, Any]:
        class_key = self.class_names[class_index]
        knowledge = self.catalog_by_index[class_index]
        return {
            "index": class_index,
            "class_key": class_key,
            "plant": knowledge["plant"],
            "condition": knowledge["condition"],
            "display_name": knowledge["display_name"],
            "probability": probability,
            "knowledge": knowledge,
        }

    def save_normalized_image(self, data: bytes) -> str:
        """Persist a safe RGB preview after successful validation."""

        image = self._open_image(data)
        filename = f"{uuid.uuid4().hex}.jpg"
        path = self.upload_dir / filename
        image.save(path, format="JPEG", quality=90, optimize=True)
        return filename
