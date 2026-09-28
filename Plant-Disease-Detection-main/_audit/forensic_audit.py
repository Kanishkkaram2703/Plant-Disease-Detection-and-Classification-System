"""Read-only forensic audit harness for the legacy Plant Disease project.

This file intentionally lives under _audit and does not alter production code,
model files, archives, or application configuration. It uses the saved
checkpoint and a deterministic, audit-only sample selected from the archive.
That sample is not an independent test set because the notebook did not save
the original split membership or random seed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.metrics import classification_report, confusion_matrix

ImageFile.LOAD_TRUNCATED_IMAGES = False

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "_audit"
APP = ROOT / "Flask Deployed App"
MODEL_PATH = ROOT / "Model" / "plant_disease_model_1_latest.pt"
TRAIN_ARCHIVE = (
    ROOT
    / "Data for Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network"
    / "Plant_leaf_diseases_dataset_with_augmentation.zip"
)
BASE_ARCHIVE = TRAIN_ARCHIVE.with_name("Plant_leaf_diseases_dataset_without_augmentation.zip")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Background_without_leaves",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def archive_infos(archive: Path) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(archive) as zf:
        return [
            info
            for info in zf.infolist()
            if not info.is_dir() and Path(info.filename).suffix.lower() in IMAGE_SUFFIXES
        ]


def class_from_member(name: str) -> str:
    parts = Path(name.replace("/", "\\")).parts
    if len(parts) < 2:
        raise ValueError(f"Cannot determine class from archive member {name!r}")
    return parts[-2]


def archive_metadata(archive: Path) -> dict[str, Any]:
    with zipfile.ZipFile(archive) as zf:
        all_files = [info for info in zf.infolist() if not info.is_dir()]
        images = [
            info
            for info in all_files
            if Path(info.filename).suffix.lower() in IMAGE_SUFFIXES
        ]
        counts = Counter(class_from_member(info.filename) for info in images)
        return {
            "archive": str(archive.relative_to(ROOT)),
            "archive_bytes": archive.stat().st_size,
            "zip_entries": len(zf.infolist()),
            "files": len(all_files),
            "images": len(images),
            "uncompressed_image_bytes": sum(info.file_size for info in images),
            "extensions": dict(Counter(Path(info.filename).suffix.lower() for info in images)),
            "classes": len(counts),
            "class_counts": dict(sorted(counts.items())),
            "minimum_class_size": min(counts.values()),
            "maximum_class_size": max(counts.values()),
            "class_names": sorted(counts),
        }


def validate_archive_images(archive: Path, do_hash: bool) -> dict[str, Any]:
    """Decode every archive image and optionally hash its exact bytes."""
    start = time.perf_counter()
    invalid: list[dict[str, str]] = []
    dimensions = Counter()
    modes = Counter()
    formats = Counter()
    hashes: dict[str, list[str]] = defaultdict(list)
    with zipfile.ZipFile(archive) as zf:
        infos = [
            info
            for info in zf.infolist()
            if not info.is_dir() and Path(info.filename).suffix.lower() in IMAGE_SUFFIXES
        ]
        for number, info in enumerate(infos, start=1):
            try:
                raw = zf.read(info)
                if do_hash:
                    hashes[hashlib.sha256(raw).hexdigest()].append(info.filename)
                with Image.open(BytesIO(raw)) as image:
                    image.verify()
                with Image.open(BytesIO(raw)) as image:
                    image.load()
                    dimensions[f"{image.width}x{image.height}"] += 1
                    modes[image.mode] += 1
                    formats[(image.format or "UNKNOWN").upper()] += 1
            except Exception as exc:  # record the file; do not hide decode failures
                invalid.append({"image": info.filename, "error": f"{type(exc).__name__}: {exc}"})
            if number % 5000 == 0:
                print(f"{archive.name}: decoded {number}/{len(infos)}", flush=True)
    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    return {
        "archive": str(archive.relative_to(ROOT)),
        "images_scanned": len(infos),
        "valid_images": len(infos) - len(invalid),
        "invalid_images": invalid,
        "dimensions": dict(dimensions),
        "modes": dict(modes),
        "formats": dict(formats),
        "exact_hash_scan": {
            "performed": do_hash,
            "unique_hashes": len(hashes) if do_hash else None,
            "duplicate_groups": len(duplicate_groups) if do_hash else None,
            "duplicate_extra_files": (
                sum(len(paths) - 1 for paths in duplicate_groups) if do_hash else None
            ),
            "groups": duplicate_groups if do_hash else [],
        },
        "seconds": time.perf_counter() - start,
    }


def cross_archive_duplicates() -> dict[str, Any]:
    """Compare exact bytes between the extracted augmented/non-augmented sets."""
    base_root = ROOT / "Data for Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network"
    roots = {
        "without_augmentation": next(base_root.glob("Plant_leaf_diseases_dataset_without_augmentation/Plant_leave_diseases_dataset_without_augmentation")),
        "with_augmentation": next(base_root.glob("Plant_leaf_diseases_dataset_with_augmentation/Plant_leave_diseases_dataset_with_augmentation")),
    }
    by_hash: dict[str, list[str]] = defaultdict(list)
    files_by_set: dict[str, int] = {}
    for set_name, root in roots.items():
        files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
        files_by_set[set_name] = len(files)
        for number, path in enumerate(files, start=1):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            by_hash[digest].append(f"{set_name}:{path.relative_to(root)}")
            if number % 10000 == 0:
                print(f"{set_name}: hashed {number}/{len(files)}", flush=True)
    cross_groups = [paths for paths in by_hash.values() if any(path.startswith("without_augmentation:") for path in paths) and any(path.startswith("with_augmentation:") for path in paths)]
    return {
        "files_by_set": files_by_set,
        "unique_hashes_across_both_sets": len(by_hash),
        "cross_archive_exact_duplicate_groups": len(cross_groups),
        "cross_archive_duplicate_files": sum(len(group) for group in cross_groups),
        "groups_sample": cross_groups[:100],
        "scope": "Exact byte equality only; no perceptual near-duplicate analysis",
    }


def load_model() -> tuple[torch.nn.Module, dict[str, Any]]:
    sys.path.insert(0, str(APP))
    import CNN  # type: ignore[import-not-found]

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = CNN.CNN(39)
    load_result = model.load_state_dict(checkpoint, strict=True)
    model.eval()
    with torch.inference_mode():
        probe = model(torch.zeros(1, 3, 224, 224))
    return model, {
        "checkpoint_type": type(checkpoint).__name__,
        "checkpoint_entries": len(checkpoint),
        "state_dict_load": str(load_result),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "input_shape": [1, 3, 224, 224],
        "output_shape": list(probe.shape),
        "output_is_finite": bool(torch.isfinite(probe).all()),
        "logit_min": float(probe.min()),
        "logit_max": float(probe.max()),
        "model_file_bytes": MODEL_PATH.stat().st_size,
    }


def training_preprocess(raw: bytes) -> torch.Tensor:
    with Image.open(BytesIO(raw)) as image:
        image = image.convert("RGB")
        width, height = image.size
        if width < height:
            new_width, new_height = 255, round(height * 255 / width)
        else:
            new_width, new_height = round(width * 255 / height), 255
        image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        left = max((new_width - 224) // 2, 0)
        top = max((new_height - 224) // 2, 0)
        image = image.crop((left, top, left + 224, top + 224))
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def app_preprocess(path_or_raw: Path | bytes) -> torch.Tensor:
    """Reproduce app.py: PIL resize directly to 224 then TF.to_tensor semantics."""
    if isinstance(path_or_raw, Path):
        image = Image.open(path_or_raw)
    else:
        image = Image.open(BytesIO(path_or_raw))
    image = image.resize((224, 224))
    array = np.asarray(image)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3 or array.shape[2] not in {1, 3, 4}:
        raise ValueError(f"Unsupported app input shape {array.shape}")
    return torch.from_numpy(array.astype(np.float32) / 255.0).permute(2, 0, 1)


def evaluate_archive(model: torch.nn.Module, batch_size: int) -> dict[str, Any]:
    """Evaluate a deterministic audit-only 1% per-class sample."""
    with zipfile.ZipFile(TRAIN_ARCHIVE) as zf:
        by_class: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
        for info in zf.infolist():
            if not info.is_dir() and Path(info.filename).suffix.lower() in IMAGE_SUFFIXES:
                by_class[class_from_member(info.filename)].append(info)
        selected: list[tuple[str, zipfile.ZipInfo]] = []
        for class_name in sorted(by_class):
            members = sorted(by_class[class_name], key=lambda item: item.filename)
            selected.extend((class_name, info) for info in members[::100])

        labels: list[int] = []
        predictions: list[int] = []
        confidences: list[float] = []
        top3_values: list[list[tuple[str, float]]] = []
        paths: list[str] = []
        class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
        batch_tensors: list[torch.Tensor] = []
        batch_meta: list[tuple[str, str]] = []
        start = time.perf_counter()

        def flush() -> None:
            if not batch_tensors:
                return
            with torch.inference_mode():
                logits = model(torch.stack(batch_tensors))
                probabilities = torch.softmax(logits, dim=1)
                values, indices = probabilities.topk(3, dim=1)
            for position, (actual_name, member_name) in enumerate(batch_meta):
                labels.append(class_to_index[actual_name])
                predictions.append(int(indices[position, 0]))
                confidences.append(float(values[position, 0]))
                top3_values.append(
                    [
                        (CLASS_NAMES[int(indices[position, rank])], float(values[position, rank]))
                        for rank in range(3)
                    ]
                )
                paths.append(member_name)
            batch_tensors.clear()
            batch_meta.clear()

        for number, (actual_name, info) in enumerate(selected, start=1):
            batch_tensors.append(training_preprocess(zf.read(info)))
            batch_meta.append((actual_name, info.filename))
            if len(batch_tensors) >= batch_size:
                flush()
            if number % 1000 == 0:
                print(f"audit evaluation: {number}/{len(selected)}", flush=True)
        flush()

    y_true = np.asarray(labels, dtype=np.int64)
    y_pred = np.asarray(predictions, dtype=np.int64)
    confidence_array = np.asarray(confidences, dtype=np.float64)
    top3_accuracy = float(
        np.mean(
            [
                CLASS_NAMES[true_label] in [name for name, _ in candidates]
                for true_label, candidates in zip(y_true, top3_values)
            ]
        )
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    error_rows: list[dict[str, Any]] = []
    for index in np.flatnonzero(y_true != y_pred):
        error_rows.append(
            {
                "image_path": paths[index],
                "actual_class": CLASS_NAMES[y_true[index]],
                "predicted_class": CLASS_NAMES[y_pred[index]],
                "confidence": float(confidence_array[index]),
                "top_3": " | ".join(
                    f"{name}:{score:.8f}" for name, score in top3_values[index]
                ),
            }
        )

    thresholds = {}
    for threshold in (0.80, 0.85, 0.90, 0.95):
        accepted = confidence_array >= threshold
        wrong = (y_true != y_pred) & accepted
        thresholds[str(threshold)] = {
            "count": int(accepted.sum()),
            "coverage": float(accepted.mean()),
            "errors": int(wrong.sum()),
            "accuracy_when_accepted": (
                float((y_true[accepted] == y_pred[accepted]).mean()) if accepted.any() else None
            ),
        }

    buckets = [
        (0.0, 0.50, "<0.50"),
        (0.50, 0.60, "0.50-0.60"),
        (0.60, 0.70, "0.60-0.70"),
        (0.70, 0.80, "0.70-0.80"),
        (0.80, 0.90, "0.80-0.90"),
        (0.90, 0.95, "0.90-0.95"),
        (0.95, 1.000001, "0.95-1.00"),
    ]
    confidence_buckets = []
    for lower, upper, label in buckets:
        selected_bucket = (confidence_array >= lower) & (confidence_array < upper)
        confidence_buckets.append(
            {
                "bucket": label,
                "count": int(selected_bucket.sum()),
                "accuracy": (
                    float((y_true[selected_bucket] == y_pred[selected_bucket]).mean())
                    if selected_bucket.any()
                    else None
                ),
                "errors": int(((y_true != y_pred) & selected_bucket).sum()),
            }
        )

    pair_counts = Counter(
        (CLASS_NAMES[y_true[index]], CLASS_NAMES[y_pred[index]])
        for index in np.flatnonzero(y_true != y_pred)
    )
    class_metrics = []
    for class_name in CLASS_NAMES:
        row = report[class_name]
        class_metrics.append(
            {
                "class": class_name,
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1-score"],
                "support": int(row["support"]),
            }
        )

    fields = ["image_path", "actual_class", "predicted_class", "confidence", "top_3"]
    write_csv(AUDIT / "error_analysis.csv", error_rows, fields)
    write_csv(
        AUDIT / "high_confidence_errors.csv",
        [row for row in error_rows if row["confidence"] >= 0.80],
        fields,
    )
    write_csv(
        AUDIT / "classification_report.csv",
        class_metrics,
        ["class", "precision", "recall", "f1", "support"],
    )
    matrix_rows = []
    for actual_index, actual_name in enumerate(CLASS_NAMES):
        row = {"actual_class": actual_name}
        row.update({predicted_name: int(matrix[actual_index, predicted_index]) for predicted_index, predicted_name in enumerate(CLASS_NAMES)})
        matrix_rows.append(row)
    write_csv(AUDIT / "confusion_matrix.csv", matrix_rows, ["actual_class", *CLASS_NAMES])

    metrics = {
        "evaluation_protocol": {
            "label": "AUDIT TEST SET",
            "independent": False,
            "reason": "The notebook did not save split membership or a random seed; this deterministic every-hundredth-per-class sample is not independent of training.",
            "source_archive": str(TRAIN_ARCHIVE.relative_to(ROOT)),
            "selection": "Sorted archive members; every hundredth image per class",
            "images": len(labels),
            "class_counts": dict(Counter(CLASS_NAMES[label] for label in labels)),
        },
        "metrics": {
            "accuracy": float((y_true == y_pred).mean()),
            "top_1_accuracy": float((y_true == y_pred).mean()),
            "top_3_accuracy": top3_accuracy,
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_precision": report["weighted avg"]["precision"],
            "weighted_recall": report["weighted avg"]["recall"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "total_errors": int((y_true != y_pred).sum()),
            "error_rate": float((y_true != y_pred).mean()),
        },
        "confidence": {
            "mean_correct": float(confidence_array[y_true == y_pred].mean()),
            "mean_incorrect": float(confidence_array[y_true != y_pred].mean()),
            "thresholds": thresholds,
            "buckets": confidence_buckets,
        },
        "confusion_pairs": [
            {"actual": actual, "predicted": predicted, "count": count}
            for (actual, predicted), count in pair_counts.most_common()
        ],
        "class_metrics": class_metrics,
        "seconds": time.perf_counter() - start,
    }
    write_json(AUDIT / "metrics.json", metrics)
    return metrics


def infer_image(model: torch.nn.Module, path: Path) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        tensor = app_preprocess(path).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.softmax(model(tensor), dim=1)[0]
        values, indices = probabilities.topk(3)
        return {
            "image": str(path.relative_to(ROOT)),
            "status": "PASS",
            "predicted_class": CLASS_NAMES[int(indices[0])],
            "confidence": float(values[0]),
            "top_3": [
                {"class": CLASS_NAMES[int(indices[i])], "confidence": float(values[i])}
                for i in range(3)
            ],
            "seconds": time.perf_counter() - start,
        }
    except Exception as exc:
        return {
            "image": str(path.relative_to(ROOT)),
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": time.perf_counter() - start,
        }


def real_image_inference(model: torch.nn.Module) -> list[dict[str, Any]]:
    paths = sorted((ROOT / "test_images").glob("*")) + sorted((ROOT / "demo_images").glob("*"))
    paths = [path for path in paths if path.suffix.lower() in IMAGE_SUFFIXES]
    results: list[dict[str, Any]] = []
    good_paths: list[Path] = []
    tensors: list[torch.Tensor] = []
    for path in paths:
        try:
            tensor = app_preprocess(path)
            if tensor.shape[0] != 3:
                raise ValueError(
                    f"app.py expects 3 channels but input has {tensor.shape[0]} channels"
                )
            tensors.append(tensor)
            good_paths.append(path)
        except Exception as exc:
            results.append(
                {
                    "image": str(path.relative_to(ROOT)),
                    "status": "FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if tensors:
        start = time.perf_counter()
        with torch.inference_mode():
            probabilities = torch.softmax(model(torch.stack(tensors)), dim=1)
        values, indices = probabilities.topk(3, dim=1)
        for position, path in enumerate(good_paths):
            results.append(
                {
                    "image": str(path.relative_to(ROOT)),
                    "status": "PASS",
                    "predicted_class": CLASS_NAMES[int(indices[position, 0])],
                    "confidence": float(values[position, 0]),
                    "top_3": [
                        {
                            "class": CLASS_NAMES[int(indices[position, rank])],
                            "confidence": float(values[position, rank]),
                        }
                        for rank in range(3)
                    ],
                    "batch_seconds": time.perf_counter() - start,
                }
            )
    results.sort(key=lambda row: row["image"])
    write_json(AUDIT / "real_image_predictions.json", results)
    expected_by_filename = {
        "Apple_ceder_apple_rust.JPG": "Apple___Cedar_apple_rust",
        "Apple_scab.JPG": "Apple___Apple_scab",
        "apple_black_rot.JPG": "Apple___Black_rot",
        "apple_healthy.JPG": "Apple___healthy",
        "background_without_leaves.jpg": "Background_without_leaves",
        "blueberry_healthy.JPG": "Blueberry___healthy",
        "cherry_healthy.JPG": "Cherry___healthy",
        "cherry_powdery_mildew.JPG": "Cherry___Powdery_mildew",
        "corn_cercospora_leaf.JPG": "Corn___Cercospora_leaf_spot Gray_leaf_spot",
        "corn_common_rust.JPG": "Corn___Common_rust",
        "corn_healthy.jpg": "Corn___healthy",
        "corn_northen_leaf_blight.JPG": "Corn___Northern_Leaf_Blight",
        "grape_black_rot.JPG": "Grape___Black_rot",
        "Grape_esca.JPG": "Grape___Esca_(Black_Measles)",
        "grape_healthy.JPG": "Grape___healthy",
        "grape_leaf_blight.JPG": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "orange_haunglongbing.JPG": "Orange___Haunglongbing_(Citrus_greening)",
        "peach_bacterial_spot.JPG": "Peach___Bacterial_spot",
        "peach_healthy.JPG": "Peach___healthy",
        "pepper_bacterial_spot.JPG": "Pepper,_bell___Bacterial_spot",
        "pepper_bell_healthy.JPG": "Pepper,_bell___healthy",
        "potato_early_blight.JPG": "Potato___Early_blight",
        "potato_healthy.JPG": "Potato___healthy",
        "potato_late_blight.JPG": "Potato___Late_blight",
        "raspberry_healthy.JPG": "Raspberry___healthy",
        "soyaben healthy.JPG": "Soybean___healthy",
        "squash_powdery_mildew.JPG": "Squash___Powdery_mildew",
        "starwberry_healthy.JPG": "Strawberry___healthy",
        "starwberry_leaf_scorch.JPG": "Strawberry___Leaf_scorch",
        "tomato-bacterial-spot2.jpg": "Tomato___Bacterial_spot",
        "tomato-leaf-curl-virus3.jpg": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "tomato-mold.jpg": "Tomato___Leaf_Mold",
        "tomato_bacterial_spot.JPG": "Tomato___Bacterial_spot",
        "tomato_early_blight.JPG": "Tomato___Early_blight",
        "tomato_healthy.JPG": "Tomato___healthy",
        "tomato_late_blight.JPG": "Tomato___Late_blight",
        "tomato_leaf_mold.JPG": "Tomato___Leaf_Mold",
        "tomato_mosaic_virus.JPG": "Tomato___Tomato_mosaic_virus",
        "tomato_septoria_leaf_spot.JPG": "Tomato___Septoria_leaf_spot",
        "tomato_spider_mites_two_spotted_spider_mites.JPG": "Tomato___Spider_mites Two-spotted_spider_mite",
        "tomato_target_spot.JPG": "Tomato___Target_Spot",
        "tomato_yellow_leaf_curl_virus.JPG": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "tomato_yellow_leaf_curl_virus2.jpg": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    }
    csv_rows = []
    for row in results:
        filename = Path(row["image"]).name
        expected = expected_by_filename.get(filename)
        csv_rows.append(
            {
                "image": row["image"],
                "status": row["status"],
                "predicted_class": row.get("predicted_class", ""),
                "confidence": row.get("confidence", ""),
                "top_3": " | ".join(
                    f"{item['class']}:{item['confidence']:.8f}" for item in row.get("top_3", [])
                ),
                "expected_class_if_determinable": expected or "",
                "expected_class_basis": "filename only; not verified ground truth" if expected else "not determinable from repository",
                "filename_comparison": (
                    "MATCH" if expected and expected == row.get("predicted_class") else "MISMATCH"
                    if expected and row.get("status") == "PASS" else "NOT_VERIFIABLE"
                ),
            }
        )
    write_csv(
        AUDIT / "real_image_predictions.csv",
        csv_rows,
        [
            "image",
            "status",
            "predicted_class",
            "confidence",
            "top_3",
            "expected_class_if_determinable",
            "expected_class_basis",
            "filename_comparison",
        ],
    )
    return results


def inventory() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or AUDIT in path.parents:
            continue
        suffix = path.suffix.lower() or "[none]"
        relative = str(path.relative_to(ROOT))
        if relative in {"Flask Deployed App\\app.py", "Flask Deployed App\\CNN.py"} or "Flask Deployed App\\templates" in relative:
            purpose, used_by, status = "Flask inference/web application", "Flask app", "IMPLEMENTED"
        elif relative == "Model\\Plant Disease Detection Code.ipynb":
            purpose, used_by, status = "Training/evaluation notebook", "Historical training workflow", "PARTIALLY IMPLEMENTED"
        elif relative == "Model\\plant_disease_model_1_latest.pt":
            purpose, used_by, status = "Saved PyTorch checkpoint", "Flask app and audit harness", "IMPLEMENTED"
        elif relative in {"Flask Deployed App\\disease_info.csv", "Flask Deployed App\\supplement_info.csv", "Flask Deployed App\\requirements.txt", "Flask Deployed App\\Procfile"}:
            purpose, used_by, status = "Application data/deployment configuration", "Flask app/deployment", "IMPLEMENTED"
        elif "Plant_leaf_diseases_dataset_with_augmentation" in relative:
            purpose, used_by, status = "Augmented dataset artifact", "Notebook dataset by matching 61,486 count", "IMPLEMENTED"
        elif "Plant_leaf_diseases_dataset_without_augmentation" in relative:
            purpose, used_by, status = "Non-augmented dataset artifact", "No evidence current notebook used it", "UNUSED"
        elif relative.startswith("test_images\\") or relative.startswith("demo_images\\"):
            purpose, used_by, status = "Manual/demo inference image", "Notebook examples or Flask upload testing", "IMPLEMENTED"
        elif suffix in {".pdf", ".md"}:
            purpose, used_by, status = "Documentation/reference", "Human reference; no runtime import", "UNKNOWN"
        else:
            purpose, used_by, status = "Unclassified project artifact", "Not established", "UNKNOWN"
        rows.append(
            {
                "file": relative,
                "purpose": purpose,
                "used_by": used_by,
                "important_content": suffix,
                "status": status,
                "bytes": path.stat().st_size,
            }
        )
    write_csv(AUDIT / "inventory.csv", rows, ["file", "purpose", "used_by", "important_content", "status", "bytes"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode", action="store_true", help="Decode/hash every image in both dataset archives")
    parser.add_argument("--evaluate", action="store_true", help="Run the audit-only model evaluation")
    parser.add_argument("--real-images", action="store_true", help="Run inference on test_images and demo_images")
    parser.add_argument("--cross-archive", action="store_true", help="Compare exact hashes between extracted archive variants")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    AUDIT.mkdir(exist_ok=True)
    inventory()
    archive_report = {"without_augmentation": archive_metadata(BASE_ARCHIVE), "with_augmentation": archive_metadata(TRAIN_ARCHIVE)}
    write_json(AUDIT / "archive_metadata.json", archive_report)
    model, model_report = load_model()
    write_json(AUDIT / "model_load.json", model_report)
    if args.decode:
        write_json(AUDIT / "decode_without_augmentation.json", validate_archive_images(BASE_ARCHIVE, do_hash=True))
        write_json(AUDIT / "decode_with_augmentation.json", validate_archive_images(TRAIN_ARCHIVE, do_hash=True))
    if args.evaluate:
        evaluate_archive(model, args.batch_size)
    if args.real_images:
        real_image_inference(model)
    if args.cross_archive:
        write_json(AUDIT / "cross_archive_duplicates.json", cross_archive_duplicates())
    print(json.dumps({"archive": archive_report, "model": model_report}, indent=2, default=json_default))


if __name__ == "__main__":
    main()
