from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "Flask Deployed App"
sys.path.insert(0, str(APP_DIR))

from app import app, doctor  # noqa: E402
from plant_doctor import CLASS_COUNT, training_preprocess  # noqa: E402


def _image_bytes(mode: str = "RGB", image_format: str = "PNG") -> bytes:
    image = Image.new(mode, (320, 240), (96, 150, 80, 180) if mode == "RGBA" else (96, 150, 80))
    handle = io.BytesIO()
    image.save(handle, format=image_format)
    return handle.getvalue()


class PlantDoctorTests(unittest.TestCase):
    def test_model_loads_with_verified_class_count(self):
        self.assertIsNotNone(doctor)
        self.assertEqual(len(doctor.class_names), CLASS_COUNT)
        self.assertEqual(CLASS_COUNT, 39)
        self.assertFalse(doctor.model.training)


    def test_training_preprocess_converts_rgba_to_rgb_tensor(self):
        with Image.open(io.BytesIO(_image_bytes("RGBA"))) as image:
            tensor = training_preprocess(image)
        self.assertEqual(tuple(tensor.shape), (1, 3, 224, 224))
        self.assertEqual(str(tensor.dtype), "torch.float32")
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)


    def test_health_reports_model_status(self):
        client = app.test_client()
        response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json() or {}
        self.assertEqual(payload["classes"], 39)
        self.assertTrue(payload["model_loaded"])


    def test_invalid_upload_is_rejected_without_inference(self):
        client = app.test_client()
        response = client.post(
            "/api/predict",
            data={"image": (io.BytesIO(b"not an image"), "broken.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertTrue("supported" in response.get_json()["error"] or "image" in response.get_json()["error"])


    def test_rgba_upload_uses_real_model_and_returns_top_three(self):
        client = app.test_client()
        response = client.post(
            "/api/predict",
            data={"image": (io.BytesIO(_image_bytes("RGBA")), "leaf.png")},
            content_type="multipart/form-data",
        )
        payload = response.get_json()
        try:
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(payload["top_predictions"]), 3)
            self.assertEqual(payload["model_version"], "plant_disease_model_1_latest.pt")
            self.assertGreaterEqual(payload["probability"], 0.0)
            self.assertLessEqual(payload["probability"], 1.0)
            self.assertTrue(payload["image_url"].startswith("/static/uploads/"))
        finally:
            generated = APP_DIR / "static" / "uploads" / payload.get("stored_filename", "")
            if generated.is_file() and generated.name != "Readme.md":
                generated.unlink()


    def test_existing_and_new_pages_render(self):
        client = app.test_client()
        paths = [
            "/",
            "/plant-doctor",
            "/index",
            "/history",
            "/library",
            "/market",
            "/fields",
            "/fields/add",
            "/fields/1",
            "/crop-management",
            "/irrigation",
            "/tasks",
            "/notifications",
            "/settings",
            "/profile",
            "/contact",
        ]
        for path in paths:
            response = client.get(path)
            self.assertIn(response.status_code, (200, 302), path)


    def test_camera_workflow_uses_browser_api_and_single_analysis_action(self):
        javascript = (APP_DIR / "static" / "js" / "app.js").read_text(encoding="utf-8")
        template = (APP_DIR / "templates" / "plant_doctor.html").read_text(encoding="utf-8")
        self.assertIn("getUserMedia", javascript)
        self.assertIn("toBlob", javascript)
        self.assertNotIn("continuous", javascript.lower())
        self.assertIn("data-capture", template)
        self.assertIn("data-analyze", template)
        self.assertIn("Capture photo", template)
        self.assertNotIn("About this screening", template)

    def test_library_has_real_condition_filter_data(self):
        client = app.test_client()
        response = client.get("/library")
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('data-library-filter', html)
        self.assertIn('data-condition="apple scab"', html.lower())
