from __future__ import annotations

import io
import sys
import unittest
import uuid
from pathlib import Path

from pymongo.errors import PyMongoError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "Flask Deployed App"
sys.path.insert(0, str(APP_DIR))

from app import app, farmer_service  # noqa: E402
from mongo import store  # noqa: E402


def mongo_is_available() -> bool:
    try:
        store.ping()
        return True
    except PyMongoError:
        return False


@unittest.skipUnless(
    mongo_is_available(),
    "Dedicated PlantCare MongoDB is not running on 127.0.0.1:27018.",
)
class FarmerMongoIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()
        self.email = f"codex-test-{uuid.uuid4().hex}@example.invalid"
        response = self.client.post(
            "/api/profile",
            json={"name": "Integration Test Farmer", "email": self.email, "farm_name": "Test Farm"},
        )
        self.assertEqual(response.status_code, 201, response.get_json())
        self.user_id = response.get_json()["data"]["_id"]
        self.created_ids = {"fields": [], "crops": [], "tasks": [], "notifications": []}

    def tearDown(self) -> None:
        from bson import ObjectId

        user_id = ObjectId(self.user_id)
        for collection in (
            "fields",
            "crop_cycles",
            "care_tasks",
            "irrigation_logs",
            "plant_health_logs",
            "predictions",
            "notifications",
        ):
            store.collection(collection).delete_many({"user_id": user_id})
        store.collection("users").delete_one({"_id": user_id})

    def test_profile_owned_farmer_records_and_dashboard(self) -> None:
        field_response = self.client.post(
            "/api/fields",
            json={
                "field_name": "North Test Field",
                "area": 2.5,
                "area_unit": "acres",
                "soil_type": "Loam",
                "irrigation_method": "Drip",
                "location": {"village": "Test Village", "state": "Test State"},
            },
        )
        self.assertEqual(field_response.status_code, 200, field_response.get_json())
        field_id = field_response.get_json()["data"]["_id"]

        crop_response = self.client.post(
            "/api/crops",
            json={
                "field_id": field_id,
                "crop_name": "Tomato",
                "variety": "Test variety",
                "planting_date": "2026-09-01",
                "current_growth_stage": "Vegetative",
            },
        )
        self.assertEqual(crop_response.status_code, 200, crop_response.get_json())
        crop_id = crop_response.get_json()["data"]["_id"]

        task_response = self.client.post(
            "/api/tasks",
            json={
                "field_id": field_id,
                "crop_cycle_id": crop_id,
                "title": "Inspect test field",
                "due_date": "2026-09-28",
                "priority": "high",
            },
        )
        self.assertEqual(task_response.status_code, 200, task_response.get_json())
        task_id = task_response.get_json()["data"]["_id"]
        self.assertEqual(self.client.post(f"/api/tasks/{task_id}/complete").status_code, 200)

        irrigation_response = self.client.post(
            "/api/irrigation",
            json={"field_id": field_id, "crop_cycle_id": crop_id, "irrigation_method": "Drip", "duration_minutes": 20},
        )
        self.assertEqual(irrigation_response.status_code, 200, irrigation_response.get_json())

        health_response = self.client.post(
            "/api/health-logs",
            json={
                "field_id": field_id,
                "crop_cycle_id": crop_id,
                "observation_date": "2026-09-28",
                "health_status": "Observed healthy",
            },
        )
        self.assertEqual(health_response.status_code, 200, health_response.get_json())

        overview = self.client.get(f"/api/fields/{field_id}/overview")
        self.assertEqual(overview.status_code, 200, overview.get_json())
        self.assertEqual(len(overview.get_json()["data"]["crops"]), 1)
        self.assertEqual(len(overview.get_json()["data"]["tasks"]), 1)
        self.assertEqual(len(overview.get_json()["data"]["irrigation"]), 1)
        self.assertEqual(len(overview.get_json()["data"]["health"]), 1)

        dashboard = self.client.get("/api/dashboard")
        self.assertEqual(dashboard.status_code, 200, dashboard.get_json())
        stats = dashboard.get_json()["data"]["stats"]
        self.assertEqual(stats["active_fields"], 1)
        self.assertEqual(stats["active_crops"], 1)
        self.assertEqual(stats["open_tasks"], 0)

        notifications = self.client.get("/api/notifications")
        self.assertEqual(notifications.status_code, 200, notifications.get_json())
        notification_id = notifications.get_json()["data"][0]["_id"]
        self.assertEqual(self.client.post(f"/api/notifications/{notification_id}/read").status_code, 200)
        self.assertEqual(self.client.post("/api/notifications/read-all").status_code, 200)

        settings = self.client.get("/api/settings")
        self.assertEqual(settings.status_code, 200, settings.get_json())
        updated_settings = self.client.patch(
            "/api/settings", json={"theme": "dark", "unit_preference": "metric", "notifications_enabled": False}
        )
        self.assertEqual(updated_settings.status_code, 200, updated_settings.get_json())
        self.assertFalse(self.client.get("/api/settings").get_json()["data"]["notifications_enabled"])

        archived = self.client.delete(f"/api/fields/{field_id}")
        self.assertEqual(archived.status_code, 200, archived.get_json())
        self.assertEqual(archived.get_json()["data"], {"deleted": False, "archived": True})
        self.assertEqual(self.client.get(f"/api/fields/{field_id}").status_code, 200)

    def test_prediction_is_saved_with_real_model_output(self) -> None:
        field_response = self.client.post(
            "/api/fields", json={"field_name": "Prediction Field", "area": 1, "area_unit": "acres"}
        )
        self.assertEqual(field_response.status_code, 200, field_response.get_json())
        field_id = field_response.get_json()["data"]["_id"]
        image_path = PROJECT_ROOT / "test_images" / "apple_healthy.JPG"
        with image_path.open("rb") as image_handle:
            response = self.client.post(
                "/api/predict",
                data={"field_id": field_id, "image": (io.BytesIO(image_handle.read()), image_path.name)},
                content_type="multipart/form-data",
            )
        payload = response.get_json()
        self.assertEqual(response.status_code, 200, payload)
        self.assertTrue(payload["persistence"]["saved"])
        self.assertEqual(self.client.get("/api/predictions").status_code, 200)
        stored_filename = payload.get("stored_filename")
        if stored_filename:
            generated = APP_DIR / "static" / "uploads" / stored_filename
            if generated.is_file():
                generated.unlink()

    def test_profile_ownership_blocks_another_profile(self) -> None:
        field_response = self.client.post(
            "/api/fields", json={"field_name": "Private Field", "area": 1, "area_unit": "acres"}
        )
        field_id = field_response.get_json()["data"]["_id"]
        other = app.test_client()
        other_email = f"codex-other-{uuid.uuid4().hex}@example.invalid"
        try:
            profile = other.post("/api/profile", json={"name": "Other Farmer", "email": other_email})
            self.assertEqual(profile.status_code, 201, profile.get_json())
            self.assertEqual(other.get(f"/api/fields/{field_id}").status_code, 404)
            from bson import ObjectId

            store.collection("users").delete_one({"_id": ObjectId(profile.get_json()["data"]["_id"])})
        finally:
            store.collection("users").delete_many({"email": other_email})

    def test_verified_rule_generation_is_deduplicated(self) -> None:
        field_response = self.client.post(
            "/api/fields", json={"field_name": "Rule Field", "area": 1, "area_unit": "acres", "soil_type": "Loamy"}
        )
        self.assertEqual(field_response.status_code, 200, field_response.get_json())
        field_id = field_response.get_json()["data"]["_id"]
        rule_key = f"codex-test-{uuid.uuid4().hex}"
        store.collection("agricultural_rules").insert_one(
            {
                "rule_id": rule_key,
                "name": "Test crop observation rule",
                "enabled": True,
                "triggers": ["crop_created"],
                "conditions": {"crop_names": ["Tomato"]},
                "task": {
                    "title": "Review tomato crop stage",
                    "description": "Record the current crop stage for this test rule.",
                    "task_type": "Crop-stage observation",
                    "priority": "medium",
                    "days_after_trigger": 1,
                },
            }
        )
        try:
            crop_response = self.client.post(
                "/api/crops",
                json={"field_id": field_id, "crop_name": "Tomato", "planting_date": "2026-09-28", "current_growth_stage": "Vegetative"},
            )
            self.assertEqual(crop_response.status_code, 200, crop_response.get_json())
            crop_id = crop_response.get_json()["data"]["_id"]
            generated = [task for task in self.client.get("/api/tasks").get_json()["data"] if task.get("generated")]
            self.assertEqual(len(generated), 1)
            self.assertEqual(generated[0]["source_rule"], "Test crop observation rule")
            from bson import ObjectId

            farmer_service.evaluate_rules(ObjectId(self.user_id), "crop_created", field_id=ObjectId(field_id), crop_id=ObjectId(crop_id))
            generated_again = [task for task in self.client.get("/api/tasks").get_json()["data"] if task.get("generated")]
            self.assertEqual(len(generated_again), 1)
        finally:
            store.collection("agricultural_rules").delete_many({"rule_id": rule_key})


if __name__ == "__main__":
    unittest.main()
