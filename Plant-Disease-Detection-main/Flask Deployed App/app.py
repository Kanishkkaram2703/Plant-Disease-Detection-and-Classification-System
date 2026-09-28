"""Flask application for the legacy Plant Doctor model.

The application keeps the existing Flask entry point and routes, exposes a
small JSON prediction boundary so upload and camera captures use one service,
and stores farmer records in the dedicated PlantCare MongoDB instance.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple

from bson import ObjectId
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from farmer_service import (
    ConflictError,
    FarmerService,
    GROWTH_STAGES,
    INDIAN_STATES,
    IRRIGATION_METHODS,
    NotFoundError,
    OwnershipError,
    SOIL_DESCRIPTIONS,
    SOIL_TYPES,
    THEMES,
    UNIT_PREFERENCES,
    ValidationError,
)
from mongo import (
    DatabaseUnavailableError,
    InvalidObjectIdError,
    parse_object_id,
    serialize,
    store as mongo_store,
)
from plant_doctor import (
    MAX_IMAGE_BYTES,
    InvalidImageError,
    ModelUnavailableError,
    PlantDoctorError,
    PlantDoctorService,
    load_catalog,
)


APP_DIR = Path(__file__).resolve().parent
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("plant-doctor")

try:
    uncertainty_threshold = float(
        os.environ.get("PLANT_DOCTOR_UNCERTAINTY_THRESHOLD", "0.90")
    )
except ValueError:
    uncertainty_threshold = 0.90
if not 0.0 <= uncertainty_threshold <= 1.0:
    uncertainty_threshold = 0.90

catalog = load_catalog(APP_DIR)
doctor: Optional[PlantDoctorService]
model_load_error: Optional[str] = None
try:
    doctor = PlantDoctorService(APP_DIR, uncertainty_threshold=uncertainty_threshold)
except Exception as exc:  # keep the UI available for a safe degraded state
    doctor = None
    model_load_error = str(exc)
    logger.exception("Plant disease model failed to load")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES
app.config["JSON_SORT_KEYS"] = False
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "development-only-change-me")

farmer_service = FarmerService(mongo_store)
mongo_available = False
mongo_load_error: Optional[str] = None
mongo_initialized = False


def _refresh_mongo_status() -> bool:
    global mongo_available, mongo_load_error, mongo_initialized
    try:
        mongo_store.ping()
        if not mongo_initialized:
            mongo_store.ensure_indexes()
            farmer_service.sync_diseases(catalog)
            mongo_initialized = True
        mongo_available = True
        mongo_load_error = None
        return True
    except DatabaseUnavailableError as exc:
        mongo_available = False
        mongo_load_error = str(exc)
        return False


if not _refresh_mongo_status():
    logger.warning("MongoDB is unavailable; farmer APIs will return a safe 503: %s", mongo_load_error)


def _service_or_error() -> PlantDoctorService:
    if doctor is None:
        raise ModelUnavailableError("The plant disease model is unavailable.")
    return doctor


def _safe_error_message(error: Exception) -> str:
    if isinstance(error, (PlantDoctorError, ProfileRequiredError, DatabaseUnavailableError, ValidationError)):
        return str(error)
    return "We could not analyze that image right now. Please try again."


class ProfileRequiredError(PermissionError):
    """Raised when a Mongo-backed operation has no signed-in profile session."""


def _current_user_id() -> Optional[ObjectId]:
    raw_user_id = session.get("user_id")
    if not raw_user_id:
        return None
    try:
        user_id = parse_object_id(raw_user_id)
    except InvalidObjectIdError:
        session.pop("user_id", None)
        return None
    try:
        if farmer_service.get_profile(user_id) is None:
            session.pop("user_id", None)
            return None
    except DatabaseUnavailableError:
        return user_id
    return user_id


def _require_user() -> ObjectId:
    if not _refresh_mongo_status():
        raise DatabaseUnavailableError(mongo_load_error or "PlantCare MongoDB is unavailable.")
    user_id = _current_user_id()
    if user_id is None:
        raise ProfileRequiredError("Create or open a farmer profile before using this feature.")
    return user_id


def _form_options() -> Dict[str, Any]:
    crop_names = sorted({item["plant"] for item in catalog if item.get("plant")})
    return {
        "soil_types": [{"value": value, "description": SOIL_DESCRIPTIONS[value]} for value in SOIL_TYPES if value != "Loam"],
        "irrigation_methods": list(IRRIGATION_METHODS),
        "states": list(INDIAN_STATES),
        "growth_stages": list(GROWTH_STAGES),
        "supported_crops": crop_names,
        "themes": list(THEMES),
        "unit_preferences": list(UNIT_PREFERENCES),
    }


def _api_error(exc: Exception) -> Tuple[Any, int]:
    if isinstance(exc, ProfileRequiredError):
        return jsonify({"status": "error", "error": str(exc), "code": "profile_required"}), 401
    if isinstance(exc, (ValidationError, InvalidObjectIdError)):
        return jsonify({"status": "error", "error": str(exc)}), 400
    if isinstance(exc, NotFoundError):
        return jsonify({"status": "error", "error": str(exc)}), 404
    if isinstance(exc, ConflictError):
        return jsonify({"status": "error", "error": str(exc)}), 409
    if isinstance(exc, OwnershipError):
        return jsonify({"status": "error", "error": "You do not have access to this resource."}), 403
    if isinstance(exc, DatabaseUnavailableError):
        return jsonify({"status": "error", "error": str(exc), "code": "database_unavailable"}), 503
    logger.exception("Unhandled farmer API failure")
    return jsonify({"status": "error", "error": "The request could not be completed."}), 500


def _json_body() -> Dict[str, Any]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValidationError("Request body must be a JSON object.")
    return payload


def _api_route(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any):
        try:
            return jsonify({"status": "ok", "data": function(*args, **kwargs)})
        except Exception as exc:
            return _api_error(exc)

    return wrapped


def _predict_file() -> Dict[str, Any]:
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        raise InvalidImageError("Choose an image before analyzing.")
    data = upload.read()
    service = _service_or_error()
    result = service.predict_bytes(data)
    stored_filename = service.save_normalized_image(data)
    result["image_url"] = url_for("static", filename=f"uploads/{stored_filename}")
    result["stored_filename"] = stored_filename
    result["original_filename"] = upload.filename
    field_id = request.form.get("field_id")
    crop_cycle_id = request.form.get("crop_cycle_id")
    result["persistence"] = {"saved": False, "reason": "profile_required"}
    if not mongo_available:
        _refresh_mongo_status()
    if field_id or crop_cycle_id:
        user_id = _require_user()
        saved = farmer_service.save_prediction(user_id, result, field_id, crop_cycle_id)
        result["prediction_id"] = saved["_id"]
        result["persistence"] = {"saved": True, "reason": "mongodb", "prediction_id": saved["_id"]}
    elif mongo_available:
        user_id = _current_user_id()
        if user_id is not None:
            try:
                saved = farmer_service.save_prediction(user_id, result)
                result["prediction_id"] = saved["_id"]
                result["persistence"] = {"saved": True, "reason": "mongodb", "prediction_id": saved["_id"]}
            except DatabaseUnavailableError:
                result["persistence"] = {"saved": False, "reason": "database_unavailable"}
    return result


@app.context_processor
def inject_shell_context() -> Dict[str, Any]:
    profile = None
    if mongo_available:
        profile = _current_user_id()
        profile = farmer_service.get_profile(profile) if profile else None
    initials = ""
    if profile:
        initials = "".join(part[:1].upper() for part in str(profile.get("name", "")).split()[:2]) or "F"
    return {
        "model_version": "plant_disease_model_1_latest.pt",
        "model_available": doctor is not None,
        "model_load_error": model_load_error,
        "mongo_available": mongo_available,
        "mongo_load_error": mongo_load_error,
        "profile_available": profile is not None,
        "current_profile": profile,
        "profile_initials": initials,
        "form_options": _form_options(),
    }


@app.route("/")
def dashboard_page():
    return render_template("dashboard.html", active_page="dashboard")


@app.route("/plant-doctor")
@app.route("/index")
def plant_doctor_page():
    return render_template("plant_doctor.html", active_page="plant-doctor")


@app.route("/history")
def history_page():
    return render_template("history.html", active_page="history")


@app.route("/library")
def library_page():
    return render_template("library.html", active_page="library", catalog=catalog)


@app.route("/market")
def market_page():
    # Preserve the historic route while keeping one verified knowledge UI.
    return redirect(url_for("library_page"))


@app.route("/fields")
def fields_page():
    return render_template("fields.html", active_page="fields")


@app.route("/fields/add")
def add_field_page():
    return render_template("add_field.html", active_page="fields", form_options=_form_options())


@app.route("/fields/<field_id>")
def field_detail_page(field_id: str):
    return render_template("field_detail.html", active_page="fields", field_id=field_id)


@app.route("/fields/<field_id>/edit")
def edit_field_page(field_id: str):
    return render_template("add_field.html", active_page="fields", edit_field_id=field_id, form_options=_form_options())


@app.route("/crop-management")
def crop_management_page():
    return render_template("crop_management.html", active_page="crop-management", form_options=_form_options())


@app.route("/irrigation")
def irrigation_page():
    return render_template("irrigation.html", active_page="irrigation")


@app.route("/tasks")
def tasks_page():
    return render_template("tasks.html", active_page="tasks")


@app.route("/notifications")
def notifications_page():
    return render_template("notifications.html", active_page="notifications")


@app.route("/settings")
def settings_page():
    return render_template("settings.html", active_page="settings")


@app.route("/profile")
def profile_page():
    return render_template("profile.html", active_page="settings")


@app.route("/logout", methods=["POST", "GET"])
def logout():
    session.pop("user_id", None)
    if request.method == "POST" or request.accept_mimetypes.best == "application/json":
        return jsonify({"status": "ok", "data": {"logged_out": True}})
    return redirect(url_for("dashboard_page"))


@app.route("/contact")
def contact_page():
    return render_template("contact.html", active_page="contact")


@app.route("/mobile-device")
def mobile_device_page():
    return redirect(url_for("plant_doctor_page"))


@app.route("/api/health")
def health_api():
    _refresh_mongo_status()
    status = "ok" if doctor is not None else "degraded"
    return jsonify(
        {
            "status": status,
            "model_loaded": doctor is not None,
            "model_version": "plant_disease_model_1_latest.pt",
            "classes": 39,
            "error": model_load_error if doctor is None else None,
            "mongodb": {"available": mongo_available, "database": "plantcare_ai", "error": mongo_load_error},
        }
    )


@app.route("/api/profile", methods=["GET", "POST", "PATCH"])
def profile_api():
    try:
        if not _refresh_mongo_status():
            raise DatabaseUnavailableError(mongo_load_error or "PlantCare MongoDB is unavailable.")
        if request.method == "GET":
            user_id = _current_user_id()
            return jsonify({"status": "ok", "data": farmer_service.get_profile(user_id) if user_id else None})
        if request.method == "POST":
            profile = farmer_service.create_profile(_json_body())
            session["user_id"] = profile["_id"]
            return jsonify({"status": "ok", "data": profile}), 201
        profile = farmer_service.update_profile(_require_user(), _json_body())
        return jsonify({"status": "ok", "data": profile})
    except Exception as exc:
        return _api_error(exc)


@app.route("/api/field-options")
@_api_route
def field_options_api():
    user_id = _require_user()
    return {
        "fields": farmer_service.list_fields(user_id, status="active"),
        "crops": farmer_service.list_crops(user_id),
        **_form_options(),
    }


@app.route("/api/settings", methods=["GET", "PATCH"])
@_api_route
def settings_api():
    user_id = _require_user()
    if request.method == "PATCH":
        return farmer_service.update_settings(user_id, _json_body())
    return farmer_service.get_settings(user_id)


@app.route("/api/fields", methods=["GET", "POST"])
@_api_route
def fields_api():
    user_id = _require_user()
    if request.method == "POST":
        return farmer_service.create_field(user_id, _json_body())
    return farmer_service.list_fields(user_id, request.args.get("status"), request.args.get("q"))


@app.route("/api/fields/<field_id>", methods=["GET", "PATCH", "DELETE"])
def field_api(field_id: str):
    try:
        user_id = _require_user()
        if request.method == "GET":
            return jsonify({"status": "ok", "data": farmer_service.get_field(user_id, field_id)})
        if request.method == "PATCH":
            return jsonify({"status": "ok", "data": farmer_service.update_field(user_id, field_id, _json_body())})
        return jsonify({"status": "ok", "data": farmer_service.delete_field(user_id, field_id)})
    except Exception as exc:
        return _api_error(exc)


@app.route("/api/fields/<field_id>/overview")
@_api_route
def field_overview_api(field_id: str):
    user_id = _require_user()
    farmer_service.get_field(user_id, field_id)
    return {
        "field": farmer_service.get_field(user_id, field_id),
        "crops": farmer_service.list_crops(user_id, field_id),
        "tasks": farmer_service.list_tasks(user_id),
        "irrigation": farmer_service.list_irrigation(user_id, field_id),
        "health": farmer_service.list_health(user_id, field_id),
        "predictions": farmer_service.list_predictions(user_id, field_id),
    }


@app.route("/api/crops", methods=["GET", "POST"])
@_api_route
def crops_api():
    user_id = _require_user()
    if request.method == "POST":
        return farmer_service.create_crop(user_id, _json_body())
    return farmer_service.list_crops(user_id, request.args.get("field_id"))


@app.route("/api/crops/<crop_id>", methods=["GET", "PATCH", "DELETE"])
def crop_api(crop_id: str):
    try:
        user_id = _require_user()
        if request.method == "GET":
            return jsonify({"status": "ok", "data": farmer_service.get_crop(user_id, crop_id)})
        if request.method == "PATCH":
            return jsonify({"status": "ok", "data": farmer_service.update_crop(user_id, crop_id, _json_body())})
        farmer_service.delete_crop(user_id, crop_id)
        return jsonify({"status": "ok", "data": {"deleted": True}})
    except Exception as exc:
        return _api_error(exc)


@app.route("/api/tasks", methods=["GET", "POST"])
@_api_route
def tasks_api():
    user_id = _require_user()
    if request.method == "POST":
        return farmer_service.create_task(user_id, _json_body())
    return farmer_service.list_tasks(user_id, request.args.get("status"))


@app.route("/api/tasks/<task_id>", methods=["PATCH"])
@_api_route
def task_update_api(task_id: str):
    return farmer_service.update_task(_require_user(), task_id, _json_body())


@app.route("/api/tasks/<task_id>/complete", methods=["POST"])
@_api_route
def task_complete_api(task_id: str):
    return farmer_service.update_task(_require_user(), task_id, {"status": "completed"})


@app.route("/api/irrigation", methods=["GET", "POST"])
@_api_route
def irrigation_api():
    user_id = _require_user()
    if request.method == "POST":
        return farmer_service.create_irrigation(user_id, _json_body())
    return farmer_service.list_irrigation(user_id, request.args.get("field_id"), request.args.get("crop_cycle_id"))


@app.route("/api/health-logs", methods=["GET", "POST"])
@_api_route
def health_logs_api():
    user_id = _require_user()
    if request.method == "POST":
        return farmer_service.create_health(user_id, _json_body())
    return farmer_service.list_health(user_id, request.args.get("field_id"), request.args.get("crop_cycle_id"))


@app.route("/api/predictions")
@_api_route
def predictions_api():
    return farmer_service.list_predictions(_require_user(), request.args.get("field_id"), request.args.get("crop_cycle_id"))


@app.route("/api/predictions/<prediction_id>")
@_api_route
def prediction_detail_api(prediction_id: str):
    return farmer_service.get_prediction(_require_user(), prediction_id)


@app.route("/api/notifications")
@_api_route
def notifications_api():
    return farmer_service.list_notifications(_require_user())


@app.route("/api/notifications/<notification_id>/read", methods=["POST"])
@_api_route
def notification_read_api(notification_id: str):
    return farmer_service.mark_notification_read(_require_user(), notification_id)


@app.route("/api/notifications/read-all", methods=["POST"])
@_api_route
def notifications_read_all_api():
    return farmer_service.mark_all_notifications_read(_require_user())


@app.route("/api/dashboard")
@_api_route
def dashboard_api():
    return farmer_service.dashboard(_require_user())


@app.route("/api/diseases")
@_api_route
def diseases_api():
    if mongo_available:
        return [serialize(item) for item in mongo_store.collection("diseases").find({}).sort("class_key", 1)]
    return catalog


@app.route("/api/predict", methods=["POST"])
def predict_api():
    try:
        return jsonify(_predict_file())
    except (InvalidImageError, ModelUnavailableError, ProfileRequiredError, DatabaseUnavailableError, ValidationError) as exc:
        status_code = 503 if isinstance(exc, (ModelUnavailableError, DatabaseUnavailableError)) else 400
        return jsonify({"error": _safe_error_message(exc), "status": "error"}), status_code
    except Exception:
        logger.exception("Unexpected prediction failure")
        return jsonify(
            {"error": "We could not analyze that image right now. Please try again.", "status": "error"}
        ), 500


@app.route("/submit", methods=["GET", "POST"])
def legacy_submit():
    if request.method == "GET":
        return redirect(url_for("plant_doctor_page"))
    try:
        result = _predict_file()
        return render_template("result.html", result=result, active_page="plant-doctor")
    except (InvalidImageError, ModelUnavailableError, ProfileRequiredError, DatabaseUnavailableError, ValidationError) as exc:
        return render_template(
            "plant_doctor.html",
            active_page="plant-doctor",
            initial_error=_safe_error_message(exc),
        ), 503 if isinstance(exc, (ModelUnavailableError, DatabaseUnavailableError)) else 400
    except Exception:
        logger.exception("Unexpected legacy submission failure")
        return render_template(
            "plant_doctor.html",
            active_page="plant-doctor",
            initial_error="We could not analyze that image right now. Please try again.",
        ), 500


@app.errorhandler(413)
def request_too_large(_error):
    message = "Images must be 8 MB or smaller."
    if request.path.startswith("/api/"):
        return jsonify({"error": message, "status": "error"}), 413
    return render_template("plant_doctor.html", active_page="plant-doctor", initial_error=message), 413


@app.errorhandler(404)
def not_found(_error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "The requested endpoint was not found.", "status": "error"}), 404
    return render_template("not_found.html", active_page="dashboard"), 404


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1")
