"""Ownership-aware farmer, field and crop services."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

from bson import ObjectId
from pymongo.errors import DuplicateKeyError, PyMongoError

from mongo import (
    DatabaseUnavailableError,
    InvalidObjectIdError,
    MongoStore,
    parse_object_id,
    serialize,
    store as default_store,
    utc_now,
)


class ValidationError(ValueError):
    pass


class NotFoundError(LookupError):
    pass


class OwnershipError(PermissionError):
    pass


class ConflictError(RuntimeError):
    pass


logger = logging.getLogger("plant-doctor.farmer")


AREA_UNITS = {"acres", "hectares", "square_metres"}
FIELD_STATUSES = {"active", "archived"}
CROP_STATUSES = {"planned", "active", "completed", "cancelled"}
TASK_STATUSES = {"pending", "in_progress", "completed", "cancelled"}
TASK_PRIORITIES = {"low", "medium", "high"}
SOIL_TYPES = ("Sandy", "Clay", "Loamy", "Silt", "Sandy loam", "Clay loam", "Other", "Loam")
SOIL_DESCRIPTIONS = {
    "Sandy": "Light soil that drains quickly.",
    "Clay": "Dense soil that holds more moisture.",
    "Loamy": "Balanced soil with good drainage and nutrient holding.",
    "Silt": "Fine soil that can hold moisture and nutrients.",
    "Sandy loam": "A lighter blend with improved moisture retention.",
    "Clay loam": "A balanced blend with stronger moisture retention.",
    "Other": "Choose this when the soil type is not listed.",
    "Loam": "Legacy value retained for existing field records.",
}
IRRIGATION_METHODS = ("Drip", "Sprinkler", "Surface irrigation", "Flood irrigation", "Rainfed", "Other")
GROWTH_STAGES = ("Seedling", "Vegetative", "Flowering", "Fruiting", "Maturity", "Other")
THEMES = ("light", "dark", "system")
UNIT_PREFERENCES = ("metric", "imperial")
INDIAN_STATES = (
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands", "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Jammu and Kashmir", "Ladakh",
    "Lakshadweep", "Puducherry",
)


def _text(data: Dict[str, Any], key: str, *, required: bool = False, maximum: int = 200) -> Optional[str]:
    value = data.get(key)
    if value is None:
        if required:
            raise ValidationError(f"{key} is required.")
        return None
    value = str(value).strip()
    if required and not value:
        raise ValidationError(f"{key} is required.")
    if len(value) > maximum:
        raise ValidationError(f"{key} is too long.")
    return value or None


def _enum(data: Dict[str, Any], key: str, values: set, *, required: bool = False) -> Optional[str]:
    value = _text(data, key, required=required, maximum=60)
    if value is not None:
        value = value.lower()
        if value not in values:
            raise ValidationError(f"{key} must be one of: {', '.join(sorted(values))}.")
    return value


def _choice(data: Dict[str, Any], key: str, values: Iterable[str], *, required: bool = False) -> Optional[str]:
    value = _text(data, key, required=required, maximum=100)
    if value is None:
        return None
    choices = {item.casefold(): item for item in values}
    selected = choices.get(value.casefold())
    if selected is None:
        raise ValidationError(f"{key} must be one of the available choices.")
    return selected


def _irrigation_choice(data: Dict[str, Any]) -> Optional[str]:
    selected = _choice(data, "irrigation_method", IRRIGATION_METHODS)
    if selected == "Other":
        return _text(data, "irrigation_method_other", maximum=80) or "Other"
    return selected


def _number(data: Dict[str, Any], key: str, *, required: bool = False, positive: bool = False) -> Optional[float]:
    value = data.get(key)
    if value in (None, ""):
        if required:
            raise ValidationError(f"{key} is required.")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{key} must be a number.") from exc
    if positive and number <= 0:
        raise ValidationError(f"{key} must be positive.")
    if number < 0:
        raise ValidationError(f"{key} cannot be negative.")
    return number


def _date(value: Any, key: str, *, required: bool = False) -> Optional[datetime]:
    if value in (None, ""):
        if required:
            raise ValidationError(f"{key} is required.")
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    try:
        parsed = datetime.strptime(str(value), "%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{key} must use YYYY-MM-DD format.") from exc
    return parsed.replace(tzinfo=timezone.utc)


def _datetime(value: Any, key: str, *, required: bool = False) -> Optional[datetime]:
    if value in (None, ""):
        if required:
            raise ValidationError(f"{key} is required.")
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{key} must be a valid ISO date/time.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _object_id(value: Any, key: str, *, required: bool = False) -> Optional[ObjectId]:
    if value in (None, ""):
        if required:
            raise ValidationError(f"{key} is required.")
        return None
    try:
        return parse_object_id(value)
    except InvalidObjectIdError as exc:
        raise ValidationError(f"{key} is invalid.") from exc


def _location(data: Dict[str, Any]) -> Dict[str, str]:
    raw = data.get("location") or {}
    if not isinstance(raw, dict):
        raise ValidationError("location must be an object.")
    output = {}
    for key in ("village", "district", "state", "country"):
        value = raw.get(key)
        if value not in (None, ""):
            value = str(value).strip()
            if len(value) > 100:
                raise ValidationError(f"location.{key} is too long.")
            output[key] = value
    return output


def _notes(data: Dict[str, Any]) -> Optional[str]:
    return _text(data, "notes", maximum=2000)


class FarmerService:
    def __init__(self, mongo_store: MongoStore = default_store):
        self.mongo = mongo_store

    @staticmethod
    def _public(document: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        return serialize(document) if document else None

    def _owned(self, collection: str, user_id: ObjectId, resource_id: Any, label: str) -> Dict[str, Any]:
        object_id = _object_id(resource_id, label, required=True)
        document = self.mongo.collection(collection).find_one({"_id": object_id, "user_id": user_id})
        if not document:
            raise NotFoundError(f"{label} was not found.")
        return document

    # Profile / development-session identity
    def create_profile(self, data: Dict[str, Any]) -> Dict[str, Any]:
        name = _text(data, "name", required=True, maximum=120)
        email = _text(data, "email", required=True, maximum=180)
        email = email.lower()
        if "@" not in email:
            raise ValidationError("email must be valid.")
        now = utc_now()
        document = {
            "name": name,
            "email": email,
            "phone": _text(data, "phone", maximum=40),
            "farm_name": _text(data, "farm_name", maximum=160),
            "location": _text(data, "location", maximum=160),
            "preferences": {
                "theme": "system",
                "notifications_enabled": True,
                "unit_preference": "metric",
            },
            "created_at": now,
            "updated_at": now,
        }
        try:
            result = self.mongo.collection("users").insert_one(document)
        except DuplicateKeyError as exc:
            raise ConflictError("A profile with this email already exists.") from exc
        document["_id"] = result.inserted_id
        return self._public(document)

    def get_profile(self, user_id: ObjectId) -> Optional[Dict[str, Any]]:
        return self._public(self.mongo.collection("users").find_one({"_id": user_id}))

    def update_profile(self, user_id: ObjectId, data: Dict[str, Any]) -> Dict[str, Any]:
        current = self.get_profile(user_id)
        if not current:
            raise NotFoundError("Profile was not found.")
        updates: Dict[str, Any] = {"updated_at": utc_now()}
        for key, maximum in (("name", 120), ("phone", 40), ("farm_name", 160), ("location", 160)):
            if key in data:
                updates[key] = _text(data, key, maximum=maximum)
        if "email" in data:
            email = _text(data, "email", required=True, maximum=180).lower()
            if "@" not in email:
                raise ValidationError("email must be valid.")
            updates["email"] = email
        try:
            self.mongo.collection("users").update_one({"_id": user_id}, {"$set": updates})
        except DuplicateKeyError as exc:
            raise ConflictError("A profile with this email already exists.") from exc
        return self._public(self.mongo.collection("users").find_one({"_id": user_id}))

    def get_settings(self, user_id: ObjectId) -> Dict[str, Any]:
        profile = self.get_profile(user_id)
        if not profile:
            raise NotFoundError("Profile was not found.")
        preferences = profile.get("preferences") or {}
        return {
            "theme": preferences.get("theme", "system"),
            "notifications_enabled": bool(preferences.get("notifications_enabled", True)),
            "unit_preference": preferences.get("unit_preference", "metric"),
        }

    def update_settings(self, user_id: ObjectId, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.get_profile(user_id):
            raise NotFoundError("Profile was not found.")
        current = self.get_settings(user_id)
        updates = dict(current)
        if "theme" in data:
            updates["theme"] = _choice(data, "theme", THEMES, required=True)
        if "unit_preference" in data:
            updates["unit_preference"] = _choice(data, "unit_preference", UNIT_PREFERENCES, required=True)
        if "notifications_enabled" in data:
            if not isinstance(data["notifications_enabled"], bool):
                raise ValidationError("notifications_enabled must be true or false.")
            updates["notifications_enabled"] = data["notifications_enabled"]
        self.mongo.collection("users").update_one(
            {"_id": user_id},
            {"$set": {"preferences": updates, "updated_at": utc_now()}},
        )
        return updates

    # Fields
    def list_fields(self, user_id: ObjectId, status: Optional[str] = None, query: Optional[str] = None) -> List[Dict[str, Any]]:
        filters: Dict[str, Any] = {"user_id": user_id}
        if status:
            if status not in FIELD_STATUSES:
                raise ValidationError("status is invalid.")
            filters["status"] = status
        if query:
            filters["field_name"] = {"$regex": str(query)[:80], "$options": "i"}
        documents = self.mongo.collection("fields").find(filters).sort("updated_at", -1)
        return [self._public(document) for document in documents]

    def create_field(self, user_id: ObjectId, data: Dict[str, Any]) -> Dict[str, Any]:
        field_name = _text(data, "field_name", required=True, maximum=120)
        area = _number(data, "area", required=True, positive=True)
        area_unit = _enum(data, "area_unit", AREA_UNITS, required=True)
        status = _enum(data, "status", FIELD_STATUSES) or "active"
        now = utc_now()
        document = {
            "user_id": user_id,
            "field_name": field_name,
            "location": _location(data),
            "area": area,
            "area_unit": area_unit,
            "soil_type": _choice(data, "soil_type", SOIL_TYPES),
            "irrigation_method": _irrigation_choice(data),
            "notes": _notes(data),
            "status": status,
            "created_at": now,
            "updated_at": now,
        }
        result = self.mongo.collection("fields").insert_one(document)
        document["_id"] = result.inserted_id
        return self._public(document)

    def get_field(self, user_id: ObjectId, field_id: Any) -> Dict[str, Any]:
        return self._public(self._owned("fields", user_id, field_id, "field_id"))

    def update_field(self, user_id: ObjectId, field_id: Any, data: Dict[str, Any]) -> Dict[str, Any]:
        self._owned("fields", user_id, field_id, "field_id")
        updates: Dict[str, Any] = {"updated_at": utc_now()}
        if "field_name" in data:
            updates["field_name"] = _text(data, "field_name", required=True, maximum=120)
        if "area" in data:
            updates["area"] = _number(data, "area", required=True, positive=True)
        if "area_unit" in data:
            updates["area_unit"] = _enum(data, "area_unit", AREA_UNITS, required=True)
        if "location" in data:
            updates["location"] = _location(data)
        if "soil_type" in data:
            updates["soil_type"] = _choice(data, "soil_type", SOIL_TYPES)
        if "irrigation_method" in data:
            updates["irrigation_method"] = _irrigation_choice(data)
        if "notes" in data:
            updates["notes"] = _notes(data)
        if "status" in data:
            updates["status"] = _enum(data, "status", FIELD_STATUSES, required=True)
        object_id = _object_id(field_id, "field_id", required=True)
        self.mongo.collection("fields").update_one({"_id": object_id, "user_id": user_id}, {"$set": updates})
        return self._public(self.mongo.collection("fields").find_one({"_id": object_id, "user_id": user_id}))

    def delete_field(self, user_id: ObjectId, field_id: Any) -> Dict[str, bool]:
        field = self._owned("fields", user_id, field_id, "field_id")
        field_object_id = field["_id"]
        dependent_collections = ("crop_cycles", "care_tasks", "irrigation_logs", "plant_health_logs", "predictions")
        if any(self.mongo.collection(name).find_one({"user_id": user_id, "field_id": field_object_id}) for name in dependent_collections):
            self.mongo.collection("fields").update_one(
                {"_id": field_object_id, "user_id": user_id},
                {"$set": {"status": "archived", "archived_at": utc_now(), "updated_at": utc_now()}},
            )
            return {"deleted": False, "archived": True}
        self.mongo.collection("fields").delete_one({"_id": field_object_id, "user_id": user_id})
        return {"deleted": True, "archived": False}

    # Crop cycles
    def list_crops(self, user_id: ObjectId, field_id: Optional[Any] = None) -> List[Dict[str, Any]]:
        filters: Dict[str, Any] = {"user_id": user_id}
        if field_id:
            filters["field_id"] = _object_id(field_id, "field_id", required=True)
        documents = self.mongo.collection("crop_cycles").find(filters).sort("updated_at", -1)
        return [self._public(document) for document in documents]

    def create_crop(self, user_id: ObjectId, data: Dict[str, Any]) -> Dict[str, Any]:
        field_id = _object_id(data.get("field_id"), "field_id", required=True)
        self._owned("fields", user_id, field_id, "field_id")
        planting_date = _date(data.get("planting_date"), "planting_date", required=True)
        harvest_date = _date(data.get("estimated_harvest_date"), "estimated_harvest_date")
        if harvest_date and harvest_date < planting_date:
            raise ValidationError("estimated_harvest_date cannot be before planting_date.")
        now = utc_now()
        document = {
            "user_id": user_id,
            "field_id": field_id,
            "crop_name": _text(data, "crop_name", required=True, maximum=100),
            "variety": _text(data, "variety", maximum=100),
            "planting_date": planting_date,
            "estimated_harvest_date": harvest_date,
            "current_growth_stage": _choice(data, "current_growth_stage", GROWTH_STAGES),
            "soil_type": _text(data, "soil_type", maximum=80),
            "irrigation_method": _text(data, "irrigation_method", maximum=80),
            "status": _enum(data, "status", CROP_STATUSES) or "active",
            "notes": _notes(data),
            "created_at": now,
            "updated_at": now,
        }
        result = self.mongo.collection("crop_cycles").insert_one(document)
        document["_id"] = result.inserted_id
        self._safe_evaluate_rules(user_id, "crop_created", field_id=field_id, crop_id=result.inserted_id)
        return self._public(document)

    def get_crop(self, user_id: ObjectId, crop_id: Any) -> Dict[str, Any]:
        return self._public(self._owned("crop_cycles", user_id, crop_id, "crop_cycle_id"))

    def update_crop(self, user_id: ObjectId, crop_id: Any, data: Dict[str, Any]) -> Dict[str, Any]:
        current = self._owned("crop_cycles", user_id, crop_id, "crop_cycle_id")
        updates: Dict[str, Any] = {"updated_at": utc_now()}
        for key, maximum in (("crop_name", 100), ("variety", 100)):
            if key in data:
                updates[key] = _text(data, key, required=(key == "crop_name"), maximum=maximum)
        if "current_growth_stage" in data:
            updates["current_growth_stage"] = _choice(data, "current_growth_stage", GROWTH_STAGES)
        if "soil_type" in data:
            updates["soil_type"] = _choice(data, "soil_type", SOIL_TYPES)
        if "irrigation_method" in data:
            updates["irrigation_method"] = _irrigation_choice(data)
        if "field_id" in data:
            field_id = _object_id(data.get("field_id"), "field_id", required=True)
            self._owned("fields", user_id, field_id, "field_id")
            updates["field_id"] = field_id
        for key in ("planting_date", "estimated_harvest_date"):
            if key in data:
                updates[key] = _date(data.get(key), key, required=(key == "planting_date"))
        planting_date = updates.get("planting_date", current.get("planting_date"))
        harvest_date = updates.get("estimated_harvest_date", current.get("estimated_harvest_date"))
        if harvest_date and planting_date and harvest_date < planting_date:
            raise ValidationError("estimated_harvest_date cannot be before planting_date.")
        if "status" in data:
            updates["status"] = _enum(data, "status", CROP_STATUSES, required=True)
        if "notes" in data:
            updates["notes"] = _notes(data)
        object_id = _object_id(crop_id, "crop_cycle_id", required=True)
        self.mongo.collection("crop_cycles").update_one({"_id": object_id, "user_id": user_id}, {"$set": updates})
        self._safe_evaluate_rules(user_id, "crop_updated", field_id=current["field_id"], crop_id=object_id)
        return self._public(self.mongo.collection("crop_cycles").find_one({"_id": object_id, "user_id": user_id}))

    def delete_crop(self, user_id: ObjectId, crop_id: Any) -> None:
        crop = self._owned("crop_cycles", user_id, crop_id, "crop_cycle_id")
        crop_id = crop["_id"]
        for name in ("care_tasks", "irrigation_logs", "plant_health_logs", "predictions"):
            if self.mongo.collection(name).find_one({"user_id": user_id, "crop_cycle_id": crop_id}):
                raise ConflictError("This crop cycle has related records and cannot be deleted.")
        self.mongo.collection("crop_cycles").delete_one({"_id": crop_id, "user_id": user_id})

    # Tasks
    def list_tasks(self, user_id: ObjectId, status: Optional[str] = None) -> List[Dict[str, Any]]:
        filters: Dict[str, Any] = {"user_id": user_id}
        if status:
            if status not in TASK_STATUSES:
                raise ValidationError("status is invalid.")
            filters["status"] = status
        return [self._public(item) for item in self.mongo.collection("care_tasks").find(filters).sort("due_date", 1).limit(200)]

    def create_task(
        self,
        user_id: ObjectId,
        data: Dict[str, Any],
        *,
        generated: bool = False,
        source_rule_id: Any = None,
        generation_reason: Optional[str] = None,
        dedup_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        field_id = _object_id(data.get("field_id"), "field_id")
        crop_id = _object_id(data.get("crop_cycle_id"), "crop_cycle_id")
        if field_id:
            self._owned("fields", user_id, field_id, "field_id")
        if crop_id:
            crop = self._owned("crop_cycles", user_id, crop_id, "crop_cycle_id")
            if field_id and crop["field_id"] != field_id:
                raise ValidationError("crop_cycle_id does not belong to field_id.")
            field_id = field_id or crop["field_id"]
        now = utc_now()
        document = {
            "user_id": user_id,
            "field_id": field_id,
            "crop_cycle_id": crop_id,
            "task_type": _text(data, "task_type", maximum=80),
            "title": _text(data, "title", required=True, maximum=160),
            "description": _text(data, "description", maximum=2000),
            "due_date": _date(data.get("due_date"), "due_date"),
            "priority": _enum(data, "priority", TASK_PRIORITIES) or "medium",
            "status": _enum(data, "status", TASK_STATUSES) or "pending",
            "source_rule": _text(data, "source_rule", maximum=120),
            "source_rule_id": serialize(source_rule_id) if source_rule_id is not None else None,
            "generation_reason": generation_reason,
            "generated": bool(generated),
            "dedup_key": dedup_key,
            "generated_at": now if generated else None,
            "notes": _notes(data),
            "created_at": now,
            "completed_at": None,
        }
        try:
            result = self.mongo.collection("care_tasks").insert_one(document)
        except DuplicateKeyError:
            if dedup_key:
                existing = self.mongo.collection("care_tasks").find_one({"user_id": user_id, "dedup_key": dedup_key})
                if existing:
                    return self._public(existing)
            raise
        document["_id"] = result.inserted_id
        self.create_notification(
            user_id,
            "Recommended task created" if generated else "Task created",
            generation_reason or document["title"],
            "task",
            result.inserted_id,
            field_id=field_id,
            crop_id=crop_id,
        )
        return self._public(document)

    def evaluate_rules(
        self,
        user_id: ObjectId,
        trigger: str,
        *,
        field_id: Any = None,
        crop_id: Any = None,
        prediction: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Apply only explicitly stored, enabled agricultural rules.

        The repository currently contains no seeded rules, so this is a safe
        no-op until a verified rule source is loaded into `agricultural_rules`.
        It deliberately does not invent watering, pesticide or fertilizer advice.
        """

        field = self._owned("fields", user_id, field_id, "field_id") if field_id else None
        crop = self._owned("crop_cycles", user_id, crop_id, "crop_cycle_id") if crop_id else None
        filters = {"enabled": True, "triggers": trigger}
        generated: List[Dict[str, Any]] = []
        for rule in self.mongo.collection("agricultural_rules").find(filters):
            conditions = rule.get("conditions") or {}
            crop_names = {str(value).casefold() for value in conditions.get("crop_names", [])}
            stages = {str(value).casefold() for value in conditions.get("growth_stages", [])}
            soil_types = {str(value).casefold() for value in conditions.get("soil_types", [])}
            methods = {str(value).casefold() for value in conditions.get("irrigation_methods", [])}
            disease_classes = {str(value).casefold() for value in conditions.get("disease_classes", [])}
            if crop_names and (not crop or str(crop.get("crop_name", "")).casefold() not in crop_names):
                continue
            if stages and (not crop or str(crop.get("current_growth_stage", "")).casefold() not in stages):
                continue
            if soil_types and (not field or str(field.get("soil_type", "")).casefold() not in soil_types):
                continue
            if methods and (not field or str(field.get("irrigation_method", "")).casefold() not in methods):
                continue
            prediction_class = str((prediction or {}).get("predicted_class") or "").casefold()
            if disease_classes and prediction_class not in disease_classes:
                continue
            task_data = rule.get("task") or {}
            title = _text(task_data, "title", required=True, maximum=160)
            days_after = task_data.get("days_after_trigger", 0)
            try:
                due_date = utc_now() + timedelta(days=max(0, int(days_after)))
            except (TypeError, ValueError):
                raise ValidationError("Agricultural rule days_after_trigger must be a whole number.")
            rule_id = rule.get("_id") or rule.get("rule_id")
            dedup_key = f"{user_id}:{rule_id}:{field_id or ''}:{crop_id or ''}:{prediction_class}"
            generated.append(self.create_task(
                user_id,
                {
                    "field_id": field_id,
                    "crop_cycle_id": crop_id,
                    "task_type": task_data.get("task_type") or "Field care",
                    "title": title,
                    "description": task_data.get("description"),
                    "due_date": due_date.strftime("%Y-%m-%d"),
                    "priority": task_data.get("priority") or "medium",
                    "source_rule": str(rule.get("name") or rule.get("rule_id") or "Verified agricultural rule"),
                },
                generated=True,
                source_rule_id=rule_id,
                generation_reason=str(rule.get("reason") or "Generated from a verified agricultural rule."),
                dedup_key=dedup_key,
            ))
        return generated

    def _safe_evaluate_rules(self, *args: Any, **kwargs: Any) -> None:
        try:
            self.evaluate_rules(*args, **kwargs)
        except (DatabaseUnavailableError, PyMongoError, ValidationError) as exc:
            logger.warning("Rule evaluation skipped: %s", exc)

    def update_task(self, user_id: ObjectId, task_id: Any, data: Dict[str, Any]) -> Dict[str, Any]:
        self._owned("care_tasks", user_id, task_id, "task_id")
        updates: Dict[str, Any] = {}
        for key, maximum in (("task_type", 80), ("title", 160), ("description", 2000), ("source_rule", 120), ("notes", 2000)):
            if key in data:
                updates[key] = _text(data, key, required=(key == "title"), maximum=maximum)
        if "due_date" in data:
            updates["due_date"] = _date(data.get("due_date"), "due_date")
        if "priority" in data:
            updates["priority"] = _enum(data, "priority", TASK_PRIORITIES, required=True)
        if "status" in data:
            status = _enum(data, "status", TASK_STATUSES, required=True)
            updates["status"] = status
            updates["completed_at"] = utc_now() if status == "completed" else None
        if not updates:
            raise ValidationError("No task changes were provided.")
        object_id = _object_id(task_id, "task_id", required=True)
        self.mongo.collection("care_tasks").update_one({"_id": object_id, "user_id": user_id}, {"$set": updates})
        task = self.mongo.collection("care_tasks").find_one({"_id": object_id, "user_id": user_id})
        if updates.get("status") == "completed":
            self.create_notification(user_id, "Task completed", task["title"], "task", object_id, field_id=task.get("field_id"), crop_id=task.get("crop_cycle_id"))
        return self._public(task)

    # Irrigation and health
    def list_irrigation(self, user_id: ObjectId, field_id: Optional[Any] = None, crop_id: Optional[Any] = None) -> List[Dict[str, Any]]:
        filters: Dict[str, Any] = {"user_id": user_id}
        if field_id:
            filters["field_id"] = _object_id(field_id, "field_id", required=True)
        if crop_id:
            filters["crop_cycle_id"] = _object_id(crop_id, "crop_cycle_id", required=True)
        return [self._public(item) for item in self.mongo.collection("irrigation_logs").find(filters).sort("recorded_at", -1).limit(200)]

    def create_irrigation(self, user_id: ObjectId, data: Dict[str, Any]) -> Dict[str, Any]:
        field_id = _object_id(data.get("field_id"), "field_id", required=True)
        self._owned("fields", user_id, field_id, "field_id")
        crop_id = _object_id(data.get("crop_cycle_id"), "crop_cycle_id")
        if crop_id:
            crop = self._owned("crop_cycles", user_id, crop_id, "crop_cycle_id")
            if crop["field_id"] != field_id:
                raise ValidationError("crop_cycle_id does not belong to field_id.")
        document = {
            "user_id": user_id,
            "field_id": field_id,
            "crop_cycle_id": crop_id,
            "recorded_at": _datetime(data.get("recorded_at"), "recorded_at") or utc_now(),
            "irrigation_method": _irrigation_choice(data),
            "duration_minutes": _number(data, "duration_minutes", positive=False),
            "notes": _notes(data),
            "created_at": utc_now(),
            "recorded_by": user_id,
        }
        result = self.mongo.collection("irrigation_logs").insert_one(document)
        document["_id"] = result.inserted_id
        self.create_notification(user_id, "Irrigation recorded", "A new irrigation record was added.", "irrigation", result.inserted_id, field_id=field_id, crop_id=crop_id)
        return self._public(document)

    def list_health(self, user_id: ObjectId, field_id: Optional[Any] = None, crop_id: Optional[Any] = None) -> List[Dict[str, Any]]:
        filters: Dict[str, Any] = {"user_id": user_id}
        if field_id:
            filters["field_id"] = _object_id(field_id, "field_id", required=True)
        if crop_id:
            filters["crop_cycle_id"] = _object_id(crop_id, "crop_cycle_id", required=True)
        return [self._public(item) for item in self.mongo.collection("plant_health_logs").find(filters).sort("observation_date", -1).limit(200)]

    def create_health(self, user_id: ObjectId, data: Dict[str, Any]) -> Dict[str, Any]:
        field_id = _object_id(data.get("field_id"), "field_id", required=True)
        self._owned("fields", user_id, field_id, "field_id")
        crop_id = _object_id(data.get("crop_cycle_id"), "crop_cycle_id")
        if crop_id:
            crop = self._owned("crop_cycles", user_id, crop_id, "crop_cycle_id")
            if crop["field_id"] != field_id:
                raise ValidationError("crop_cycle_id does not belong to field_id.")
        document = {
            "user_id": user_id,
            "field_id": field_id,
            "crop_cycle_id": crop_id,
            "observation_date": _date(data.get("observation_date"), "observation_date", required=True),
            "health_status": _text(data, "health_status", required=True, maximum=80),
            "symptoms_notes": _text(data, "symptoms_notes", maximum=2000),
            "image_reference": _text(data, "image_reference", maximum=500),
            "created_at": utc_now(),
        }
        result = self.mongo.collection("plant_health_logs").insert_one(document)
        document["_id"] = result.inserted_id
        self.create_notification(user_id, "Plant health recorded", "A new plant health observation was added.", "health", result.inserted_id, field_id=field_id, crop_id=crop_id)
        return self._public(document)

    # Predictions
    def save_prediction(self, user_id: ObjectId, result: Dict[str, Any], field_id: Any = None, crop_id: Any = None) -> Dict[str, Any]:
        field_object_id = _object_id(field_id, "field_id")
        crop_object_id = _object_id(crop_id, "crop_cycle_id")
        if crop_object_id:
            crop = self._owned("crop_cycles", user_id, crop_object_id, "crop_cycle_id")
            if field_object_id and crop["field_id"] != field_object_id:
                raise ValidationError("crop_cycle_id does not belong to field_id.")
            field_object_id = field_object_id or crop["field_id"]
        if field_object_id:
            self._owned("fields", user_id, field_object_id, "field_id")
        primary = result["prediction"]
        prediction = {
            "user_id": user_id,
            "field_id": field_object_id,
            "crop_cycle_id": crop_object_id,
            "image_reference": result.get("image_url"),
            "stored_filename": result.get("stored_filename"),
            "predicted_class": primary.get("class_key"),
            "predicted_plant": primary.get("plant"),
            "predicted_condition": primary.get("condition"),
            "model_probability": primary.get("probability"),
            "top_predictions": [
                {key: item.get(key) for key in ("index", "class_key", "plant", "condition", "display_name", "probability")}
                for item in result.get("top_predictions", [])
            ],
            "model_version": result.get("model_version"),
            "created_at": utc_now(),
        }
        inserted = self.mongo.collection("predictions").insert_one(prediction)
        prediction["_id"] = inserted.inserted_id
        self._safe_evaluate_rules(
            user_id,
            "prediction_saved",
            field_id=field_object_id,
            crop_id=crop_object_id,
            prediction=prediction,
        )
        return self._public(prediction)

    def list_predictions(self, user_id: ObjectId, field_id: Optional[Any] = None, crop_id: Optional[Any] = None) -> List[Dict[str, Any]]:
        filters: Dict[str, Any] = {"user_id": user_id}
        if field_id:
            filters["field_id"] = _object_id(field_id, "field_id", required=True)
        if crop_id:
            filters["crop_cycle_id"] = _object_id(crop_id, "crop_cycle_id", required=True)
        return [self._public(item) for item in self.mongo.collection("predictions").find(filters).sort("created_at", -1).limit(200)]

    def get_prediction(self, user_id: ObjectId, prediction_id: Any) -> Dict[str, Any]:
        return self._public(self._owned("predictions", user_id, prediction_id, "prediction_id"))

    # Notifications and dashboard
    def create_notification(
        self,
        user_id: ObjectId,
        title: str,
        message: str,
        notification_type: str,
        related_id: ObjectId,
        *,
        field_id: Optional[ObjectId] = None,
        crop_id: Optional[ObjectId] = None,
    ) -> None:
        try:
            if not self.get_settings(user_id).get("notifications_enabled", True):
                return
            self.mongo.collection("notifications").insert_one(
                {
                    "user_id": user_id,
                    "title": title[:160],
                    "message": message[:1000],
                    "type": notification_type[:60],
                    "related_entity": notification_type,
                    "related_entity_id": related_id,
                    "field_id": field_id,
                    "crop_cycle_id": crop_id,
                    "read": False,
                    "created_at": utc_now(),
                }
            )
        except PyMongoError:
            # The primary write should not fail because an optional notification failed.
            return

    def list_notifications(self, user_id: ObjectId) -> List[Dict[str, Any]]:
        return [self._public(item) for item in self.mongo.collection("notifications").find({"user_id": user_id}).sort("created_at", -1).limit(200)]

    def mark_notification_read(self, user_id: ObjectId, notification_id: Any) -> Dict[str, Any]:
        object_id = _object_id(notification_id, "notification_id", required=True)
        result = self.mongo.collection("notifications").update_one({"_id": object_id, "user_id": user_id}, {"$set": {"read": True}})
        if result.matched_count == 0:
            raise NotFoundError("Notification was not found.")
        return self._public(self.mongo.collection("notifications").find_one({"_id": object_id, "user_id": user_id}))

    def mark_all_notifications_read(self, user_id: ObjectId) -> Dict[str, int]:
        result = self.mongo.collection("notifications").update_many(
            {"user_id": user_id, "read": False}, {"$set": {"read": True}}
        )
        return {"updated": result.modified_count}

    def dashboard(self, user_id: ObjectId) -> Dict[str, Any]:
        fields = self.mongo.collection("fields")
        crops = self.mongo.collection("crop_cycles")
        tasks = self.mongo.collection("care_tasks")
        predictions = self.mongo.collection("predictions")
        total_fields = fields.count_documents({"user_id": user_id, "status": {"$ne": "archived"}})
        active_crops = crops.count_documents({"user_id": user_id, "status": "active"})
        pending_tasks = tasks.count_documents({"user_id": user_id, "status": {"$in": ["pending", "in_progress"]}})
        health_checks = predictions.count_documents({"user_id": user_id})
        return {
            "stats": {
                "total_fields": total_fields,
                "active_fields": total_fields,
                "active_crops": active_crops,
                "pending_tasks": pending_tasks,
                "open_tasks": pending_tasks,
                "health_checks": health_checks,
                "completed_tasks": tasks.count_documents({"user_id": user_id, "status": "completed"}),
            },
            "fields": self.list_fields(user_id),
            "recent_predictions": self.list_predictions(user_id)[:5],
            "upcoming_tasks": self.list_tasks(user_id, status="pending")[:5],
            "recent_notifications": self.list_notifications(user_id)[:5],
        }

    def sync_diseases(self, catalog: Iterable[Dict[str, Any]]) -> int:
        count = 0
        collection = self.mongo.collection("diseases")
        for item in catalog:
            document = dict(item)
            document.pop("index", None)
            document["source"] = "disease_info.csv and supplement_info.csv"
            document["updated_at"] = utc_now()
            collection.update_one({"class_key": item["class_key"]}, {"$set": document}, upsert=True)
            count += 1
        return count
