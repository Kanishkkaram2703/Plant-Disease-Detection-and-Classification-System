"""MongoDB connection and serialization boundary for PlantCare."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any, Dict, Optional

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError


# This project intentionally has one local MongoDB boundary.  Keep the URI
# fixed so an inherited environment variable cannot accidentally target the
# unrelated default MongoDB instance on port 27017.
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://127.0.0.1:27018/plantcare_ai")
MONGODB_DATABASE = os.environ.get("MONGODB_DATABASE", "plantcare_ai")


class DatabaseUnavailableError(Exception):
    """Raised when PlantCare MongoDB cannot be reached."""


class InvalidObjectIdError(ValueError):
    """Raised when an API identifier is not a valid Mongo ObjectId."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_object_id(value: Any) -> ObjectId:
    if isinstance(value, ObjectId):
        return value
    if not isinstance(value, str) or not ObjectId.is_valid(value):
        raise InvalidObjectIdError("Invalid resource identifier.")
    return ObjectId(value)


def serialize(value: Any) -> Any:
    """Convert MongoDB values into JSON-safe values without exposing internals."""

    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, list):
        return [serialize(item) for item in value]
    if isinstance(value, tuple):
        return [serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialize(item) for key, item in value.items()}
    return value


class MongoStore:
    """One lazy, reusable client for the dedicated PlantCare database."""

    COLLECTIONS = (
        "users",
        "fields",
        "crop_cycles",
        "predictions",
        "diseases",
        "care_tasks",
        "irrigation_logs",
        "plant_health_logs",
        "notifications",
        "agricultural_rules",
    )

    def __init__(self, uri: str = MONGODB_URI, database: str = MONGODB_DATABASE):
        self.uri = uri
        self.database_name = database
        self.client: Optional[MongoClient] = None
        self.db = None

    def connect(self):
        if self.client is None:
            self.client = MongoClient(
                self.uri,
                serverSelectionTimeoutMS=1500,
                connectTimeoutMS=1500,
                socketTimeoutMS=3000,
                tz_aware=True,
            )
            self.db = self.client[self.database_name]
        return self.db

    def ping(self) -> Dict[str, Any]:
        try:
            database = self.connect()
            return database.command("ping")
        except (ServerSelectionTimeoutError, PyMongoError) as exc:
            raise DatabaseUnavailableError(
                "PlantCare MongoDB is unavailable. Start the configured instance on port 27018."
            ) from exc

    def collection(self, name: str) -> Collection:
        if name not in self.COLLECTIONS:
            raise ValueError("Unknown PlantCare collection.")
        try:
            database = self.connect()
            return database[name]
        except PyMongoError as exc:
            raise DatabaseUnavailableError("PlantCare MongoDB is unavailable.") from exc

    def ensure_indexes(self) -> None:
        """Create only indexes needed by current ownership and timeline queries."""

        try:
            self.collection("users").create_index([("email", ASCENDING)], unique=True)
            self.collection("fields").create_index([("user_id", ASCENDING), ("status", ASCENDING)])
            self.collection("fields").create_index([("user_id", ASCENDING), ("updated_at", DESCENDING)])
            self.collection("crop_cycles").create_index([("user_id", ASCENDING), ("field_id", ASCENDING)])
            self.collection("crop_cycles").create_index([("user_id", ASCENDING), ("status", ASCENDING)])
            self.collection("predictions").create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
            self.collection("predictions").create_index([("field_id", ASCENDING), ("crop_cycle_id", ASCENDING)])
            self.collection("care_tasks").create_index([("user_id", ASCENDING), ("status", ASCENDING), ("due_date", ASCENDING)])
            self.collection("care_tasks").create_index([("field_id", ASCENDING), ("crop_cycle_id", ASCENDING)])
            self.collection("care_tasks").create_index([("user_id", ASCENDING), ("dedup_key", ASCENDING)], unique=True, sparse=True)
            self.collection("irrigation_logs").create_index([("user_id", ASCENDING), ("recorded_at", DESCENDING)])
            self.collection("plant_health_logs").create_index([("user_id", ASCENDING), ("observation_date", DESCENDING)])
            self.collection("notifications").create_index([("user_id", ASCENDING), ("read", ASCENDING), ("created_at", DESCENDING)])
            self.collection("diseases").create_index([("class_key", ASCENDING)], unique=True)
        except (ServerSelectionTimeoutError, PyMongoError) as exc:
            raise DatabaseUnavailableError("PlantCare MongoDB is unavailable.") from exc


store = MongoStore()
