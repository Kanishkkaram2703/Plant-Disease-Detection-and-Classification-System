# PlantCare MongoDB boundary

The farmer and Plant Doctor persistence layer uses one dedicated local MongoDB
instance:

```text
mongodb://127.0.0.1:27018/plantcare_ai
```

The application does not connect to port `27017`. The checked-in configuration
is `PlantCare-MongoDB/mongod.cfg`; it binds only to `127.0.0.1` and stores data
under `PlantCare-MongoDB/data`.

## Start the dedicated instance

From the project directory, start only the configured instance:

```powershell
& "C:\Program Files\MongoDB\Server\8.3\bin\mongod.exe" --config ".\PlantCare-MongoDB\mongod.cfg"
```

In a second terminal, verify the exact port:

```powershell
& "C:\Program Files\mongosh\mongosh.exe" "mongodb://127.0.0.1:27018/plantcare_ai" --quiet --eval "db.runCommand({ping:1})"
```

The Flask process starts in a safe degraded mode when this instance is offline;
Mongo-backed endpoints then return a clear `503` instead of silently using a
different database.

## Collections

The service owns these collections: `users`, `fields`, `crop_cycles`,
`predictions`, `diseases`, `care_tasks`, `irrigation_logs`,
`plant_health_logs`, `notifications`, and `agricultural_rules`.

Ownership is enforced with a `user_id` on farmer-owned records. ObjectId,
field/crop relationships, date values, enum values and text lengths are
validated at the service boundary. Indexes are created by the application on
first successful MongoDB startup.

## API surface

The implemented JSON endpoints are:

```text
GET/POST/PATCH /api/profile
GET/PATCH       /api/settings
GET/POST        /logout
GET/POST       /api/fields
GET/PATCH/DELETE /api/fields/<field_id>
GET            /api/fields/<field_id>/overview
GET/POST       /api/crops
GET/PATCH/DELETE /api/crops/<crop_id>
GET/POST       /api/tasks
PATCH          /api/tasks/<task_id>
POST           /api/tasks/<task_id>/complete
GET/POST       /api/irrigation
GET/POST       /api/health-logs
GET            /api/predictions
GET            /api/predictions/<prediction_id>
GET            /api/notifications
POST           /api/notifications/<notification_id>/read
POST           /api/notifications/read-all
GET            /api/dashboard
GET            /api/diseases
```

The existing `POST /api/predict` remains the single Plant Doctor inference
boundary. It accepts optional `field_id` and `crop_cycle_id` multipart fields
and stores the real model result when a farmer profile is available.

No Cassandra, SQLite, chatbot, LLM or voice persistence is used by this
implementation.

## Task rules

The `agricultural_rules` collection is the only source used by automatic task
generation. A rule must be explicitly enabled and declare its trigger,
conditions, task content, and source/reason. The application currently ships
with no fabricated or unverified rules, so ordinary crop and prediction events
do not generate recommendations until a verified rule document is loaded.
Generated tasks store `generated`, `source_rule_id`, `generation_reason`,
`generated_at`, and a unique `dedup_key`. Manual farmer tasks are stored with
`generated: false`.

Deleting a field with related records archives it instead of deleting the
field or its history. A field without related records can be hard-deleted.
