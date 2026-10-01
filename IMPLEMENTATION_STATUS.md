# PlantCare AI implementation status

Audit and implementation scope: the legacy Flask project at
`Plant-Disease-Detection-main`. The separate PlantCare 38-class model was not
used.

## Current application

- Frontend: Flask server-rendered HTML, CSS and browser JavaScript.
- Backend: Flask with a JSON prediction endpoint at `POST /api/predict`.
- Model: `Model/plant_disease_model_1_latest.pt`, loaded once when the process starts.
- Architecture: `Flask Deployed App/CNN.py`, 39 output classes.
- Knowledge: `disease_info.csv` and `supplement_info.csv`; no new disease claims were added.
- Prediction history: MongoDB persistence when a farmer profile is available;
  browser `localStorage` remains a fallback for anonymous Plant Doctor use.
- Farmer pages: server-rendered Flask pages backed by the MongoDB service layer.
- Farmer UI: existing green SaaS visual system preserved; profile/account state,
  real settings, field option controls, field archive protection, condition
  filtering, task provenance and notification read-all behavior are wired to
  real endpoints.

## Plant Doctor implementation

Upload and camera captures both submit to the same `/api/predict` service. The
service validates the file, converts RGB/RGBA/grayscale-compatible inputs to
RGB, reproduces the verified training transform, runs the real CPU model, and
returns the primary result plus top three model outputs.

The transform is:

`Resize(255) -> CenterCrop(224) -> RGB -> ToTensor (/255) -> NCHW`

The displayed probability is the model's softmax score. It is explicitly not a
calibrated confidence or a disease-severity score. A configurable 90% screening
threshold is used to present an uncertain state; this threshold is based on the
audit-only sample and is not independent validation.

## Camera status

The camera workflow is real browser `getUserMedia` support with permission,
live preview, capture, retake, preview and analyze states. It does not run
continuous inference. Automated physical-camera verification is **NOT VERIFIED**
because the available Playwright `npx` bootstrap did not start in this
environment. Manual device verification is required.

## Farmer workspace status

Dashboard, Fields, Add Field, Field Details, Crop Management, Irrigation,
Tasks, Notifications, Profile and Settings screens are connected to the
ownership-aware MongoDB service layer. They only display records returned by
the API and use explicit empty states when no records exist.

- MongoDB: implemented at `mongodb://127.0.0.1:27018/plantcare_ai`; see
  `MONGODB.md`.
- Cassandra: not used.
- Irrigation calculation engine: not implemented; only completion records are
  stored.
- Crop-stage calculation: not implemented; stage remains a farmer-selected
  observation.
- Automatic task generation: rule-engine architecture and deduplication are
  implemented, but no unverified rules are seeded. No generated task appears
  unless a verified `agricultural_rules` document is present.
- Notifications: persistence, event notifications, preferences, read state
  and mark-all-read are implemented; predictive notification rules are not
  implemented.
- Existing data audit: duplicate-looking user-owned field/crop records were
  observed with creation times milliseconds apart. They were not deleted
  because their origin was not established; new form submissions are guarded
  against double submission.
- Identity boundary: profile-scoped Flask sessions are implemented for local
  ownership checks; password authentication, account recovery and production
  identity management are not implemented.
- Chatbot, LLM and voice assistant: not implemented.

## Verification

The Plant Doctor test suite is `tests/test_plant_doctor.py` and runs with
standard-library `unittest`. Mongo-backed integration coverage is in
`tests/test_farmer_mongodb.py` and requires the dedicated instance on port
27018. It covers profile ownership, field/crop/task/irrigation/health CRUD,
notifications, dashboard data and prediction persistence. The live Flask
server was checked for health, page routes and a real image prediction when
the required services were available.

The forensic model limitations remain: no independent test set, exact duplicate
risk, known high-confidence errors and no calibrated confidence. This phase
does not retrain or replace the model.
