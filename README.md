# Plant Disease Detection and PlantCare Farmer Workspace

This repository contains a 39-class PyTorch plant disease model and its Flask application. The application also stores farmer profiles, fields, crop cycles, tasks, irrigation logs, plant health logs, notifications, and prediction history in MongoDB.

## Repository structure

```text
Flask Deployed App/   Flask routes, templates, static files, and model service
Model/                PyTorch model, training notebook, and model documentation
PlantCare-MongoDB/    Local Windows MongoDB configuration
_audit/               Forensic model audit evidence and reports
demo_images/          Demo images
test_images/          Existing image examples
tests/                Plant Doctor and MongoDB integration tests
```

The production model is `Model/plant_disease_model_1_latest.pt`. It is loaded by `Flask Deployed App/plant_doctor.py` through the architecture in `Flask Deployed App/CNN.py`.

## Requirements

- Python 3.8 is recommended because the project uses the legacy CPU-only PyTorch 1.8.1 runtime.
- MongoDB is required for farmer data and prediction persistence.
- Git LFS is required to download the model file from GitHub. Run `git lfs pull` after cloning.

## Run locally on Windows

From the repository root, install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create the local MongoDB folders:

```powershell
New-Item -ItemType Directory -Force '.\PlantCare-MongoDB\data' | Out-Null
New-Item -ItemType Directory -Force '.\PlantCare-MongoDB\log' | Out-Null
```

Start MongoDB in the first terminal:

```powershell
& 'C:\Program Files\MongoDB\Server\8.3\bin\mongod.exe' --config '.\PlantCare-MongoDB\mongod.cfg'
```

MongoDB uses port `27018` and database `plantcare_ai`.

Start Flask in a second terminal:

```powershell
Set-Location '.\Flask Deployed App'
python app.py
```

Open `http://127.0.0.1:5000/`.

## Run with Docker Compose

Docker Compose starts the Flask application and the MongoDB service:

```powershell
docker compose up --build
```

Open `http://127.0.0.1:5000/`.

Stop the services without deleting stored MongoDB data:

```powershell
docker compose down
```

The Compose setup maps MongoDB to host port `27018`. Inside the Compose network, the Flask container uses `mongodb:27017`.

The model is managed through Git LFS. If the model file is only a small LFS pointer after cloning, install Git LFS and run:

```powershell
git lfs pull
```

## Verify the application

Run the tests from the repository root:

```powershell
python -m unittest discover -s tests -v
```

The Plant Doctor tests do not require MongoDB. The farmer integration tests require the local MongoDB instance on port `27018` and clean up their test records.

Check the application health endpoint:

```text
http://127.0.0.1:5000/api/health
```

## MongoDB connection

The local connection is:

```text
mongodb://127.0.0.1:27018/plantcare_ai
```

The Docker connection is supplied through `MONGODB_URI` in `compose.yaml`. The application keeps the same database name in both environments.

## Scope

The application provides plant image screening, farmer field records, crop records, task tracking, irrigation and health logs, notifications, disease catalog data, and prediction history. Model probabilities are softmax scores and are not calibrated guarantees. The model can support screening, but its result should be checked against the plant and the visible symptoms.

Do not commit MongoDB data, MongoDB logs, uploaded images, credentials, Python cache files, or the original training dataset archive.
