# Plant Disease Detection and PlantCare Farmer Workspace

This repository contains the legacy 39-class PyTorch Plant Doctor model and its Flask application. The current application keeps the existing model and adds ownership-aware farmer, field, crop, task, irrigation-log, plant-health-log, notification and prediction persistence.

## Start the dedicated MongoDB instance

PlantCare uses only the dedicated local MongoDB instance at port `27018` and database `plantcare_ai`. It never targets MongoDB's default port `27017`.

```powershell
& "C:\Program Files\MongoDB\Server\8.3\bin\mongod.exe" --config ".\PlantCare-MongoDB\mongod.cfg"
```

Verify it in another terminal:

```powershell
& "C:\Program Files\mongosh\mongosh.exe" "mongodb://127.0.0.1:27018/plantcare_ai" --quiet --eval "db.runCommand({ping:1})"
```

## Start Flask

From `Flask Deployed App`:

```powershell
python app.py
```

Open `http://127.0.0.1:5000/`. Create a farmer profile before creating fields or persisting linked predictions.

## Install dependencies

```powershell
python -m pip install -r requirements.txt
```

The model artifact remains at `Model/plant_disease_model_1_latest.pt`. It is loaded by `Flask Deployed App/CNN.py` and is not retrained or replaced by the application.

## Test

With the dedicated MongoDB instance running:

```powershell
python -m unittest discover -s tests -v
```

The Plant Doctor tests run without MongoDB. Farmer integration tests require the instance on port `27018` and clean up their generated records. See `MONGODB.md` and `IMPLEMENTATION_STATUS.md` for boundaries and current limitations.
