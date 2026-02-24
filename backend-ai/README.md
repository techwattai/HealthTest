# Backend-AI Synthetic Vitals Data Generator & Analysis

## Overview

This project generates synthetic patient vitals data for machine learning and exploratory data analysis (EDA). It includes data generation, EDA notebooks, model training utilities, and a FastAPI API for live ML predictions.

## Project Structure

```
.
├── main.py                  # FastAPI app entry point
├── Endpoints/               # API endpoints
│   ├── body_vitals.py      # WebSocket endpoint for vitals streaming
│   └── ai_appointments.py   # AI appointments endpoints
├── Data Generator/
│   └── dataGenerator.py     # Synthetic data generator script
├── Datasets/
│   └── vitals.csv          # Generated synthetic vitals dataset
├── EDA/
│   └── vitals.ipynb        # Exploratory Data Analysis notebook
├── Models/
│   ├── label_encoder_condition.pkl
│   ├── label_encoder_status.pkl
│   ├── multioutput_model.pkl
│   └── scaler.pkl
├── TrainingPipeline/
│   └── train.ipynb         # Model training notebook
├── Configurations/
│   └── config.py           # Application settings
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
└── README.md
```

## FastAPI Live Prediction API

The API provides real-time patient vitals monitoring and ML-based health predictions.

### WebSocket Endpoint

- **URL**: `ws://localhost:8000/api/ws/predict`
- **Description**: Streams synthetic vitals data and predictions every second
- **Response Format**:
```json
{
    "timestamp": "2025-10-18 10:30:45",
    "data": {
        "heart_rate": 75,
        "resp_rate": 16,
        "blood_pressure_systolic": 120,
        "blood_pressure_diastolic": 80,
        "spo2": 98,
        "temperature_c": 37.0,
        "glucose_mgdl": 100
    },
    "prediction": {
        "label_status": "normal",
        "probable_condition": "healthy",
        "confidence": {
            "label_status": 95.5,
            "probable_condition": 92.3
        }
    }
}
```

### Run the API Server

```sh
uvicorn main:app --reload
```

### Example WebSocket Client (Python)

```python
import websockets
import asyncio
import json

async def listen():
    uri = "ws://localhost:8000/api/ws/predict"
    async with websockets.connect(uri) as ws:
        while True:
            msg = await ws.recv()
            print(json.loads(msg))

asyncio.run(listen())
```

## Data Generation

- Generate synthetic data using [`Data Generator/dataGenerator.py`](Data%20Generator/dataGenerator.py)
- Includes vital signs and their corresponding status labels
- Configurable sample size (default: 10,000 records)

```sh
python "Data Generator/dataGenerator.py"
```

## Model Training & EDA

1. Generate data and place in `Datasets/vitals.csv`
2. Explore data using `EDA/vitals.ipynb`
3. Train models with `TrainingPipeline/train.ipynb`
4. Models are saved to `Models/` directory

## Configuration

Edit `Configurations/config.py` to set:
- Model paths
- Deployment environment
- Other application settings

## Requirements

- Python 3.13+
- FastAPI
- Uvicorn
- NumPy
- Pandas
- Scikit-learn
- Joblib
- Websockets
- See `pyproject.toml` for complete dependencies

## License

MIT License