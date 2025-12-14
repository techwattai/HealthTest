from fastapi import WebSocket
import joblib, time, random, asyncio
import pandas as pd
from Configurations.config import settings


# ----------------------------
# Load trained model & objects
# ----------------------------
# Paths to saved model & preprocessing
if settings.DEPLOYMENT == "local":
    model_path = r"C:\Users\TechWatt\Desktop\techwatt\backend-ai\Models\multioutput_model.pkl"
    scaler_path = r"C:\Users\TechWatt\Desktop\techwatt\backend-ai\Models\scaler.pkl"
    status_encoder_path = r"C:\Users\TechWatt\Desktop\techwatt\backend-ai\Models\label_encoder_status.pkl"
    condition_encoder_path = r"C:\Users\TechWatt\Desktop\techwatt\backend-ai\Models\label_encoder_condition.pkl"

else:
    model_path = "Models/multioutput_model.pkl"
    scaler_path = "Models/scaler.pkl"
    status_encoder_path = "Models/label_encoder_status.pkl"
    condition_encoder_path = "Models/label_encoder_condition.pkl"


# Load the trained model and preprocessing objects
clf = joblib.load(model_path)
scaler = joblib.load(scaler_path)
le_status = joblib.load(status_encoder_path)
le_condition = joblib.load(condition_encoder_path)

# Feature names used during training
FEATURE_NAMES = [
    "heart_rate",
    "resp_rate",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "spo2",
    "temperature_c",
    "glucose_mgdl"
]


async def sensor_data_stream(websocket: WebSocket):
    """Continuously send random patient vitals and ML predictions."""
    await websocket.accept()
    try:
        while True:
            # Generate fake vitals data
            new_data = {
                "heart_rate": random.randint(50, 160),
                "resp_rate": random.randint(10, 30),
                "blood_pressure_systolic": random.randint(90, 160),
                "blood_pressure_diastolic": random.randint(60, 100),
                "spo2": random.randint(85, 100),
                "temperature_c": round(random.uniform(35.0, 40.0), 1),
                "glucose_mgdl": random.randint(60, 200),
            }

            # Convert to DataFrame with correct feature names
            new_data_df = pd.DataFrame([new_data], columns=FEATURE_NAMES)

            # Scale features
            new_data_scaled = scaler.transform(new_data_df)

            # Predict both label_status & probable_condition
            pred = clf.predict(new_data_scaled)
            
            # Decode predictions
            status_pred = le_status.inverse_transform([pred[0][0]])[0]
            condition_pred = le_condition.inverse_transform([pred[0][1]])[0]
           

            # Build message
            message = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "data": new_data,
                "prediction": {
                    "label_status": status_pred,
                    "probable_condition": condition_pred,
                    "confidence": {
                        "label_status": round(max(clf.predict_proba(new_data_scaled)[0][0]), 2) * 100,
                        "probable_condition": round(max(clf.predict_proba(new_data_scaled)[1][0]), 2) * 100
                    }
                }
            }

            # Send prediction to client
            await websocket.send_json(message)

            # Wait 1 second before next reading
            await asyncio.sleep(1)

    except asyncio.CancelledError:
        await websocket.close()
    except Exception as e:
        print(f"Unexpected error in sensor_data_stream: {e}")
        await websocket.close()

