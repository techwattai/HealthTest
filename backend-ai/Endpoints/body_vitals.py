import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Vitals.body_vitals import sensor_data_stream
from fastapi import APIRouter, WebSocket


router = APIRouter()

# For WebSocket endpoint to stream sensor data and predictions
@router.websocket("/ws/predict",)
async def websocket_endpoint(websocket:WebSocket):
    """
    WebSocket endpoint to stream sensor data and predictions.

    Protocol:
    - The client should connect to this endpoint using a WebSocket.
    - The server will periodically send JSON-formatted sensor data and prediction results.
    - The client is not required to send messages, but if it does, they should be JSON objects (specify expected keys if any).
    - Example server message:
      {
        "timestamp": "2024-06-01T12:00:00Z",
        "sensor_data": {...},
        "prediction": "Normal"
      }
    """
    await sensor_data_stream(websocket)


