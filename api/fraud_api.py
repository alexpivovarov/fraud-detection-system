'''
FastAPI endpoint for real-time fraud scoring
'''

from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel # for defining data schemas (what the API expects/returns)
from typing import Optional # type hint for fields that can be None
import joblib # loads the saved XGBoost model from disk
import pandas as pd
import numpy as np # For log1p calculation
from pathlib import Path
from contextlib import asynccontextmanager # for the startup/shutdown lifecycle
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predictor import FraudPredictor
from src.streaming.redis_client import FraudRedisClient


# Global variables for model and redis (None initially, then populated when the app starts)
predictor = None
redis_client = None

# Load model at startup
MODEL_BUNDLE_PATH = Path(__file__).parent.parent / "src" / "models" / "fraud_model_bundle.pkl"


@asynccontextmanager
async def lifespan(app: FastAPI):
    '''Load model and connect to Redis on startup, cleanup on shutdown.
    
    App lifecycle manager.
    
    Server starts
         ↓
    Code BEFORE yield runs (load model, connect Redis)
         ↓
    yield → App handles requests
         ↓
    Server stops
         ↓
    Code AFTER yield runs (cleanup)
    '''
    global predictor, redis_client

    # Startup
    try:
        predictor = FraudPredictor(str(MODEL_BUNDLE_PATH))
    except FileNotFoundError:
        print(f"Warning: Model bundle not found at {MODEL_BUNDLE_PATH}")

    try:
        redis_client = FraudRedisClient()
    except Exception as e:
        print(f"Warning: Could not connect to Redis: {e}")

    yield # App runs here

    # Shutdown
    print("Shutting down...")


# initialise app with lifespan (creates the FastAPI app with metadata and attaches the lifespan handler)
app = FastAPI(
    title = "Fraud Detection API",
    description = "Real-time fraud scoring for financial transactions",
    version = "1.0.0",
    lifespan=lifespan
)

# What the API expects as input (request body)
class Transaction(BaseModel):
    '''Transaction input schema'''
    TransactionID: int
    TransactionAmt: float
    card1: int # card identifier
    card4: Optional[str] = None # card network
    card6: Optional[str] = None # card type
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    addr1: Optional[float] = None
    ProductCD: Optional[str] = None
    DeviceType: Optional[str] = None

# What the API returns (the response)
class FraudDetection(BaseModel):
    '''Prediction response schema'''
    transaction_id: int
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    threshold: float

@app.get("/")
def root():
    return {"message": "Fraud Detection API", "status": "running"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "redis_connected": redis_client is not None
    }

@app.post("/predict", response_model=FraudDetection) # the main endpoint
def predict_fraud(transaction: Transaction):
    '''
    Score a transaction for fraud probability.
    1. Check model exists
        ↓
    2. Convert transaction to DataFrame
        ↓
    3. Get velocity features from Redis (if available)
        ↓
    4. Prepare features for model
        ↓
    5. model.predict_proba() → fraud probability
        ↓
    6. Apply threshold (0.70) → is_fraud True/False
        ↓
    7. Determine risk level (High/Medium/Low)
        ↓
    8. Update Redis with this transaction
        ↓
    9. Return FraudDetection response
    '''
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Get velocity features from Redis if available
    velocity_features= {}
    if redis_client:
        try:
            velocity_features = redis_client.get_velocity_features(str(transaction.card1))
        except Exception:
            pass

    # Use predictor (single source of truth)
    result = predictor.predict(transaction.model_dump(), velocity_features)
    

    # Update Redis with this transaction
    if redis_client:
        try:
            redis_client.update_card_profile(str(transaction.card1), transaction.model_dump()) # Send data to Redis server over the network
        except Exception:
            pass

    return FraudDetection(
        transaction_id = transaction.TransactionID,
        fraud_probability=round(result['fraud_probability'], 4),
        is_fraud = result['is_fraud'],
        risk_level = result['risk_level'],
        threshold = result['threshold']
    ) 

    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)