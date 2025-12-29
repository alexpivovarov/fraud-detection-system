'''
FastAPI endpoint for real-time fraud scoring
'''

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.streaming.redis_client import FraudRedisClient


# Global variables for model and redis
model = None
redis_client = None

# Load model at startup
MODEL_PATH = Path(__file__).parent.parent / "src" / "models" / "xgboost_model.pkl"


@asynccontextmanager
async def lifespan(app: FastAPI):
    '''Load model and connect to Redis on startup, cleanup on shutdown.'''
    global model, redis_client

    # Startup
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Warning: Model not found at {MODEL_PATH}")

    try:
        redis_client = FraudRedisClient()
    except Exception as e:
        print(f"Warning: Model not found at {MODEL_PATH}")

    yield # App runs here

    # Shutdown
    print("Shutting down...")


# initialise app with lifespan
app = FastAPI(
    title = "Fraud Detection API",
    description = "Real-time fraud scoring for financial transactions",
    version = "1.0.0",
    lifespan=lifespan
)

class Transaction(BaseModel):
    '''Transaction input schema'''
    TransactionID: int
    TransactionAmt: float
    card1: int
    card4: Optional[str] = None
    card6: Optional[str] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    addr1: Optional[float] = None
    ProductCD: Optional[str] = None
    DeviceType: Optional[str] = None

class FraudDetection(BaseModel):
    '''Prediction response schema'''
    transaction_id: int
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    threshold: float

def prepare_features(df: pd.DataFrame, velocity_features: dict) -> pd.DataFrame:
        '''
        Prepare features for model prediciton.
        Must match training feature set.
        '''
        # Add velocity features
        df['txn_count_recent'] = velocity_features.get('txn_count_recent', 0)
        df['avg_amount_recent'] = velocity_features.get('avg_amount_recent', 0)

        # Add basic engineered features
        df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])

        # Select only numeric columns the model expects
        feature_cols = ['TransactionAmt', 'TransactionAmt_log', 'card1', 'txn_count_recent', 'avg_amount_recent']

        # Fill missing with 0
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        return df[feature_cols]

@app.get("/")
def root():
    return {"message": "Fraud Detection API", "status": "running"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "redis_connected": redis_client is not None
    }

@app.post("/predict", response_model=FraudDetection)
def predict_fraud(transaction: Transaction):
    '''
    Score a transaction for fraud probability.
    '''
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert to DataFrame (model expects this format)
    tx_dict = transaction.model_dump()
    df = pd.DataFrame([tx_dict])

    # Get velocity features from Redis if available
    velocity_features= {}
    if redis_client:
        try:
            card_id = str(transaction.card1)
            velocity_features = redis_client.get_velocity_features(card_id)
        except Exception:
            pass

    # Prepare features
    features = prepare_features(df, velocity_features)

    # Predict
    fraud_prob = model.predict_proba(features)[0][1]

    # Apply threshold
    threshold = 0.70
    is_fraud = fraud_prob >= threshold

    # Determine risk level
    if fraud_prob >= 0.8:
        risk_level = "High"
    elif fraud_prob >= 0.5:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # Update Redis with this transaction
    if redis_client:
        try:
            redis_client.update_card_profile(str(transaction.card1), tx_dict)
        except Exception:
            pass

    return FraudDetection(
        transaction_id = transaction.TransactionID,
        fraud_probability=round(fraud_prob, 4),
        is_fraud=is_fraud,
        risk_level=risk_level,
        threshold=threshold
    ) 

    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)