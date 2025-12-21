"""
Fraud Detection System Configuration
"""
from pathlib import Path

#Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model parameters
RANDOM_STATE = 42
Test_size = 0.2 # 20% of data for testing, 80% for training
Validation_size = 0.1

# Feature groups 
TRANSACTION_FEATURES = [
    "TransactionAmt", "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"
]

VELOCITY_WINDOWS = [1, 6, 24, 168] # count transactions in 1h, 6h, 24h, 1 week windows

# Class imbalance
FRAUD_WEIGHT = 10 # A missed fraud costs 10x more than a false alarm.

#Thresholds
HIGH_RISK_THRESHOLD = 0.8 # score > 0.8 -> likely fraud
MEDIUM_RISK_THRESHOLD = 0.5 # score 0.5-0.8 -> needs review
LOW_RISK_THRESHOLD = 0.2

# Infrastructure
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TRANSACTION_TOPIC = "transactions"
KAFKA_SCORED_TOPIC = "scored_transactions"
REDIS_HOST = "localhost"
REDIS_PORT = 6379

# API
API_HOST = "0.0.0.0"
API_PORT = 8000

# Optuna hyperparameter tuning

BEST_XGBOOST_PARAMS = {
    'n_estimators': 235,
    'max_depth': 10,
    'learning_rate': 0.104,
    'scale_pos_weight': 25.35,
    'min_child_weight': 7,
    'subsample': 0.892,
    'colsample_bytree': 0.948,
    'random_state': 42,
    'eval_metric': 'auc'
}