# Fraud Detection System

A comprehensive end-to-end machine learning pipeline for real-time financial fraud detection, built to mirror industry-standard systems used by modern banks.

## Project Overview

This project implements a complete fraud detection system featuring:

- **Machine Learning Pipeline**: XGBoost model with Optuna hyperparameter tuning
- **Real-time Streaming**: Kafka-based transaction processing
- **Feature Engineering**: Velocity features, behavioral profiling, and graph analysis
- **Production Infrastructure**: Docker, Redis for caching, FastAPI for serving
- **Visual Dashboard**: Streamlit interface for real-time fraud scoring

### Key Results

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.893 |
| F1 Score (optimized threshold) | 0.49 |
| Optimal Threshold | 0.70 |
| Precision | 0.66 |
| Recall | 0.38 |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     STREAMLIT DASHBOARD                         │
│                  (Visual fraud scoring UI)                      │
├─────────────────────────────────────────────────────────────────┤
│                       FASTAPI ENDPOINT                          │
│                    (REST API for scoring)                       │
├─────────────────────────────────────────────────────────────────┤
│                      FRAUD PREDICTOR                            │
│            (Single source of truth for inference)               │
├──────────────────┬──────────────────┬───────────────────────────┤
│  KAFKA PRODUCER  │  KAFKA CONSUMER  │  REDIS                    │
│  (Streams txns)  │  (Processes)     │  (Card profiles, velocity)│
├──────────────────┴──────────────────┴───────────────────────────┤
│                        XGBOOST MODEL                            │
│              (Hyperparameter-tuned with Optuna)                 │
├─────────────────────────────────────────────────────────────────┤
│                      FEATURE ENGINEERING                        │
│    (Time features, velocity, card behavior, graph analysis)     │
├─────────────────────────────────────────────────────────────────┤
│                       DATA LAYER                                │
│              (IEEE-CIS Fraud Detection Dataset)                 │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
fraud-detection-system/
├── api/
│   ├── __init__.py
│   └── fraud_api.py             # FastAPI endpoint for real-time scoring
├── config/
│   ├── __init__.py
│   └── settings.py              # Centralized configuration
├── src/
│   ├── features/
│   │   ├── feature_engineering.py   # FraudFeatureEngineer class
│   │   └── graph_features.py        # Graph-based fraud network detection
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py             # FraudPredictor class (single source of truth)
│   ├── models/
│   │   ├── training.py              # XGBoost training with Optuna
│   │   ├── create_bundle.py         # Creates model bundle for inference
│   │   ├── fraud_model_bundle.pkl   # Model + feature names + mappings
│   │   └── xgboost_model.pkl        # Trained model
│   ├── scripts/
│   │   └── seed_redis.py            # Populates Redis with card history
│   ├── streaming/
│   │   ├── kafka_producer.py        # Streams transactions to Kafka
│   │   ├── kafka_consumer.py        # Real-time processing & predictions
│   │   └── redis_client.py          # Card profile storage
│   └── utils/
├── data/
│   ├── raw/                         # IEEE-CIS dataset
│   └── processed/
├── models/                          # Saved models
├── docker-compose.yml               # Kafka, Zookeeper, Redis
├── streamlit_app.py                 # Visual dashboard
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- ~8GB RAM (for model training)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/alexpivovarov/fraud-detection-system.git
   cd fraud-detection-system
   ```

2. **Install dependencies**
   ```bash
   pip3 install -r requirements.txt
   ```

3. **Download the dataset**
   
   Download the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection) from Kaggle and place the CSV files in `data/raw/`.

4. **Start infrastructure**
   ```bash
   docker-compose up -d
   ```

5. **Seed Redis with card history**
   ```bash
   python3 src/scripts/seed_redis.py
   ```

## Usage

### Option 1: Streamlit Dashboard (Recommended for Demo)

Start the API and dashboard:

```bash
# Terminal 1: Start API
python3 api/fraud_api.py

# Terminal 2: Start dashboard
python3 -m streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

### Option 2: REST API

Start the API:

```bash
python3 api/fraud_api.py
```

Test with curl:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"TransactionID": 1, "TransactionAmt": 150.0, "card1": 13926}'
```

Response:
```json
{
  "transaction_id": 1,
  "fraud_probability": 0.0173,
  "is_fraud": false,
  "risk_level": "Low",
  "threshold": 0.7
}
```

Interactive docs available at `http://localhost:8000/docs`

### Option 3: Kafka Streaming Pipeline

Process transactions in real-time:

```bash
# Terminal 1: Start consumer
python3 src/streaming/kafka_consumer.py

# Terminal 2: Start producer
python3 src/streaming/kafka_producer.py
```

Output:
```
Processed 500 transactions, 0 fraud alerts
Processed 1000 transactions, 0 fraud alerts
🚨 FRAUD ALERT! TransactionID=2990862, Amount=$32.97, Prob=88.51%, Risk=High
```

### Training the Model

```bash
python3 src/models/training.py
```

This will:
- Load and preprocess the IEEE-CIS dataset
- Apply time-based train/test split (prevents data leakage)
- Run Optuna hyperparameter optimization (50 trials)
- Train XGBoost with optimized parameters
- Save the model bundle

## Technical Details

### Feature Engineering

The `FraudFeatureEngineer` class generates features across multiple categories:

| Category | Features |
|----------|----------|
| Time | Hour of day, day of week, weekend flag |
| Velocity | Transaction count per card, average amount |
| Amount | Log transform, decimal, is_round flag |
| Email | Gmail/Yahoo/Hotmail flags, fraud rate |

### Class Imbalance Handling

The dataset has a 3.5% fraud rate (1:27 ratio). Addressed via:
- `scale_pos_weight` in XGBoost (~25)
- Threshold optimization (0.50 → 0.70)
- Time-based splitting to prevent data leakage

### Hyperparameter Optimization

Optuna Bayesian optimization found:

```python
{
    'n_estimators': 235,
    'max_depth': 10,
    'learning_rate': 0.104,
    'scale_pos_weight': 25.35,
    'min_child_weight': 7,
    'subsample': 0.892,
    'colsample_bytree': 0.948
}
```

### Model Bundle

The `fraud_model_bundle.pkl` contains everything needed for inference:
- Trained XGBoost model
- Feature names (407 features)
- Category mappings
- Optimal threshold (0.70)
- Best hyperparameters

### FraudPredictor Class

Single source of truth for inference, used by both API and Kafka consumer:

```python
from src.inference.predictor import FraudPredictor

predictor = FraudPredictor('src/models/fraud_model_bundle.pkl')
result = predictor.predict(transaction, velocity_features)
# Returns: {'fraud_probability': 0.85, 'is_fraud': True, 'risk_level': 'High'}
```

### Graph Analysis

NetworkX-based analysis detects fraud rings by:
- Building a graph of cards → emails/addresses/devices
- Calculating fraud neighbor ratios
- Identifying cards connected to known fraudsters

## Technology Stack

| Component | Technology |
|-----------|------------|
| ML Framework | XGBoost, scikit-learn |
| Hyperparameter Tuning | Optuna |
| API | FastAPI, Uvicorn |
| Dashboard | Streamlit |
| Streaming | Apache Kafka |
| Caching | Redis |
| Graph Analysis | NetworkX |
| Container Orchestration | Docker Compose |
| Model Persistence | joblib |

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) from Kaggle:
- ~590,000 transactions
- 3.5% fraud rate
- 400+ features (transaction, identity, device)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed system status |
| `/predict` | POST | Score a transaction |
| `/docs` | GET | Interactive API documentation |

## Future Improvements

- [ ] Add LSTM for sequential transaction patterns
- [ ] Implement ensemble methods (LightGBM, RandomForest)
- [ ] Integrate SHAP for model explainability
- [ ] Add anomaly detection layer (Isolation Forest)
- [ ] Fix velocity feature alignment between training and inference
- [ ] Add authentication to API

## License

MIT License

## Acknowledgments

- Kaggle for the IEEE-CIS Fraud Detection dataset
