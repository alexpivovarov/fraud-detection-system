# Fraud Detection System

A comprehensive end-to-end machine learning pipeline for real-time financial fraud detection, built to mirror industry-standard systems used by modern banks.

## Project Overview

This project implements a complete fraud detection system featuring:

- **Machine Learning Pipeline**: XGBoost model with hyperparameter tuning (Optuna)
- **Real-time Streaming**: Kafka-based transaction processing
- **Feature Engineering**: Velocity features, behavioral profiling, and graph analysis
- **Production Infrastructure**: Docker, Redis for caching, real-time scoring

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
│                      FRAUD DETECTION SYSTEM                     │
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
├── config/
│   ├── __init__.py
│   └── settings.py              # Centralized configuration
├── src/
│   ├── features/
│   │   ├── feature_engineering.py   # FraudFeatureEngineer class
│   │   └── graph_features.py        # Graph-based fraud network detection
│   ├── models/
│   │   ├── training.py              # XGBoost training with Optuna
│   │   └── xgboost_model.pkl        # Trained model
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

## Usage

### Training the Model

```bash
python3 src/models/training.py
```

This will:
- Load and preprocess the IEEE-CIS dataset
- Apply time-based train/test split (prevents data leakage)
- Run Optuna hyperparameter optimization (50 trials)
- Train XGBoost with optimized parameters
- Save the model to `models/xgboost_model.pkl`

### Real-time Streaming Pipeline

**Terminal 1** - Start the consumer:
```bash
python3 src/streaming/kafka_consumer.py
```

**Terminal 2** - Start the producer:
```bash
python3 src/streaming/kafka_producer.py
```

Transactions will flow through Kafka, get scored by the model, and trigger fraud alerts when probability exceeds 0.70.

### Graph Analysis

Detect fraud networks by analyzing relationships between cards, emails, and addresses:

```bash
python3 src/features/graph_features.py
```

## Technical Details

### Feature Engineering

The `FraudFeatureEngineer` class generates features across multiple categories:

| Category | Features |
|----------|----------|
| Time | Hour of day, day of week, weekend flag |
| Velocity | Transaction count in 1h/6h/24h/1 week windows |
| Card Behavior | Transaction count per card, average amount |
| Amount | Log transform, deviation from card average |

### Class Imbalance Handling

The dataset has a 3.5% fraud rate (1:27 ratio). Addressed via:
- `scale_pos_weight` in XGBoost (~25-27)
- Threshold optimization (0.50 → 0.70)
- Time-based splitting to prevent leakage

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

### Graph Analysis

NetworkX-based analysis detects fraud rings by:
- Building a graph of cards → emails/addresses/devices
- Calculating fraud neighbor ratios
- Identifying cards connected to known fraudsters

Example output:
```
Card 13926: fraud_neighbor_ratio=12.9% (high risk)
Card 2755: fraud_neighbor_ratio=6.2% (moderate risk)
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| ML Framework | XGBoost, scikit-learn |
| Hyperparameter Tuning | Optuna |
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

## Future Improvements

- [ ] Add LSTM for sequential transaction patterns
- [ ] Implement ensemble methods (LightGBM, RandomForest)
- [ ] Add Streamlit dashboard for real-time monitoring
- [ ] Integrate SHAP for model explainability
- [ ] Add anomaly detection layer (Isolation Forest)

## License

MIT License

## Acknowledgments

- Kaggle for the IEEE-CIS Fraud Detection dataset
