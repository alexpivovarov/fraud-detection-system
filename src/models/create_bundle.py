'''
Create model bundle from existing saved files
'''
import joblib
import pandas as pd
from pathlib import Path

# Load existing model and mappings
model = joblib.load('src/models/xgboost_model.pkl')
category_mappings = joblib.load('src/models/category_mappings.pkl')

# Get feature names from the model itself
feature_names = model.get_booster().feature_names

# Best params from Optuna run
best_params = {
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

# Create bundle
model_bundle = {
    'model': model,
    'feature_names': feature_names,
    'category_mappings': category_mappings,
    'best_threshold': 0.70,
    'best_params': best_params
}

joblib.dump(model_bundle, 'src/models/fraud_model_bundle.pkl')
print(f"Model bundle saved!")
print(f"Feature count: {len(feature_names)}")
print(f"First 10 features: {feature_names[:10]}")
