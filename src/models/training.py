'''
Model training for Fraud Detection
'''

import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import sys

# Adding project's root directory to Python's import path so we can import from other folders in the project
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features.feature_engineering import FraudFeatureEngineer

# The function returns a tuple of 2 DataFrames
def time_based_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]: 
    '''
    Split data chronologically based on TransactionDT.
    '''
    df_sorted = df.sort_values('TransactionDT').reset_index(drop=True) # sorts the DataFrame by the TransactionDT; drop=True discards the index (clean sequential indexing makes the split calculation simpler)

    split_idx = int(len(df_sorted) * (1 - test_size)) # calculating the split point

    train_df = df_sorted.iloc[:split_idx] # Everything before row 800 (earlier transactions)
    test_df = df_sorted.iloc[split_idx:] # Everything from row 800 onwards (later transactions)

    print(f"Training set: {len(train_df)} transactions")
    print(f"Test set: {len(test_df)} transactions")
    print(f"Training fraud rate: {train_df['isFraud'].mean():.2%}") # the mean of 0s and 1s gives the proportion of 1s (frauds); format as percentage to 2 d.p.
    print(f"Test fraud rate: {test_df['isFraud'].mean():.2%}")

    return train_df, test_df


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, 
                    X_test: pd.DataFrame, y_test: pd.Series) -> xgb.XGBClassifier:
    '''
    Train XGBoost model with class imbalance hanfling
    '''

    # Calculate class weight (ratio of non-fraud to fraud)
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    print(f"Scale pos weight: {scale_pos_weight:.2f}")

    # Initialize model
    model = xgb.XGBClassifier(
    n_estimators = 100,
    max_depth = 6,
    learning_rate = 0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='auc'
    )

    # Train
    print("Training XGBoost")
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Evaluate
    print("\n" + "="*50)
    print("Model Evaluation")
    print("="*50)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud']))

    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

    return model





if __name__ == "__main__":

    DATA_PATH = "/Users/alexpivovarov/fraud-detection-system/data/raw/train_transaction.csv"

    print("Loading data ...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} transactions")

    train_df, test_df = time_based_split(df)


    print("\nApplying feature engineering to training data ... ")
    engineer = FraudFeatureEngineer()
    train_df = engineer.fit_transform(train_df)

    print("\nApplying feature engineering to test data ... ")
    test_df = engineer.transform(test_df)

    X_train, y_train = engineer.prepare_features_for_model(train_df)
    X_test, y_test = engineer.prepare_features_for_model(test_df)

    model = train_xgboost(X_train, y_train, X_test, y_test)