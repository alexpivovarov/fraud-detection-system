'''
Inference pipeline - loads model bundle and makes predictions
'''

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

class FraudPredictor:
    '''
    Loads the trained model bundle and applies the same feature engineering used during training.
    '''

    def __init__(self, bundle_path: str):
        bundle = joblib.load(bundle_path)
        self.model = bundle['model']
        self.feature_names = bundle['feature_names']
        self.category_mappings = bundle.get('category_mappings', {})
        self.threshold = bundle.get('best_threshold', 0.70)
        print(f"Loaded model with {len(self.feature_names)} features")
        print(f"Threshold: {self.threshold}")

    def predict(self, transaction: dict, velocity_features: dict = None) -> dict:
        '''
        Score a single transaction.

        Args:
            transaction: dict with transaction data
            velocity_features: dict with velocity features from Redis (optional)

        Returns:
            dict with fraud_probability, is_fraud, risk_level
        '''

        # Convert to DataFrame
        df = pd.DataFrame([transaction])

        # Apply feature engineering
        df = self._engineer_features(df, velocity_features or {})

        # Align features to match model expectations
        df = self._align_features(df)

        # Predict
        fraud_prob = self.model.predict_proba(df)[0][1]

        return {
            'fraud_probability': float(fraud_prob),
            'is_fraud': fraud_prob >= self.threshold,
            'risk_level': self._get_risk_level(fraud_prob),
            'threshold': self.threshold
        }
    
    def _engineer_features(self, df: pd.DataFrame, velocity: dict) -> pd.DataFrame:
        '''Apply feature engineering to match training'''

        # Transaction amount features
        df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
        df['TransactionAmt_decimal'] = df['TransactionAmt'] % 1
        df['TransactionAmt_is_round'] = (df['TransactionAmt'] % 1 == 0).astype(int)

        # Bin transaction amount
        bins = [0, 50, 100, 200, 500, 1000, 2000, 5000, float('inf')]
        df['TransactionAmt_bin'] = pd.cut(df['TransactionAmt'], bins=bins, labels=False)

        # Email features
        email_col = df.get('P_emaildomain', pd.Series(['']))
        df['email_is_gmail'] = (email_col == 'gmail.com').astype(int)
        df['email_is_yahoo'] = (email_col == 'yahoo.com').astype(int)
        df['email_is_hotmail'] = (email_col == 'hotmail.com').astype(int)

        # Velocity features from Redis
        df['card1_txn_count'] = velocity.get('txn_count_recent', 0)
        df['time_since_last_txn'] = velocity.get('time_since_last', 0)
        df['rapid_txn_flag'] = velocity.get('rapid_txn_flag', 0)
        df['amount_zscore_card'] = 0 # Would need card history to calculate
        df['email_fraud_rate'] = 0 # Would need historical data

        # Time features (default to 0 if not provided)
        df['Transaction_hour'] = 12 # Default to noon
        df['Transaction_day'] = 3 # Default to Wednesday
        df['Transaction_is_night'] = 0
        df['Transaction_is_weekend'] = 0

        return df
    
    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Ensure DataFrame has exact features model expects, in correct order'''

        # Add missing columns with 0
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0

        # Return columns in exact order model expects
        return df[self.feature_names]
    
    def _get_risk_level(self, prob: float) -> str:
        if prob >= 0.8:
            return "High"
        elif prob >= 0.5:
            return "Medium"
        return "Low"
