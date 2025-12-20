'''
Feature engineering for Fraud Detection
'''
import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import VELOCITY_WINDOWS, RANDOM_STATE

#=====
# Section 1: Imports and Class setup
#====

class FraudFeatureEngineer:
    '''
    Feature engineering pipeline for fraud detection.

    Usage:
        engineer = FraudFeatureEngineer()
        df = engineer.fit_transform(train_df)
    '''

    def __init__(self):
        self.card_stats = {}
        self.email_stats = {}
        self.fitted = False

    #=====
    # Section 2: Fit and transform methods
    #====


    def fit(self, df: pd.DataFrame) -> 'FraudFeatureEngineer':
        '''Learn statistics from training data'''
        print("Fitting feature engineer...")

        # Learn card-level statistics (average amont per card)
        self.card_stats = df.groupby('card1').agg({
            'TransactionAmt': ['mean', 'std', 'median']    
        }).to_dict()

        # Learn email domain risk scores
        if 'P_emaildomain' in df.columns:
            email_fraud = df.groupby('P_emaildomain')['isFraud'].mean()
            self.email_stats['P_emaildomain_risk'] = email_fraud.to_dict()

        self.fitted = True
        print("Feature engineer fitted.")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Apply feature engineering to dataframe.'''
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
        
        df = df.copy()

        print("Engineering features...")
        df = self._add_transaction_features(df)
        df = self._add_time_features(df)
        df = self._add_card_features(df)
        df = self._add_email_features(df)
        df = self._add_velocity_features(df)

        print(f"Total features after engineering: {len(df.columns)}")
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Fit and transform in one step.'''
        return self.fit(df).transform(df)
    

    #=======
    # Section 3 'Transaction Features"
    #======

    def _add_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Basic transaction amount transfromations.'''
        print(" Adding transaction features ...")

        # Log transform (handles skewness)
        df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])

        # Amount bins
        df['TransactionAmt_bin'] = pd.cut(
            df['TransactionAmt'],
            bins = [0, 50, 100, 250, 500, 1000, 5000, np.inf],
            labels = [0, 1, 2, 3, 4, 5, 6]
        ).astype(float)

        # Decimal part (fraud often uses round numbers)
        df['TransactionAmt_decimal'] = df['TransactionAmt'] - df['TransactionAmt'].astype(int)
        df['TransactionAmt_is_round'] = (df['TransactionAmt_decimal'] == 0).astype(int)

        return df
    
    #=====
    # Section 4: Time features
    #=====
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Extract temporal patterns.'''
        print(" Adding time features ...")

        if 'TransactionDT' not in df.columns:
            return df
        
        # TransactionDT is seconds from a reference point
        df['Transaction_hour'] = (df['TransactionDT'] // 3600) % 24 # Hour of day (0-23)
        df['Transaction_day'] = (df['TransactionDT'] // 86400) % 7 # Day of week (0-6)

        # Time of day categories
        df['Transaction_is_night'] = ((df['Transaction_hour'] >= 22) | 
                                      (df['Transaction_hour'] <= 5)).astype(int)
        df['Transaction_is_weekend'] = (df['Transaction_day'] >= 5).astype(int)

        return df
    
    #====
    # Section 5: Card Features
    #====
    def _add_card_features(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Card-based features and deviation from card history.'''
        print(" Adding card features ...")
        
        if 'card1' not in df.columns:
            return df
        
        # Amount deviation from card's typical behaviour
        card_mean = df['card1'].map(self.card_stats.get(('TransactionAmt', 'mean'), {}))
        card_std = df['card1'].map(self.card_stats.get(('TransactionAmt', 'std'), {}))

        # Z-score: how unusual is this amount for this card?
        df['amount_zscore_card'] = (df['TransactionAmt'] - card_mean) / (card_std + 1)
        df['amount_zscore_card'] = df['amount_zscore_card'].fillna(0)

        return df


    #====
    # Section 6: Email Features
    #====
    def _add_email_features(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Email domain risk features'''
        print("Adding email features...")

        if 'P_emaildomain' not in df.columns:
            return df
        
        # Email domain risk score from training data
        if 'P_emaildomain_risk' in self.email_stats:
            df['email_fraud_rate'] = df['P_emaildomain'].map(self.email_stats['P_emaildomain_risk']).fillna(0.035) # Default to overall fraud rate


        # Email provider features
        df['email_is_gmail'] = df['P_emaildomain'].str.contains('gmail', na=False).astype(int)
        df['email_is_yahoo'] = df['P_emaildomain'].str.contains('yahoo', na=False).astype(int)
        df['email_is_hotmail'] = df['P_emaildomain'].str.contains('hotmail', na=False).astype(int)

        return df
    
    #=====
    # Section 7: Velocity Features
    #=====

    def _add_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Transaction velocity - counts in time windows'''
        print(" Adding velocity features ...")

        if 'TransactionDT' not in df.columns or 'card1' not in df.columns:
            return df
        
        # Sort by time
        df = df.sort_values('TransactionDT').reset_index(drop=True)

        # Time since last transaction (same card)
        df['time_since_last_txn'] = df.groupby('card1')['TransactionDT'].diff() # calculates time snce last transaction
        df['time_since_last_txn'] = df['time_since_last_txn'].fillna(999999) # the first transaction for each card has no previous one, we fill diff with "long time ago"

        # Rapid succession flag (within 1 minute)
        df['rapid_txn_flag'] = (df['time_since_last_txn'] < 60).astype(int)

        # Transaction count per card
        df['card1_txn_count'] = df.groupby('card1')['TransactionID'].transform('count')

        return df
    

    #===
    # Section 8: Prepare for the model
    #===

    def prepare_features_for_model(self, df: pd.DataFrame, target_col: str = 'isFraud') -> Tuple[pd.DataFrame, pd.Series]: # takes a DataFrame and returns the feauture matrix (X) and the target variable (Y)
        """
        Final preparation: prepare dataset for model training by cleaning and formatting the features.
        """
        # Remove columns that shouldn't be used as features 
        exclude_cols = ['TransactionID', 'TransactionDT', target_col]

        # Keep only numeric columns
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols].copy() # feature matrix
        Y = df[target_col].copy() if target_col in df.columns else None # target vector Y

        # ML models need numeric input, so this converts text columns into integer codes
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype('category').cat.codes

        # Handling infinity values
        X = X.replace([np.inf, -np.inf], 0) # First convert inf to NaN
        X = X.fillna(X.median()) # Fill all NaN with median

        print(f"Final feature matrix: {X.shape}")

        return X, Y