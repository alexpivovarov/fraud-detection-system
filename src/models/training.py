'''
Model training for Fraud Detection
'''

import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path
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