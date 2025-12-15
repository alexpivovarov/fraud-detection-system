"""
Data loading and preprocessing utilities
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, RANDOM_STATE

def load_ieee_cis_data(sample_frac: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load IEEE-CIS fraud detection dataset

    Args:
        sample_frac: if provided, sample this fraction of data

    Returns:
        train_df, test_df: Training and test dataframes
    """
    print("Loading transaction data...")
    train_transaction = pd.read_csv(RAW_DATA_DIR / "train_transaction.csv")
    train_identity = pd.read_csv(RAW_DATA_DIR / "")

    #Merge transaction with identity data
    #Use 'left' join because not all transactions have identity info
    train_df = train_transaction.merge(train_identity, on="TransactionID", how="left")

    print(f"Loaded {len(train_df):,} transactions")
    print(f"Fraud rate: {train_df["isFraud"].mean():.2%}")

    # Check for test data
    test_transaction_path = RAW_DATA_DIR / "test_transaction.csv"
    if test_transaction_path.exists():
        test_transaction = pd.read_csv(test_transaction_path)
        test_identity = pd.read_csv(RAW_DATA_DIR / "test_identity.csv")
        test_df = test_transaction.merge(test_identity, on="TransactionID", how="left")
    else:
        test_df = None

    return train_df, test_df


def get_feature_types(df: pd.DataFrame) -> dict:
    """
    Categorize features by type.
    Returns:
        Dictionary with "numeric", "categorical", "binary" feature lists
    """
    numeric_features = []
    categorical_features = []
    binary_features = []

    for col in df.columns:
        if col in ["TransactionID", "isFraud"]:
            continue

        n_unique = df[col].nunique()

        if n_unique == 2:
            binary_features.append(col)
        elif df[col].dtype in ["object", "category"] or n_unique < 50:
            categorical_features.append(col)
        else:
            numeric_features.append(col)

    return {
        "numeric": numeric_features,
        "categorical": categorical_features,
        "binary": binary_features
    }

def calculate_missing_stats(df: pd.DataFrame) -> pd.DataFrame:
    '''Calculate missing value statistics for each column
        Returns: DataFrame with missing counts and percentages, sorted by most missing
    '''
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    stats = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct,
        "dtype": df.types
    })

    return stats[stats["missing_count"] > 0].sort_values("missing_pct", ascending=False)

def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    '''
    Reduce memory usage by downcasting numeric types.
    Essential for handling large datasets.
    '''
    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtype

        # Skipping non-numeric columns (strings, objects)
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            # Downcast integers
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)

            # Downcast floats
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024 ** 2

    if verbose:
        print(f"Memory usage: {start_mem:.2f} MB -> {end_mem:.2f} MB({100 * (start_mem - end_mem) / start_mem:.1f} % reduction)")
    
    return df