"""
Exploratory data Analysis

After running the script:

1. Data shape and structure

2. Fraud distribution (class imbalance)

3. Feature types and missing values

4. Key patterns in fraudulent vs legitimate transactions


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))
from src.utils.data_loader import (
    load_ieee_cis_data,
    get_feature_types,
    calculate_missing_stats,
    reduce_memory_usage
)
from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

def run_eda():
    '''Run complete exploratory data analysis'''

    print("FRAUD DETECTION - EXPLORATORY DATA ANALYSIS")

    # =======================
    # 1. Load data
    # =======================

    print("\n[1/6] Loading data ...")

    try:
        train_df, _ = load_ieee_cis_data()
    except FileNotFoundError:
        print("\n Dataset not found!")


    # Reduce memory
    train_df = reduce_memory_usage(train_df)


    # ===============
    # 2 Basic statistics
    # ===============
    print("\n[2/6] Basic statistics")
    print(f"Total transactions: {len{train_df}:,}")
    print(f"Total features: {len(train_df.columns)}")
    print(f"Memory usage: {train_df.memory_usage().sum() / 1024**2:.1f} MB")

    # ========
    # 3. Class Imbalance
    # ========
    print("\n[3/6] Class Distribution (Target: isFraud)")
    print("-" * 40)

    fraud_counts = train_df['isFraud'].value_counts()
    fraud_pct = train_df['isFraud'].value_counts(normalize=True) * 100

    print(f"Legitimate (0): {fraud_counts[0]:,} ({fraud_pct[0]:.2f}%)")
    print(f"Fraudulent (0): {fraud_counts[1]:,} ({fraud_pct[1]:.2f}%)")
    print(f"Imbalance ratio: 1:{fraud_counts[0] // fraud_counts[1]}")

    # Plot class distribution
    fig, axes = plt.subplots(1, 2, figsize=(12,4))

    # Bar plot
    ax1 = axes[0]
    bars = ax1.bar(['Legitimate', 'Fraudulent'], fraud_counts.values, color=['#2ecc71', '#e74c3c'])

    ax1.set_ylabel('Count')
    ax1.set_title('Transaction Class Distribution')
    for bar, count in zip(bars, fraud_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, f'{count:,}', ha='center', va='bottom')

    # Pie chart
    ax2 = axes[0]
    ax2.pie(fraud_counts.values, labels=['Legitimate', 'Fraudulent'],
            autopct='%1.2f%%', colors=['#2ecc71', '#e74c3c'],
            explode = [0, 0.1])
    ax2.set_title('Class Imbalance Visualization')

    plt.tight_layout()
    plt.savefig(PROCESSED_DATA_DIR / 'class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

    #===========
    # 4. Feature Analysis
    #===========
    print("\n[4/6] Featrue Analysis")

    feature_types = get_feature_types(train_df)
    



