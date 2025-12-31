import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from src.streaming.redis_client import FraudRedisClient


redis_client = FraudRedisClient()


# Load some real transactions

df = pd.read_csv('data/raw/train_transaction.csv', nrows=10000)

for _, row in df.iterrows():
    card_id = str(row['card1'])
    transaction = {
        'TransactionAmt': row['TransactionAmt'],
        'TransactionID': row['TransactionID']
    }
    redis_client.update_card_profile(card_id, transaction)

print("Redis seeded with 10,000 transactions")