'''
Kafka Consumer - reads transactions from Kafka and processes them
'''

import sys
import json
import joblib
import pandas as pd
from pathlib import Path
from kafka import KafkaConsumer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.predictor import FraudPredictor
from src.streaming.redis_client import FraudRedisClient

MODEL_BUNDLE_PATH = Path(__file__).parent.parent / "models" / "fraud_model_bundle.pkl"

def create_consumer(topic: str = 'transactions'):
    '''Create Kafka consumer.'''
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers = ['localhost:9092'],
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        auto_offset_reset = 'earliest', # Start from beginning
        group_id = 'fraud-detection-group-v3'
    )
    return consumer

def consume_transactions():
    '''
    Consume transactions from Kafka and process them.
    '''
    consumer = create_consumer()
    redis_client = FraudRedisClient()

    # Load predictor (same as API uses)
    print(f"Loading model from {MODEL_BUNDLE_PATH}...")
    predictor = FraudPredictor(str(MODEL_BUNDLE_PATH))

    print("Listening for transactions...")

    fraud_count = 0
    total_count = 0

    for message in consumer: # the consumer waits for messages from Kafka. Each time a new transaction arrives, the loop runs once
        transaction = message.value # extracting transaction dict
        card_id = str(transaction.get('card1', 'unknown')) # get the cardID or 'unknown' if missing

        # Get velocity features from Redis
        velocity_features = redis_client.get_velocity_features(card_id)

        # Update Redis with this transaction
        profile = redis_client.update_card_profile(card_id, transaction)
        redis_client.add_transaction_to_history(card_id, transaction)

        total_count += 1

        # Make prediciton using predictor
        try:
            result = predictor.predict(transaction, velocity_features)

            if result['is_fraud']:
                fraud_count += 1
                print(f"FRAUD ALERT! TransactionID={transaction.get('TransactionID')},"
                        f"Amount=${transaction.get('TransactionAmt', 0):.2f}, "
                        f"Prob={result['fraud_probability']:.2%}, "
                        f"Risk={result['risk_level']}")
            elif total_count % 500 == 0:
                print(f"Processed {total_count}, {fraud_count} fraud alerts")

        except Exception as e:
            if total_count % 1000 == 0:
                print(f"Error: {e}")
        
if __name__ == "__main__":
    consume_transactions()