'''
Kafka Consumer - reads transactions from Kafka and processes them
'''

import json
import joblib
import pandas as pd
from kafka import KafkaConsumer
from redis_client import FraudRedisClient

MODEL_PATH = "/Users/alexpivovarov/fraud-detection-system/src/models/xgboost_model.pkl"
MAPPINGS_PATH = "/Users/alexpivovarov/fraud-detection-system/src/models/category_mappings.pkl"

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

def prepare_features(transaction: dict, feature_names: list, category_mappings: dict) -> pd.DataFrame:
    '''
    Prepare features to match model's expected output
    '''
    features = {name: 0 for name in feature_names}

    for key, value in transaction.items():
        if key in features:
            if pd.isna(value):
                features[key] = 0
            elif isinstance(value, str):
                features[key] = category_mappings[key].get(value, -1) # use saved mapping, -1 for unseen values
            else:
                features[key] = value

    return pd.DataFrame([features])

def consume_transactions():
    '''
    Consume transactions from Kafka and process them.
    '''
    consumer = create_consumer()
    redis_client = FraudRedisClient()

    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    category_mappings = joblib.load(MAPPINGS_PATH)
    feature_names = model.get_booster().feature_names
    print(f"Model loaded! Expects {len(feature_names)} features")


    print("Listening for transactions...")

    fraud_count = 0
    total_count = 0

    for message in consumer: # the consumer waits for messages from Kafka. Each time a new transaction arrives, the loop runs once
        transaction = message.value # extracting transaction dict
        card_id = str(transaction.get('card1', 'unknown')) # get the cardID or 'unknown' if missing

        # Update Redis
        profile = redis_client.update_card_profile(card_id, transaction)
        redis_client.add_transaction_to_history(card_id, transaction)

        total_count += 1

        # Make prediciton
        try:
            features = prepare_features(transaction, feature_names, category_mappings)
            probability = model.predict_proba(features)[0][1]

            if probability > 0.7: # Our tuned threshold
                fraud_count += 1
                print(f"Fraud! TransactionID={transaction.get('TransactionID')},"
                      f"Amount=${transaction.get('TransactionAmt', 0):.2f}, "
                      f"Prob={probability:.2%}")
            elif total_count % 500 == 0:
                print(f"Processed {total_count}, {fraud_count} fraud alerts")
        except Exception as e:
            if total_count % 1000 == 0:
                print(f"Error: {e}")
        
if __name__ == "__main__":
    consume_transactions()