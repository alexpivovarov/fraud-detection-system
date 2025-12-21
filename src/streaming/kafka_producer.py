'''
Kafka Producer - streams tranasctions to Kafka
'''

import json
import time
import pandas as pd
from kafka import KafkaProducer
from pathlib import Path

def create_producer():
    '''Create Kafka producer.'''
    producer = KafkaProducer(
        bootstrap_servers = ['localhost:9092'],
        value_serializer=lambda x: json.dumps(x).encode('utf-8') # json.dumps(x) - Python dict -> JSON string; .encode('utf-8') - string -> bytes
    )
    return producer

def stream_transactions(csv_path: str, topic: str = 'transactions', delay: float=0.1):
    '''
    Stream transactions from CSV to Kafka.

    Args:
        csv_path: Path to tranasction CSV
        topic: Kafka topic name
        delay: Seconds between messages (simulates real-time)
    '''

    producer = create_producer()

    print(f"Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path)

    print(f"Streaming {len(df)} transactions to topic '{topic}' ... ")

    for idx, row in df.iterrows():
        # Convert row to dictionary
        transaction = row.to_dict()

        # Send to Kafka
        producer.send(topic, value=transaction)

        if idx % 1000 == 0: # prints progress every 1000 transactions
            print(f"Sent {idx} transactions ...")

        time.sleep(delay) #Pauses for delay seconds between each transaction. Simulates real-time flow instead of blasting everything instantly.
    
    producer.flush() # Kafka batches messages for efficiency. flush() forces all pending messages to be sent before the program exits. Without it, some transactions might be lost.

    print("Streaming complete!")

if __name__ == "__main__":
    DATA_PATH = "/Users/alexpivovarov/fraud-detection-system/data/raw/train_transaction.csv"

    # Stream with 0.01 second delay (100 transactions/second)
    stream_transactions(DATA_PATH, delay=0.01)