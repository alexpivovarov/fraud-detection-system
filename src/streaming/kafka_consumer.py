'''
Kafka Consumer - reads transactions from Kafka and processes them
'''

import json
from kafka import KafkaConsumer

from redis_client import FraudRedisClient

def create_consumer(topic: str = 'transactions'):
    '''Create Kafka consumer.'''
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers = ['localhost:9092'],
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        auto_offset_reset = 'earliest', # Start from beginning
        group_id = 'fraud-detection-group'
    )
    return consumer

def consume_transactions():
    '''
    Consume transactions from Kafka and process them.
    '''
    consumer = create_consumer()
    redis_client = FraudRedisClient()

    print("Listening for transactions...")

    for message in consumer: # the consumer waits for messages from Kafka. Each time a new transaction arrives, the loop runs once
        transaction = message.value # extracting transaction dict
        card_id = str(transaction.get('card1', 'unknown')) # get the cardID or 'unknown' if missing

        # Update Redis
        profile = redis_client.update_card_profile(card_id, transaction)
        redis_client.add_transaction_to_history(card_id, transaction)


        print(f"Received: TransactionID={transaction.get('TransactionID')}, "
              f"Amount=${transaction.get('TransactionAmt', 0):.2f},"
              f"Card transaction count={profile['transaction_count']}")
        
if __name__ == "__main__":
    consume_transactions()