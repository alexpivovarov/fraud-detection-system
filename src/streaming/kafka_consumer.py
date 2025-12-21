'''
Kafka Consumer - reads transactions from Kafka and processes them
'''

import json
from kafka import KafkaConsumer

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

    print("Listening for transactions...")

    for message in consumer:
        transaction = message.value

        print(f"Received: TransactionID={transaction.get('TransactionID')}, "
              f"Amount=${transaction.get('TransactionAmt', 0):.2f}")
        
if __name__ == "__main__":
    consume_transactions()