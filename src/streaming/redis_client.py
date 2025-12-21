'''
Redis client for real-time feature storage
'''
import redis
import json
from typing import Dict, Optional

class FraudRedisClient:
    '''
    Redis client for storing card profiles and transaction history.
    '''

    def __init__(self, host: str = 'localhost', port: int=6379):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        print(f"Connected to Redis at {host}:{port}")


    def update_card_profile(self, card_id: str, transaction: Dict):
        '''
        Update card profile with new transaction
        '''
        key = f"card:{card_id}"

        # Get existing profile or create new
        profile = self.get_card_profile(card_id) or {
            'transaction_count': 0,
            'total_amount': 0,
            'avg_amount': 0,
            'last_transaction_time': 0
        }

        # Update stats
        profile['transaction_count'] += 1
        profile['total_amount'] += transaction.get('TransactionAmt', 0)
        profile['avg_amount'] = profile['total_amount'] / profile['transaction_count']
        profile['last_transaction_time'] = transaction.get('TransactionDT', 0)

        # Save to Redis
        self.client.set(key, json.dumps(profile))

        return profile
    
    def get_card_profile(self, card_id: str) -> Optional[Dict]:
        '''
        Get card profile from Redis
        '''
        key = f"card:{card_id}"
        data = self.client.get(key)

        if data:
            return json.loads(data)
        return None
    
    def add_transaction_to_history(self, card_id: str, transaction: Dict):
        '''
        Add transaction to card's recent history (keeps last 10)
        '''
        key = f"card:{card_id}:history"

        # Add to list (left push)
        self.client.lpush(key, json.dumps(transaction))

        # Keep only last 10 transactions
        self.client.ltrim(key, 0, 9)
    
    def get_transaction_history(self, card_id: str, count: int=10):
        '''
        Get recent transactions for a card
        '''
        key = f"card:{card_id}:history"
        history = self.client.lrange(key, 0, count - 1)

        return [json.loads(t) for t in history]
    
    def get_velocity_features(self, card_id: str) -> Dict:
        '''
        Calcualte velocity features from recent history
        '''

        history = self.get_transaction_history(card_id)
        profile = self.get_card_profile(card_id)

        if not history:
            return {
                'txn_count_recent': 0,
                "avg_amount_recent": 0,
                'time_since_last': 999999
            }
        
        amounts = [t.get('TransactionAmt', 0) for t in history]

        return {
            'txn_count_recent': len(history),
            'avg_amount_recent': sum(amounts) / len(amounts) if amounts else 0,
            'total_card_txns': profile['transaction_count'] if profile else 0
        }
    
if __name__ == "__main__":
    # Test the Redis client
    client = FraudRedisClient()

    #Test transaction
    test_txn = {
        'TransactionID': 12345,
        'TransactionAmt': 150.00,
        'TransactionDT': 1000000,
        'card1': 4567
    }

    card_id = str(test_txn['card1'])

    # Update profile
    profile = client.update_card_profile(card_id, test_txn)
    print(f"Card profile: {profile}")

    # Add to history
    client.add_transaction_to_history(card_id, test_txn)

    # Ger velocity features
    velocity = client.get_velocity_features(card_id)
    print(f"Velocity features {velocity}")