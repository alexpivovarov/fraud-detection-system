'''
Graph analysis for fraud detection
'''

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict

class FraudGraphAnalyzer:
    '''
    Builds a graph of relationships between entities (cards, emails, addresses) to detect fraud networks
    '''

    def __init__(self):
        self.graph = nx.Graph() # creates an empty graph structure to store connections
        self.fraud_nodes = set() # creates an empty set to track which cards are known fraudsters

    def build_graph(self, df: pd.DataFrame):
        '''
        Build graph from transaction data.
        Nodes: cards, emails, addresses
        Edges: connection when they appear in same transaction
        '''
        print("Building fraud graph...")

        for idx, row in df.iterrows(): # loops through each row of the DataFrame
            card = f"card_{row.get('card1', 'unknown')}" # get the card1 value, with a fallback (unknown is default if 'card1' is None)

            # Connect card to email
            if pd.notna(row.get('P_emaildomain')):
                email = f"email_{row['P_emaildomain']}"
                self.graph.add_edge(card, email)

            # Connect card to address
            if pd.notna(row.get('addr1')):
                addr = f"addr_{int(row['addr1'])}" # create prefixed string address node name
                self.graph.add_edge(card, addr) 
            
            # Connect card to device
            if pd.notna(row.get('DeviceType')):
                device = f"device_{row['DeviceType']}"
                self.graph.add_edge(card, device)

            # Track fraud nodes
            if row.get('isFraud') == 1:
                self.fraud_nodes.add(card) # add the node to the set of known fraudsters

            if idx % 100000 == 0:
                print(f" Processed {idx} transactions ...")

            
        print(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")    
        print(f"Known fraud nodes: {len(self.fraud_nodes)}")
        
    def get_graph_features(self, card_id: str) -> dict:
        '''
        Calculate graph-based features for a card
        '''
        card_node = f"card_{card_id}"

        if card_node not in self.graph:
            return {
                'degree': 0,
                'fraud_neighbor_count': 0,
                'fraud_neighbor_ratio': 0,
                'shared_email_cards': 0,
                'shared_addr_cards': 0,
            }
            
        # Degree: how many connections does this card have?
        degree = self.graph.degree(card_node)

        # Get all neighbors (emails, addresses, devices connected to this card)
        neighbors = set(self.graph.neighbors(card_node))

        # Find other cards connected through shared entities
        connected_cards = set()
        shared_email_cards = 0
        shared_addr_cards = 0

        for neighbor in neighbors:
            neighbor_connections = set(self.graph.neighbors(neighbor))
            cards = {n for n in neighbor_connections if n.startswith('card_')}
            connected_cards.update(cards)

            if neighbor.startswith('email'):
                shared_email_cards += len(cards) - 1 # Exclude self
            elif neighbor.startswith('addr_'):
                shared_addr_cards += len(cards) - 1

        connected_cards.discard(card_node) # Remove self

        # How many connected cards are fraudulent?
        fraud_neighbors = connected_cards.intersection(self.fraud_nodes)
        fraud_neighbor_count = len(fraud_neighbors)
        fraud_neighbor_ratio = fraud_neighbor_count / len(connected_cards) if connected_cards else 0

        return {
            'degree': degree,
            'fraud_neighbor_count': fraud_neighbor_count,
            'fraud_neighbor_ratio': fraud_neighbor_ratio,
            'shared_email_cards': shared_email_cards,
            'shared_addr_cards': shared_addr_cards
        }
        
if __name__ == "__main__":
    DATA_PATH = "/Users/alexpivovarov/fraud-detection-system/data/raw/train_transaction.csv"

    print("Loading data ...")
    df = pd.read_csv(DATA_PATH, nrows=50000) # Start with 50k for testing

    analyzer = FraudGraphAnalyzer()
    analyzer.build_graph(df)

    # Test on a few cards
    test_cards = df['card1'].head(5).tolist()
    for card in test_cards:
        features = analyzer.get_graph_features(str(card))
        print(f"\nCard {card}: {features}")
