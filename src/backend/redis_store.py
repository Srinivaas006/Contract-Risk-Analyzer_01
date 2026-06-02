import numpy as np

class MockRedisStore:
    def __init__(self):
        # Store pre-calculated mathematical embeddings for our 50 items
        self.store = {}
        self.generate_mock_item_vectors()

    def generate_mock_item_vectors(self, num_items=50, embedding_dim=16):
        """Generates and stores static vectors for items."""
        np.random.seed(42)
        for item_id in range(num_items):
            # Create a random vector for each item representing its characteristics
            vector = np.random.uniform(-1, 1, embedding_dim).astype(np.float32)
            self.store[f"item:{item_id}"] = vector

    def get_item_vector(self, item_id):
        """Retrieves an item vector instantly."""
        return self.store.get(f"item:{item_id}")

    def get_all_items(self):
        """Returns all items currently stored."""
        return [int(k.split(":")[1]) for k in self.store.keys()]

# Initialize a single shared instance of our store
kv_database = MockRedisStore()