import torch
import torch.nn as nn

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=16):
        super(TwoTowerModel, self).__init__()
        
        # --- USER TOWER ---
        # Converts User IDs into a continuous mathematical vector
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        # Linear layer to blend in continuous user features (like recency and sequences)
        self.user_dense = nn.Linear(embedding_dim + 2, embedding_dim) 
        
        # --- ITEM TOWER ---
        # Converts Item IDs into a continuous mathematical vector
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, user_ids, item_ids, user_features):
        # 1. Process User Tower
        user_emb = self.user_embedding(user_ids)
        # Combine the user id embedding with numerical features (recency/sequence)
        user_combined = torch.cat([user_emb, user_features], dim=1)
        user_vector = self.relu(self.user_dense(user_combined))
        
        # 2. Process Item Tower
        item_vector = self.relu(self.item_embedding(item_ids))
        
        # 3. Compute Similarity Score (Dot Product match between User and Item)
        # We perform a row-by-row multiplication and sum the dimensions
        similarity = torch.sum(user_vector * item_vector, dim=1)
        
        # Use Sigmoid to output a final score between 0.0 (poor match) and 1.0 (perfect recommendation)
        return torch.sigmoid(similarity)

if __name__ == "__main__":
    # Test with mock parameters
    model = TwoTowerModel(num_users=100, num_items=50)
    print("Two-Tower Model Neural Network Architecture created successfully!")