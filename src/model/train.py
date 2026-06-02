import torch
import torch.nn as nn
import torch.optim as optim
from src.data_engineering.preprocess import generate_synthetic_data, prepare_tensors
from src.model.two_tower_model import TwoTowerModel

def train_pipeline():
    # 1. Setup constants and configuration
    NUM_USERS = 100
    NUM_ITEMS = 50
    EPOCHS = 10
    
    # 2. Fetch and prepare data
    df = generate_synthetic_data(num_users=NUM_USERS, num_items=NUM_ITEMS)
    user_t, item_t, num_feats, labels = prepare_tensors(df)
    
    # 3. Initialize Model, Loss Function, and Optimizer
    model = TwoTowerModel(num_users=NUM_USERS, num_items=NUM_ITEMS)
    criterion = nn.BCELoss() # Binary Cross Entropy Loss for 0/1 classification
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("Starting training loop...\n")
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass: pass data through the model to get recommendations
        predictions = model(user_t, item_t, num_feats)
        
        # Calculate loss (how wrong were the recommendations?)
        loss = criterion(predictions, labels)
        
        # Backward pass: calculate weights adjustments
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{EPOCHS} Finished | System Loss: {loss.item():.4f}")
        
    # 5. Save your trained model parameters to the models directory
    torch.save(model.state_dict(), "models/recommendation_model.pth")
    print("\nModel trained successfully and saved to 'models/recommendation_model.pth'")

if __name__ == "__main__":
    train_pipeline()