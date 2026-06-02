import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from src.nlp.classifier import LegalClauseClassifier

# 1. Create a custom Dataset processor for legal text strings
class LegalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize inputs according to transformer specifications
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward Pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Compute Loss
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()
        
        # Backward Pass & Parameter Tuning
        loss.backward()
        optimizer.step()
        
    return total_loss / len(data_loader)

if __name__ == "__main__":
    print("Setting up Week 2 Transformer Fine-Tuning Loop...")
    
    # Check if CPU or GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using processing device target: {device}")
    
    # Mock data representing processed sections from the CUAD dataset
    # 1 = Contains critical/risky legal clause, 0 = Generic text segment
    sample_texts = [
        "This agreement shall automatically terminate if either party files for bankruptcy.",
        "The parties agree to meet annually in London to review operational performance frameworks.",
        "In no event shall total liability exceed the aggregate fees paid under this statement of work.",
        "The standard office hours of operation are Monday through Friday, 9 AM to 5 PM."
    ]
    sample_labels = [1, 0, 1, 0] 
    
    # Initialize tokenizer and dataset handlers
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    dataset = LegalDataset(sample_texts, sample_labels, tokenizer)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Initialize our legal architecture model
    model = LegalClauseClassifier(model_name="roberta-base", num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()
    
    # Run a quick validation training loop pass
    print("Executing prototype training epoch step...")
    epoch_loss = train_epoch(model, data_loader, loss_fn, optimizer, device)
    print(f"Epoch Complete! Calculation Baseline Loss: {epoch_loss:.4f}")
    
    # Create models directory and save checkpoint if it passes
    torch.save(model.state_dict(), "models/legal_classifier.pth")
    print("Model checkpoint weights saved to 'models/legal_classifier.pth' successfully!")