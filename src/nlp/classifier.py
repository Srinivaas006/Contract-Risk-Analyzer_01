"""
Legal Clause Classifier - Week 2: Advanced NLP & Fine-Tuning
A transformer-based neural network head for classifying legal contract clauses.
Built on top of RoBERTa (or any HuggingFace model) via fine-tuning on CUAD.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple, Optional
import os


class LegalClauseClassifier(nn.Module):
    """
    Fine-tunable legal clause classifier.
    Architecture: Pre-trained Transformer Encoder → [CLS] pooling → MLP Head → Labels

    Usage:
        model = LegalClauseClassifier(model_name="roberta-base", num_labels=2)
        outputs = model(input_ids, attention_mask)   # Returns raw logits
    """

    def __init__(self, model_name: str = "roberta-base", num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels

        # Load the pre-trained encoder backbone
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for roberta-base

        # Classification head: maps [CLS] embedding → label logits
        self.classifier_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            input_ids:      (batch_size, seq_len) tokenized input
            attention_mask: (batch_size, seq_len) 1 for real tokens, 0 for padding
        Returns:
            logits: (batch_size, num_labels) raw unnormalized scores
        """
        # Run the transformer encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Extract [CLS] token representation (first token = sentence summary)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (batch, hidden_size)

        # Pass through classification head
        logits = self.classifier_head(cls_embedding)
        return logits

    def predict_proba(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get softmax probability scores (for inference, no grad)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=-1)
        return probabilities

    def predict_text(self, text: str, tokenizer, device: str = "cpu", max_len: int = 256) -> dict:
        """
        Convenience method: classify a raw text string.
        Returns dict with label, confidence, and probabilities.
        """
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        probs = self.predict_proba(input_ids, attention_mask)
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_class].item()

        label_map = {0: "BENIGN", 1: "RISKY_CLAUSE"}

        return {
            "label": label_map.get(predicted_class, str(predicted_class)),
            "confidence": round(confidence, 4),
            "probabilities": {
                "BENIGN": round(probs[0][0].item(), 4),
                "RISKY_CLAUSE": round(probs[0][1].item(), 4),
            }
        }

    def save(self, path: str):
        """Save model weights to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"✅ Model saved to: {path}")

    @classmethod
    def load(cls, path: str, model_name: str = "roberta-base", num_labels: int = 2):
        """Load model weights from disk."""
        model = cls(model_name=model_name, num_labels=num_labels)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        print(f"✅ Model loaded from: {path}")
        return model


if __name__ == "__main__":
    print("Testing LegalClauseClassifier...")
    model = LegalClauseClassifier(model_name="roberta-base", num_labels=2)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")

    # Quick forward pass test
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    result = model.predict_text(
        "This agreement shall automatically terminate if either party files for bankruptcy.",
        tokenizer=tokenizer
    )
    print(f"\nTest prediction: {result}")
    print("✅ LegalClauseClassifier working correctly!")
