import torch
import torch.nn as nn

class DPRLoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, queries: torch.tensor, documents: torch.tensor):
        """
        @param queries: batch_size x embedding_dim
        @param documents: batch_size x embedding_dim

        Computes cross entropy between the query 

        Returns: A 0d torch tensor 
        """
        scores = (queries @ documents.T) / self.temperature
        
        labels = torch.arange(queries.shape[0], device=scores.device)
        return nn.functional.cross_entropy(scores, labels)

if __name__ == "__main__":
    # Verification Check
    B = 64
    D = 128
    torch.manual_seed(42)

    # Initialize the loss function
    criterion = DPRLoss(temperature=1.0)

    # Create dummy tensors
    dummy_queries = torch.randn(B, D)
    dummy_passages = torch.randn(B, D)

    # Calculate loss
    scalar_loss = criterion(dummy_queries, dummy_passages)

    print(f"Computed loss: {scalar_loss.item():.4f}")
    print(f"Loss shape: {scalar_loss.shape}")
    assert scalar_loss.ndim == 0, "Loss must be a scalar (0-dim tensor)!"
    print("Sprint 1 Verification passed successfully.")