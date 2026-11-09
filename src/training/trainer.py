import torch
import torch.nn as nn
import torch.optim as optim

from src.training.loss import DPRLoss

class DPRTrainer:
    def __init__(self, encoder: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module):
        self.encoder = encoder
        self.optimizer = optimizer
        self.criterion = criterion
        
    def train_step(self, query_ids: torch.Tensor, document_ids: torch.Tensor) -> float:
        """
        Executes a single forward and backward pass for a batch.
        """
        self.encoder.train()
        self.optimizer.zero_grad()
        
        # Pass token IDs through the encoder to get dense embeddings
        query_embeddings = self.encoder(query_ids)
        document_embeddings = self.encoder(document_ids)
        
        # Calculate loss using the in-batch negatives DPRLoss
        loss = self.criterion(query_embeddings, document_embeddings)
        
        # Backpropagation
        loss.backward()
        self.optimizer.step()
        
        return loss.item()