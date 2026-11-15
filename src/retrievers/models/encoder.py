import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from src.retrievers.models.pooling import masked_pool

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers: int, n_heads: int, emb_dim: int, hidden_dim: int, feed_forward_dim: int, pooling_type: str = 'mean'):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.emb_dim =    emb_dim
        self.hidden_dim = hidden_dim
        self.feed_forward_dim = feed_forward_dim
        self.pooling_type = pooling_type
        self.setup_model()

    def setup_model(self):
        """
        Instantiate the model.
        """
        self.model = None
        raise NotImplementedError()

    def pool(self, x, attention_mask=None):
        """
        Given a batch_size x sequence_length x emb_dim tensor, pool the input along the sequence_length dimension
        @param attention_mask: batch_size x sequence_length, 1 = real token, 0 = padding (None = no padding)
        Returns: A pooled vector of shape batch_size x emb_dim
        """
        # Reads the pooling type from self.pooling_type, as main's encoder already does.
        # Without the mask, mean and max would mix padding into the result.
        return masked_pool(x, attention_mask, self.pooling_type)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Run a forward pass on the model
        """
        raise NotImplementedError()
