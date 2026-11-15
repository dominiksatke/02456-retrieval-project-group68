import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.retrievers.models.pooling import masked_pool


class PositionalEncoding(nn.Module):
    """Adds fixed sinusoidal position vectors to token embeddings: [B, L, d] -> [B, L, d]."""
    def __init__(self, emb_dim, max_len):
        super().__init__()
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * -(math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


def attention(Q, K, V, mask=None):
    """Scaled dot-product attention. Q, K, V: [..., L, d]; mask broadcastable to [..., L, L], 0 = ignore."""
    d_h = Q.size(-1)
    assert Q.shape[-1] == K.shape[-1]
    assert K.shape[-2] == V.shape[-2]
    scores = Q @ K.transpose(-2,-1) / math.sqrt(d_h)  # swap only the LAST TWO dimensions → works for [L,d], [B,L,d] and [B,h,L,d]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -math.inf)
    weights = scores.softmax(dim=-1) # last dimension
    out = weights @ V
    return out, weights


class MultiHeadedAttention(nn.Module):
    """Multi-head self-attention: x [B, L, d], mask [B, L] -> (out [B, L, d], weights [B, h, L, L])."""
    def __init__(self, emb_dim: int, n_heads: int):
        super().__init__()
        assert emb_dim % n_heads == 0 #check emb dimension is divisible by number of heads
        self.d_h = emb_dim // n_heads # store the per-head size d_h 128/4 =32
        self.n_heads = n_heads
        self.W_q = nn.Linear(emb_dim, emb_dim) # query, key, value and output projections
        self.W_k = nn.Linear(emb_dim, emb_dim)
        self.W_v = nn.Linear(emb_dim, emb_dim)
        self.W_o = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, mask=None):
        B, L, _ = x.shape # x: [B, L, 128]

        # view: cut each token's 128 numbers into 4 × 32
        # transpose: move "heads" before "tokens" [B, 4, L, 32]
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_h).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_h).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_h).transpose(1, 2)

        if mask is not None:
            # Same mask applied to all heads.
            mask = mask[:, None, None, :]

        # attention runs on all heads in parallel → out [B, h, L, d_h]
        out, weights = attention(Q, K, V, mask=mask)
        # concat heads and mix them with W_o
        out = out.transpose(1, 2).contiguous().view(B, L, self.n_heads * self.d_h)
        out = self.W_o(out)
        return out, weights


class EncoderBlock(nn.Module):
    """One pre-LN Transformer encoder block: x [B, L, d], mask [B, L] -> (x [B, L, d], weights [B, h, L, L])."""
    def __init__(self, emb_dim: int, n_heads: int, feed_forward_dim: int):
        super().__init__()
        self.attention = MultiHeadedAttention(emb_dim, n_heads)
        self.feed_forward = nn.Sequential(nn.Linear(emb_dim, feed_forward_dim), nn.GELU(), nn.Linear(feed_forward_dim, emb_dim))
        self.norm1 = nn.LayerNorm(emb_dim) # before attention
        self.norm2 = nn.LayerNorm(emb_dim) # before FFN

    def forward(self, x, mask=None):
        # sub-layer 1: normalise, attend, add back
        attn_out, weights = self.attention(self.norm1(x), mask)
        x = x + attn_out # residual: add the input back
        # sub-layer 2: normalise, feed-forward, add back
        x = x + self.feed_forward(self.norm2(x))
        return x, weights


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers: int, n_heads: int, emb_dim: int, hidden_dim: int, feed_forward_dim: int,
                 vocab_size: int = 30522, max_len: int = 256, pooling_type: str = 'mean', cfg=None):
        super().__init__()
        if cfg is not None:
            pooling_type = cfg.pooling_type
        assert hidden_dim == emb_dim // n_heads, f"hidden_dim ({hidden_dim}) must equal emb_dim // n_heads ({emb_dim // n_heads})"
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.feed_forward_dim = feed_forward_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pooling_type = pooling_type
        self.setup_model()

    def setup_model(self):
        """Create the token embedding, positional encoding, encoder blocks and final LayerNorm."""
        self.token_embedding = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(self.emb_dim, self.max_len)
        self.blocks = nn.ModuleList([EncoderBlock(self.emb_dim, self.n_heads, self.feed_forward_dim) for _ in range(self.n_layers)])
        self.final_norm = nn.LayerNorm(self.emb_dim)


    def pool(self, x, attention_mask=None):
        """
        Given a batch_size x sequence_length x emb_dim tensor, pool the input along the sequence_length dimension
        @param attention_mask: batch_size x sequence_length, 1 = real token, 0 = padding (None = no padding)
        Returns: A pooled vector of shape batch_size x emb_dim
        """
        # Reads the pooling type from self.pooling_type, as main's encoder already does.
        # Without the mask, mean and max would mix padding into the result.
        return masked_pool(x, attention_mask, self.pooling_type)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, return_attention: bool = False):
        """
        @param input_ids: [B, L] token ids
        @param attention_mask: [B, L], 1 = token, 0 = padding
        Returns: [B, emb_dim]; with return_attention=True also a list of n_layers tensors [B, n_heads, L, L]
        """
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x) #position added
        attention_maps = []
        for block in self.blocks:
            x, weights = block(x, attention_mask)
            attention_maps.append(weights)
        x = self.final_norm(x)
        pooled = self.pool(x, attention_mask)
        if return_attention:
            return pooled, attention_maps
        return pooled
