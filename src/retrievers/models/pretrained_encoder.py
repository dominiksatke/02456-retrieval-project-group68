import torch
import torch.nn as nn
from transformers import AutoModel
from src.retrievers.models.pooling import masked_pool

class PretrainedEncoder(nn.Module):
    """ Wraps a pretrained Hugging Face encoder (e.g. bert-base-uncased or answerdotai/ModernBERT-base)
    so it has the same call and output as the scratch TransformerEncoder: 
    encoder(input_ids, attention_mask) -> batch_size x emb_dim (not normalised) emb_dim is the backbone's own size, 768 for bert-base."""

    def __init__(self, model_name: str = 'bert-base-uncased', pooling_type: str = 'cls', model: nn.Module = None):
        super().__init__()
        #Eager attention so attention maps can be returned for the visualisations
        self.model = model if model is not None else AutoModel.from_pretrained(model_name, attn_implementation='eager')
        self.pooling_type = pooling_type
        self.emb_dim = self.model.config.hidden_size

    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, return_attention: bool = False):
        """ @param input_ids: batch_size x sequence_length, integer
        @param attention_mask: batch_size x sequence_length, 1 = token, 0 = padding
        @param return_attention: Also return one batch_size x n_heads x sequence_length x sequence_length map per layer. """
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=return_attention)
        pooled = masked_pool(outputs.last_hidden_state, attention_mask, self.pooling_type)
        if return_attention:
            return pooled, list(outputs.attentions)
        return pooled