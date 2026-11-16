import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.retrievers.base_retriever import Retriever
from src.retrievers.models.encoder import TransformerEncoder
from src.retrievers.models.pretrained_encoder import PretrainedEncoder
from src.data_utils.tokenization import TextTokenizer


def build_encoder(model_cfg) -> nn.Module:
    """ Builds one (untrained) encoder from the model config.
    model_cfg.type is 'scratch' (our TransformerEncoder) or 'pretrained' (a Hugging Face backbone). """

    if model_cfg.type == 'pretrained':
        return PretrainedEncoder(model_name=model_cfg.model_name, pooling_type=model_cfg.pooling_type)

    if model_cfg.type == 'scratch':
        return TransformerEncoder(n_layers=model_cfg.n_layers, n_heads=model_cfg.n_heads, emb_dim=model_cfg.emb_dim,
                                  hidden_dim=model_cfg.hidden_dim, feed_forward_dim=model_cfg.feed_forward_dim,
                                  vocab_size=model_cfg.vocab_size, max_len=model_cfg.max_len, cfg=model_cfg)

    raise ValueError(f'Unknown model type {model_cfg.type}. Options are scratch | pretrained')


@torch.no_grad()
def encode_texts(encoder: nn.Module, texts: list[str], encode_fn, batch_size: int, device: str) -> torch.tensor:
    """Encodes texts in batches and returns a len(texts) x emb_dim tensor (on cpu), in the same order as texts.
    Texts are processed shortest-first so each batch has little padding, then put back in the original order. """

    was_training = encoder.training
    encoder.eval()

    lengths = [len(text) for text in texts]
    order = sorted(range(len(texts)), key=lambda i: lengths[i])
    vectors = [None] * len(texts)

    for start in range(0, len(texts), batch_size):
        batch_indices = order[start:start + batch_size]
        input_ids, attention_mask = encode_fn([texts[i] for i in batch_indices])
        batch_vectors = encoder(input_ids.to(device), attention_mask.to(device)).float().cpu()
        for i, vector in zip(batch_indices, batch_vectors):
            vectors[i] = vector

    encoder.train(was_training)
    return torch.stack(vectors)


class DPR(Retriever):
    """
    Dense retriever. Every passage is encoded once into a vector (the index); a query is scored
    against all passages with a dot product, and a higher score means more relevant.
    """
    def __init__(self, cfg, documents: list[dict] = None, index_path: str = None,
                 query_encoder: nn.Module = None, document_encoder: nn.Module = None) -> None:
        """@param cfg: Experiment config with cfg.device and cfg.model.
        @param query_encoder, document_encoder: Pass encoders directly (e.g. during training, for validation).
            If not given, they are built from cfg.model and loaded from cfg.model.checkpoint_path if set."""
        
        super(DPR, self).__init__(documents, index_path)
        self.cfg = cfg
        self.device = cfg.get('device', 'cpu')
        self.batch_size = cfg.model.get('encode_batch_size', 64)
        self.tokenizer = TextTokenizer(name=cfg.model.tokenizer_name,
                                       query_max_len=cfg.model.query_max_len,
                                       passage_max_len=cfg.model.passage_max_len)

        if query_encoder is None or document_encoder is None:
            query_encoder, document_encoder = self.__load_models()
        self.query_encoder = query_encoder.to(self.device)
        self.document_encoder = document_encoder.to(self.device)

        self.passage_vectors = self.__load_or_encode_passages(cfg.model.get('passage_vectors_path', None))

        #Must match the loss used in training: normalize=True there means cosine similarity, so retrieval uses it as well
        self.normalize = cfg.model.get('normalize', False)
        if self.normalize:
            self.passage_vectors = F.normalize(self.passage_vectors, dim=-1)

    def __load_models(self):
        query_encoder = build_encoder(self.cfg.model)
        document_encoder = build_encoder(self.cfg.model)

        checkpoint_path = self.cfg.model.get('checkpoint_path', None)
        if checkpoint_path:
            #One file with keys query_encoder and document_encoder (state dicts), config and record
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            query_encoder.load_state_dict(checkpoint['query_encoder'])
            document_encoder.load_state_dict(checkpoint['document_encoder'])
        return query_encoder, document_encoder

    def __load_or_encode_passages(self, vectors_path: str = None) -> torch.tensor:
        """Returns the N x emb_dim passage matrix. Rows follow the order of self.index.GetDocuments().
        If vectors_path is given, vectors are loaded from it when they match the current documents, and saved to it otherwise."""

        documents = self.index.GetDocuments()
        document_ids = [document.GetId() for document in documents]

        if vectors_path and os.path.exists(vectors_path):
            saved = torch.load(vectors_path, map_location='cpu')
            if saved['document_ids'] == document_ids:
                return saved['vectors']
            print(f'Saved passage vectors in {vectors_path} do not match the current documents. Re-encoding.')

        vectors = encode_texts(self.document_encoder, [document.GetText() for document in documents],
                               self.tokenizer.encode_passages, self.batch_size, self.device)

        if vectors_path:
            os.makedirs(os.path.dirname(vectors_path) or '.', exist_ok=True)
            torch.save({'document_ids': document_ids, 'vectors': vectors}, vectors_path)
        return vectors

    def CalculateScores(self, queries: list[str]) -> np.ndarray:
        """@param queries: The query texts. Returns: num_queries x num_documents scores (dot products), columns in the order of self.index.GetDocuments()."""

        query_vectors = encode_texts(self.query_encoder, queries, self.tokenizer.encode_queries, self.batch_size, self.device)
        if self.normalize:
            query_vectors = F.normalize(query_vectors, dim=-1)
        scores = query_vectors @ self.passage_vectors.T
        return scores.numpy()