import torch
import torch.nn as nn
import torch.nn.functional as F


def _score_matrix(queries: torch.Tensor, documents: torch.Tensor,
                  temperature: float = 1.0, normalize: bool = False) -> torch.Tensor:
    """
    Build the [B, B] score matrix shared by the loss and the accuracy helper.

    Row i belongs to query i, column j to document j, so entry (i, j) is the
    similarity between query i and document j. The correct pairs are on the diagonal.
    """
    # Fail early with a readable message. A silent shape mistake here (for example
    # [B, L, d] instead of [B, d]) would still "train" and only show up as bad recall.
    if queries.dim() != 2 or documents.dim() != 2:
        raise ValueError(f'Expected [B, d] tensors, got {tuple(queries.shape)} and {tuple(documents.shape)}')
    if queries.shape != documents.shape:
        # Row i of `documents` must be the positive for row i of `queries`,
        # so both need the same batch size and the same vector size.
        raise ValueError(f'Query and document batches must match, got {tuple(queries.shape)} and {tuple(documents.shape)}')
    if temperature <= 0:
        # Dividing by 0 gives inf/nan; a negative value would flip the ranking.
        raise ValueError(f'temperature must be > 0, got {temperature}')

    if normalize:
        # Optional (not in the PDF): scale every vector to length 1 so the dot product
        # becomes cosine similarity. The encoders still output unnormalised vectors,
        # as the shared interface requires; normalisation happens only here.
        queries = F.normalize(queries, dim=-1)
        documents = F.normalize(documents, dim=-1)

    # queries @ documents.T (not documents @ queries.T): this puts queries on the rows.
    # cross_entropy applies softmax along each row, so each query is compared against
    # all documents in the batch, exactly as the sum over D_j in PDF Section 2.2.
    scores = queries @ documents.T

    # Temperature lives here and nowhere else (shared interface, Section 3).
    # Dividing by a small temperature stretches the scores apart, which makes the
    # softmax sharper. With temperature = 1.0 this line changes nothing.
    return scores / temperature


def dpr_inbatch_loss(queries: torch.Tensor, documents: torch.Tensor,
                     temperature: float = 1.0, normalize: bool = False) -> torch.Tensor:
    """
    In-batch negatives loss from PDF Section 2.2.

    @param queries:     [B, d] query vectors g(Q_i), unnormalised
    @param documents:   [B, d] document vectors f(D_i); row i is the positive for query i
    @param temperature: scores are divided by this before the softmax (1.0 = PDF formula)
    @param normalize:   if True use cosine similarity instead of the raw dot product

    Returns: a 0d tensor (one number), the mean over the B queries of
             -log( exp(s_ii) / sum_j exp(s_ij) ),  with s_ij = f(D_j)^T g(Q_i) / temperature
    """
    scores = _score_matrix(queries, documents, temperature, normalize)

    # The "class" each query should pick is its own index: query 0 -> document 0, etc.
    # Created on the same device as the scores so the code runs unchanged on GPU.
    labels = torch.arange(scores.shape[0], device=scores.device)

    # cross_entropy = log-softmax + negative log-likelihood in one numerically stable
    # call. Writing exp() and log() by hand overflows for large dot products, which
    # unnormalised 128-d vectors easily produce.
    # Default reduction='mean' averages over the batch, as the interface specifies.
    return F.cross_entropy(scores, labels)


@torch.no_grad()  # a monitoring number only; it must never contribute gradients
def in_batch_accuracy(queries: torch.Tensor, documents: torch.Tensor, normalize: bool = False) -> float:
    """
    Share of queries whose own document gets the highest score in the batch.
    1/B is chance level, 1.0 means every query ranks its positive first.
    Temperature is left out on purpose: dividing by a positive number never changes
    which score is the largest.
    """
    scores = _score_matrix(queries, documents, 1.0, normalize)
    labels = torch.arange(scores.shape[0], device=scores.device)
    return (scores.argmax(dim=1) == labels).float().mean().item()


class DPRLoss(nn.Module):
    def __init__(self, temperature: float = 1.0, normalize: bool = False):
        # Both arguments have defaults so the starter's `DPRLoss()` call still works
        # and gives exactly the PDF loss (raw dot product, no temperature).
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, queries: torch.tensor, documents: torch.tensor):
        """
        @param queries: batch_size x embedding_dim
        @param documents: batch_size x embedding_dim

        Computes cross entropy between the query-document score matrix and the
        diagonal (query i belongs with document i).

        Returns: A 0d torch tensor
        """
        return dpr_inbatch_loss(queries, documents, self.temperature, self.normalize)
