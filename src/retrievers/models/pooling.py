import torch


def masked_pool(x: torch.Tensor, mask: torch.Tensor = None, pooling_type: str = 'cls') -> torch.Tensor:
    """
    Turn one vector per token into one vector per text.

    @param x:            [B, L, d] token vectors from the last encoder layer
    @param mask:         [B, L], 1 = real token, 0 = padding. None means "no padding".
    @param pooling_type: 'cls' | 'mean' | 'max'

    Returns: [B, d], not normalised (normalisation, if any, happens in loss.py)

    Lives in its own file so it can be tested without building a full encoder, and so
    edits to encoder.py (Role B) and to pooling (Role C) never touch the same lines.
    TransformerEncoder.pool() in encoder.py only forwards to it.
    """
    if x.dim() != 3:
        raise ValueError(f'Expected x of shape [B, L, d], got {tuple(x.shape)}')

    if mask is None:
        # Keeps the starter's one-argument call pool(x) working: treat every position as real.
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    if mask.shape != x.shape[:2]:
        raise ValueError(f'Expected mask of shape {tuple(x.shape[:2])}, got {tuple(mask.shape)}')

    # The tokenizer gives 0/1 integers; masked_fill needs booleans.
    # unsqueeze(-1) makes it [B, L, 1] so one mask value covers all d numbers of a token.
    keep = mask.to(torch.bool).unsqueeze(-1)

    if pooling_type == 'cls':
        # [CLS] is always at position 0 (shared interface), so it is never padding and
        # needs no mask here. It only becomes a summary of the text because attention
        # lets it read the other tokens; keeping padding out of that is the job of the
        # attention mask inside the encoder layers, not of pooling.
        return x[:, 0]

    if pooling_type == 'mean':
        # Zero out padding so it adds nothing to the sum...
        summed = x.masked_fill(~keep, 0.0).sum(dim=1)                # [B, d]
        # ...and divide by the number of REAL tokens, not by L. Dividing by L would
        # shrink the vectors of short texts just because they were padded more.
        # clamp(min=1) only guards against division by zero for an all-padding row.
        n_real = keep.sum(dim=1).clamp(min=1)                        # [B, 1]
        return summed / n_real

    if pooling_type == 'max':
        # Padding is set to -inf, not 0: if every real value in a dimension is negative,
        # a 0 from a padding position would win the max and leak into the result.
        # (A row with no real token would give -inf, but [CLS] is always present.)
        return x.masked_fill(~keep, float('-inf')).max(dim=1).values  # .values: max also returns indices

    raise ValueError(f"Unknown pooling_type '{pooling_type}', expected 'cls', 'mean' or 'max'")
