import math
import warnings

import torch
import torch.nn as nn

from src.training.loss import DPRLoss, in_batch_accuracy

# Every batch is a dict of integer tensors:
#   'query_ids'  [B, Lq]   token ids of the queries
#   'query_mask' [B, Lq]   1 = real token, 0 = padding
#   'doc_ids'    [B, Ld]   token ids of the passages; row i is the positive for query i
#   'doc_mask'   [B, Ld]
BATCH_KEYS = ('query_ids', 'query_mask', 'doc_ids', 'doc_mask')


class Trainer:
    def __init__(self, criterion: nn.Module = None, lr: float = 3e-4, weight_decay: float = 0.01,
                 epochs: int = 10, grad_clip: float = 1.0, device: str = 'cpu',
                 checkpoint_path: str = None, config: dict = None, seed: int = None):
        """
        @param criterion:       loss module; default is the PDF loss (dot product, temperature 1)
        @param lr:              learning rate. 3e-4 for the scratch encoder, about 2e-5 for pretrained BERT
        @param weight_decay:    AdamW weight decay
        @param epochs:          number of passes over the training data
        @param grad_clip:       maximum gradient norm per step, None to switch off
        @param device:          'cpu' or 'cuda'
        @param checkpoint_path: where the best model is saved, None to save nothing
        @param config:          model settings, stored in the checkpoint so the encoders can be rebuilt
        @param seed:            fixes the random numbers so a run can be repeated
        """
        self.criterion = criterion if criterion is not None else DPRLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.config = config
        self._warned_duplicates = False
        if seed is not None:
            torch.manual_seed(seed)

    def _parameters(self, query_encoder: nn.Module, document_encoder: nn.Module) -> list:
        """All trainable weights of both encoders, each listed once (also if the same encoder is passed twice)."""
        seen, params = set(), []
        for p in list(query_encoder.parameters()) + list(document_encoder.parameters()):
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                params.append(p)
        return params

    def _make_optimizer(self, query_encoder: nn.Module, document_encoder: nn.Module) -> torch.optim.Optimizer:
        # One optimizer for both encoders, since the loss depends on both
        return torch.optim.AdamW(self._parameters(query_encoder, document_encoder),
                                 lr=self.lr, weight_decay=self.weight_decay)

    def _to_device(self, batch: dict) -> dict:
        missing = [k for k in BATCH_KEYS if k not in batch]
        if missing:
            raise KeyError(f'Batch is missing {missing}; expected keys {BATCH_KEYS}')
        return {k: batch[k].to(self.device) for k in BATCH_KEYS}

    def _check_duplicates(self, batch: dict) -> None:
        """Warns once if a query or passage appears twice in a batch.
        The loss would then treat a correct passage as a wrong one for the other row."""
        if self._warned_duplicates:
            return
        for key, what in (('doc_ids', 'passage'), ('query_ids', 'query')):
            ids = batch[key]
            if torch.unique(ids, dim=0).shape[0] < ids.shape[0]:
                warnings.warn(f'The same {what} appears more than once in a batch; '
                              f'in-batch negatives assume every {what} is unique.')
                self._warned_duplicates = True

    def train_step(self, query_encoder: nn.Module, document_encoder: nn.Module,
                   batch: dict, optimizer: torch.optim.Optimizer) -> dict:
        """One forward pass, one backward pass and one weight update. Returns the loss and in-batch accuracy."""
        batch = self._to_device(batch)
        self._check_duplicates(batch)

        optimizer.zero_grad()

        # Queries go through g and passages through f; the masks let them ignore padding
        query_vectors = query_encoder(batch['query_ids'], batch['query_mask'])     # [B, d]
        document_vectors = document_encoder(batch['doc_ids'], batch['doc_mask'])   # [B, d]

        loss = self.criterion(query_vectors, document_vectors)
        if not torch.isfinite(loss):
            # Stop before a nan update breaks all the weights
            raise RuntimeError(f'Loss is {loss.item()}; lower the learning rate or check the encoder outputs.')

        loss.backward()
        if self.grad_clip is not None:
            # Large dot products can give very large gradients; clipping limits the step size
            torch.nn.utils.clip_grad_norm_(self._parameters(query_encoder, document_encoder), self.grad_clip)
        optimizer.step()

        accuracy = in_batch_accuracy(query_vectors.detach(), document_vectors.detach(),
                                     normalize=getattr(self.criterion, 'normalize', False))
        return {'loss': loss.item(), 'accuracy': accuracy}

    def train(self, query_encoder: nn.Module, document_encoder: nn.Module,
              train_loader=None, val_fn=None, epochs: int = None) -> list:
        """Epoch loop with validation and checkpointing (not implemented yet)."""
        raise NotImplementedError('Use train_step() or overfit_check() for now')

    def overfit_check(self, query_encoder: nn.Module, document_encoder: nn.Module, batch: dict,
                      steps: int = 300, lr: float = None, target_loss: float = 0.05) -> dict:
        """
        Trains on one fixed batch (e.g. 64 FiQA pairs) until it is memorised.
        The loss should fall from about ln(B) towards 0 and in-batch accuracy should reach 1.0.
        This checks that both encoders learn and that query i is paired with passage i.
        """
        query_encoder.to(self.device)
        document_encoder.to(self.device)
        # Eval mode turns off dropout, which would keep the loss from reaching 0
        query_encoder.eval()
        document_encoder.eval()

        # No weight decay here, because the only goal is to memorise the batch
        optimizer = torch.optim.AdamW(self._parameters(query_encoder, document_encoder),
                                      lr=lr if lr is not None else self.lr, weight_decay=0.0)

        losses, step = [], {'loss': math.inf, 'accuracy': 0.0}
        for _ in range(steps):
            step = self.train_step(query_encoder, document_encoder, batch, optimizer)
            losses.append(step['loss'])
            if step['loss'] < target_loss and step['accuracy'] == 1.0:
                break

        batch_size = batch['query_ids'].shape[0]
        result = {'initial_loss': losses[0], 'final_loss': step['loss'], 'final_accuracy': step['accuracy'],
                  'steps': len(losses), 'chance_loss': math.log(batch_size), 'losses': losses,
                  'passed': step['loss'] < target_loss and step['accuracy'] == 1.0}
        print(f"overfit check on {batch_size} pairs: loss {result['initial_loss']:.3f} -> {result['final_loss']:.4f} "
              f"in {result['steps']} steps, in-batch acc {result['final_accuracy']:.3f} "
              f"(chance loss ln({batch_size}) = {result['chance_loss']:.3f}) -> {'PASSED' if result['passed'] else 'FAILED'}")
        return result
