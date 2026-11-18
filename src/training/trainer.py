import math
import os

import torch
import torch.nn as nn

from src.training.loss import DPRLoss, in_batch_accuracy


class Trainer:
    """Trains a query encoder and a document encoder with in-batch negatives (DPR).

    Each batch is a dict with 'query_ids', 'query_mask', 'doc_ids', 'doc_mask'.
    Row i of the docs is the positive passage for query i.
    """

    def __init__(self, criterion=None, lr=3e-4, weight_decay=0.01, epochs=10,
                 grad_clip=1.0, device='cpu', checkpoint_path=None, config=None, seed=None):
        self.criterion = criterion if criterion is not None else DPRLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.config = config  # saved in the checkpoint so we can rebuild the model later
        if seed is not None:
            torch.manual_seed(seed)

    def get_params(self, query_encoder, document_encoder):
        # if both encoders are the same (shared weights) we only want the params once
        if query_encoder is document_encoder:
            return list(query_encoder.parameters())
        return list(query_encoder.parameters()) + list(document_encoder.parameters())

    def train_step(self, query_encoder, document_encoder, batch, optimizer):
        q_ids = batch['query_ids'].to(self.device)
        q_mask = batch['query_mask'].to(self.device)
        d_ids = batch['doc_ids'].to(self.device)
        d_mask = batch['doc_mask'].to(self.device)

        optimizer.zero_grad()
        q_vecs = query_encoder(q_ids, q_mask)      # [B, d]
        d_vecs = document_encoder(d_ids, d_mask)   # [B, d]

        loss = self.criterion(q_vecs, d_vecs)
        if not torch.isfinite(loss):
            raise RuntimeError('loss is nan or inf, try a lower learning rate')
        loss.backward()

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.get_params(query_encoder, document_encoder), self.grad_clip)
        optimizer.step()

        acc = in_batch_accuracy(q_vecs.detach(), d_vecs.detach(),
                                normalize=getattr(self.criterion, 'normalize', False))
        return {'loss': loss.item(), 'accuracy': acc}

    def train(self, query_encoder, document_encoder, train_loader, val_fn=None, epochs=None):
        """val_fn(query_encoder, document_encoder) should return a score where higher is better
        (e.g. Recall@10 on validation). The best epoch is saved. Returns a list with one dict per epoch."""
        if epochs is None:
            epochs = self.epochs

        query_encoder.to(self.device)
        document_encoder.to(self.device)
        optimizer = torch.optim.AdamW(self.get_params(query_encoder, document_encoder),
                                      lr=self.lr, weight_decay=self.weight_decay)

        history = []
        best_val = -math.inf

        for epoch in range(1, epochs + 1):
            query_encoder.train()
            document_encoder.train()

            total_loss, total_acc, n = 0.0, 0.0, 0
            for batch in train_loader:
                out = self.train_step(query_encoder, document_encoder, batch, optimizer)
                total_loss += out['loss']
                total_acc += out['accuracy']
                n += 1

            if n == 0:
                raise ValueError('train_loader gave no batches')
            record = {'epoch': epoch, 'loss': total_loss / n, 'accuracy': total_acc / n, 'val': None}

            if val_fn is not None:
                query_encoder.eval()
                document_encoder.eval()
                with torch.no_grad():
                    record['val'] = float(val_fn(query_encoder, document_encoder))

                if record['val'] > best_val:
                    best_val = record['val']
                    self.save_checkpoint(query_encoder, document_encoder, record)
            else:
                # no validation -> just keep the latest model
                self.save_checkpoint(query_encoder, document_encoder, record)

            history.append(record)
            print(f"epoch {epoch}: loss {record['loss']:.4f}, acc {record['accuracy']:.3f}, val {record['val']}")

        return history

    def save_checkpoint(self, query_encoder, document_encoder, record=None):
        if self.checkpoint_path is None:
            return
        os.makedirs(os.path.dirname(self.checkpoint_path) or '.', exist_ok=True)
        torch.save({
            'query_encoder': query_encoder.state_dict(),
            'document_encoder': document_encoder.state_dict(),
            'config': self.config,
            'record': record,
        }, self.checkpoint_path)

    def overfit_check(self, query_encoder, document_encoder, batch, steps=300, lr=None):
        """Sanity check: train on a single batch, the loss should go from about ln(B) to ~0."""
        query_encoder.to(self.device)
        document_encoder.to(self.device)
        query_encoder.eval()  # no dropout, otherwise it can't fully memorise the batch
        document_encoder.eval()

        optimizer = torch.optim.AdamW(self.get_params(query_encoder, document_encoder),
                                      lr=lr or self.lr, weight_decay=0.0)

        losses = []
        for _ in range(steps):
            out = self.train_step(query_encoder, document_encoder, batch, optimizer)
            losses.append(out['loss'])

        B = batch['query_ids'].shape[0]
        passed = losses[-1] < 0.05 and out['accuracy'] == 1.0  # batch memorised
        print(f"overfit check: start loss {losses[0]:.3f} (ln(B) = {math.log(B):.3f}), "
              f"end loss {losses[-1]:.4f}, acc {out['accuracy']:.3f} -> {'PASSED' if passed else 'FAILED'}")
        return {'initial_loss': losses[0], 'final_loss': losses[-1],
                'final_accuracy': out['accuracy'], 'losses': losses, 'passed': passed}
