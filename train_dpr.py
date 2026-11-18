"""Train the DPR query and passage encoders.

Examples:
    python train_dpr.py model.checkpoint_path=checkpoints/scratch.pt
    python train_dpr.py model=bert train.lr=2e-5 model.checkpoint_path=checkpoints/bert.pt
    python train_dpr.py train.overfit_check=true   # only one batch, to test that things work
"""
import json
import os

import hydra
import torch
from omegaconf import OmegaConf

from src.data_utils.corpus import Corpus
from src.data_utils.tokenization import TextTokenizer, build_training_pairs, DPRBatchSampler
from src.retrievers.dpr import build_encoder
from src.evaluation.validation import make_val_fn
from src.training.loss import DPRLoss
from src.training.trainer import Trainer


@hydra.main(config_path='config', config_name='fiqa_train.yaml', version_base=None)
def main(cfg):
    if not cfg.train.overfit_check and not cfg.model.checkpoint_path:
        raise ValueError('please set model.checkpoint_path')

    torch.manual_seed(cfg.train.seed)  # before building the encoders so init is the same every run

    # data
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')
    tokenizer = TextTokenizer(cfg.model.tokenizer_name, cfg.model.query_max_len, cfg.model.passage_max_len)
    pairs = build_training_pairs(corpus.train)
    train_loader = DPRBatchSampler(pairs, tokenizer, batch_size=cfg.train.batch_size, seed=cfg.train.seed)
    print(f'number of training pairs: {len(pairs)}')

    # separate encoders for queries and passages
    query_encoder = build_encoder(cfg.model)
    document_encoder = build_encoder(cfg.model)

    loss_fn = DPRLoss(temperature=cfg.train.temperature, normalize=cfg.model.get('normalize', False))
    trainer = Trainer(criterion=loss_fn,
                      lr=cfg.train.lr,
                      weight_decay=cfg.train.weight_decay,
                      epochs=cfg.train.epochs,
                      grad_clip=cfg.train.grad_clip,
                      device=cfg.device,
                      checkpoint_path=cfg.model.checkpoint_path,
                      config=OmegaConf.to_container(cfg.model, resolve=True))

    if cfg.train.overfit_check:
        batch = next(iter(train_loader))
        trainer.overfit_check(query_encoder, document_encoder, batch)
        return

    # validation recall is used to pick the best epoch
    val_fn = make_val_fn(cfg, corpus, k=cfg.top_k)
    history = trainer.train(query_encoder, document_encoder, train_loader=train_loader, val_fn=val_fn)

    # save the training history for plotting
    history_path = os.path.splitext(cfg.model.checkpoint_path)[0] + '_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print('done, model saved to', cfg.model.checkpoint_path)


if __name__ == '__main__':
    main()
