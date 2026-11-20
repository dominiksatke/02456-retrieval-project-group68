import os
import csv
import json
import time
import hashlib
import hydra
import torch
import logging
from omegaconf import OmegaConf
from src.data_utils.corpus import Corpus
from src.data_utils.query import Query
from src.retrievers.tf_idf import TFIDF
from src.retrievers.bm25 import BM25
from src.retrievers.dpr import DPR
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.evaluation.metrics import Evaluate


def build_queries(corpus: Corpus, split) -> list[Query]:
    return [Query(question, id=query_id, relevant_document_ids=corpus.relevant_ids[query_id])
            for question, query_id in zip(split['question'], split['query_id'])]


def get_device(requested: str) -> str:
    if requested == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, using cpu.')
        return 'cpu'
    if requested == 'mps' and not torch.backends.mps.is_available():
        print('MPS not available, using cpu.')
        return 'cpu'
    return requested


def vectors_path(model_cfg, dataset_name: str, method_name: str) -> str:
    """A file name unique to this dataset, model config and checkpoint file (including when it was last written),
    so saved passage vectors are only reused for exactly the same model and weights. """

    key = json.dumps(OmegaConf.to_container(model_cfg, resolve=True), sort_keys=True)
    checkpoint = model_cfg.get('checkpoint_path', None)
    if checkpoint and os.path.exists(checkpoint):
        key += str(os.path.getmtime(checkpoint))
    return f'outputs/vectors/{dataset_name}_{method_name}_{hashlib.md5(key.encode()).hexdigest()[:8]}.pt'


def build_retriever(method, corpus: Corpus, dataset_name: str, cfg, device: str):
    documents = corpus.all_documents

    if method.type == 'tf_idf':
        return TFIDF(documents=documents)

    if method.type == 'bm25':
        return BM25(documents=documents, k1=method.get('k1', 1.5), b=method.get('b', 0.75))

    if method.type == 'dpr':
        model_cfg = OmegaConf.load(f'config/model/{method.model}.yaml')
        model_cfg = OmegaConf.merge(model_cfg, method.get('model_overrides', {}) or {})
        #Only cache when the weights are fixed (a checkpoint, or an untrained pretrained backbone); a scratch encoder without a checkpoint is random every run
        fixed_weights = model_cfg.get('checkpoint_path', None) or model_cfg.type == 'pretrained'
        model_cfg.passage_vectors_path = vectors_path(model_cfg, dataset_name, method.name) if cfg.cache_passage_vectors and fixed_weights else None
        return DPR(OmegaConf.create({'device': device, 'model': model_cfg}), documents=documents)

    raise ValueError(f'Unknown method type {method.type}. Options are tf_idf | bm25 | dpr')


def save_results(rows: list[dict], results_path: str):
    """Adds rows to the CSV, keeping results from earlier runs. Columns are merged if a run used different ks."""
    os.makedirs(os.path.dirname(results_path) or '.', exist_ok=True)
    existing = []
    if os.path.exists(results_path):
        with open(results_path, newline='') as f:
            existing = list(csv.DictReader(f))

    columns = []
    for row in existing + rows:
        columns += [column for column in row if column not in columns]

    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval='')
        writer.writeheader()
        writer.writerows(existing + rows)


def run_experiments(cfg) -> list[dict]:
    device = get_device(cfg.device)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    all_rows = []

    for dataset_name in cfg.datasets:
        corpus = Corpus(dataset_name=dataset_name, base_dir=f'{cfg.base_path}/retrieval_data')
        splits = {'train': corpus.train, 'validation': corpus.val, 'test': corpus.test}
        queries = {name: build_queries(corpus, splits[name]) for name in cfg.splits if splits.get(name) is not None}
        print(f'\n=== {dataset_name}: {len(corpus.all_documents)} passages, splits {[f"{n} ({len(q)} queries)" for n, q in queries.items()]}')

        for method in cfg.methods:
            print(f'\n--- {method.name}')
            start = time.time()
            try:
                retriever = build_retriever(method, corpus, dataset_name, cfg, device)
            except NotImplementedError:
                print(f'Skipped: {method.type} is not implemented yet.')
                continue

            rows = []
            for split_name, split_queries in queries.items():
                relevancies = retrieveQueryAndGetRelevancies(retriever, split_queries, k=max(cfg.ks))
                scores = Evaluate(relevancies, split_queries, ks=list(cfg.ks))
                print(f'{split_name}: ' + ', '.join(f'{name} {value:.4f}' for name, value in scores.items()))
                rows.append({'timestamp': timestamp, 'dataset': dataset_name, 'split': split_name, 'method': method.name,
                             'n_queries': len(split_queries), **{name: round(value, 4) for name, value in scores.items()}})

            print(f'({time.time() - start:.0f}s)')
            save_results(rows, cfg.results_path)
            all_rows += rows

            del retriever
            if device == 'cuda':
                torch.cuda.empty_cache()

    print(f'\nResults appended to {cfg.results_path}')
    return all_rows


@hydra.main(config_path='config', config_name='experiment.yaml', version_base=None)
def main(cfg):
    logging.getLogger('httpx').setLevel(logging.WARNING)
    run_experiments(cfg)


if __name__ == '__main__':
    main()