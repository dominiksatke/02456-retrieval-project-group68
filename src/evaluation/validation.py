import copy
from omegaconf import open_dict
from src.data_utils.query import Query
from src.retrievers.dpr import DPR
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.evaluation.metrics import Evaluate


def make_val_fn(cfg, corpus, k: int = 10):
    """ @param cfg: Experiment config (cfg.device and cfg.model, as used by DPR).
    @param corpus: The Corpus to validate on. Queries come from corpus.val; the search pool is corpus.all_documents.
    @param k: Recall cutoff. 
    Returns val_fn(query_encoder, document_encoder) -> Recall@k on the validation split, for the trainer."""

    #It's built once and reused every epoch
    queries = [Query(question, id=query_id, relevant_document_ids=corpus.relevant_ids[query_id])
               for question, query_id in zip(corpus.val['question'], corpus.val['query_id'])]
    documents = corpus.all_documents

    #The weights change every epoch, so passage vectors must always be re-encoded, never loaded from a saved file
    val_cfg = copy.deepcopy(cfg)
    with open_dict(val_cfg):
        val_cfg.model.passage_vectors_path = None

    def val_fn(query_encoder, document_encoder) -> float:
        dpr = DPR(val_cfg, documents=documents, query_encoder=query_encoder, document_encoder=document_encoder)
        relevancies = retrieveQueryAndGetRelevancies(dpr, queries, k=k)
        return Evaluate(relevancies, queries, ks=[k])[f'R@{k}']

    return val_fn