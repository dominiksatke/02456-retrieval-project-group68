from src.data_utils.query import Query

def calculatePrecision(relevancies: list[bool], k: int):
    return sum(relevancies[:k]) / k

def calculateRecall(relevancies: list[bool], query: Query, k: int):
    return sum(relevancies[:k]) / query.getNumberOfRelevantDocuments()

def calculateReciprocalRank(relevancies: list[bool], k: int):
    for rank, is_relevant in enumerate(relevancies[:k], start=1):
        if is_relevant:
            return 1 / rank
    return 0.0

def Evaluate(relevancies: list[list[bool]], queries: list[Query], ks: list[int] = None):
    assert len(relevancies) == len(queries), 'The number of relevant passages must match the number of queries'
    n_queries = len(queries)

    if ks is None:
        ks = [len(relevancies[0])]

    scores = {}
    for k in ks:
        assert k <= len(relevancies[0]), f'Cannot evaluate at k={k}: only {len(relevancies[0])} documents were retrieved'
        total_precision, total_recall, total_rr = 0, 0, 0
        for i in range(n_queries):
            total_precision += calculatePrecision(relevancies[i], k)
            total_recall += calculateRecall(relevancies[i], queries[i], k)
            total_rr += calculateReciprocalRank(relevancies[i], k)
        scores[f'P@{k}'] = total_precision / n_queries
        scores[f'R@{k}'] = total_recall / n_queries
        scores[f'MRR@{k}'] = total_rr / n_queries
    return scores