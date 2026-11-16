import numpy as np
from scipy.sparse import csr_matrix
from src.retrievers.tf_idf import TFIDF


class BM25(TFIDF):
    def __init__(self, documents: list[dict] = None, index_path: str = None, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        super(BM25, self).__init__(documents, index_path)

        if type(self) == BM25:
            self.bm25_matrix = self.GetBM25Matrix()

    def GetDocumentLengths(self):
        document_lengths = {}
        average_document_length = 0
        for document in self.index.GetDocuments():
            length = len(self.PreprocessText(document.GetText()))
            document_lengths[document] = length
            average_document_length += length
        
        return document_lengths, average_document_length/len(self.index.GetDocuments())

    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """
        Function to compute BM25 IDF:
            IDF(q) = log((N - n_q + 0.5) / (n_q + 0.5) + 1)

        Returns:
            dict: Term with its BM25 IDF score   
        """
        n_documents = len(self.index.GetDocuments())
        
        return {term: np.log((n_documents - n_term + 0.5) / (n_term + 0.5) + 1)
                for term, n_term in self.GetDocumentFrequencies().items()}

    def GetBM25Matrix(self):
        """
        Function to compute the BM25 matrix:
            Element (i, j) = IDF(q_j) * f(q_j, D_i) * (k1 + 1) / (f(q_j, D_i) + k1 * (1 - b + b * |D_i| / avgdl))

        Returns:
            csr_matrix: n_documents x n_terms BM25 matrix
        """
        self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}
        document_lengths, average_document_length = self.GetDocumentLengths()

        n_documents = len(self.index.GetDocuments())
        n_terms = len(self.corpus_vocabulary)

        rows, cols, values = [], [], []
        for i, document in enumerate(self.index.GetDocuments()):
            length_norm = 1 - self.b + self.b * document_lengths[document] / average_document_length
            for term, count in self.GetDocumentTermCounts(document).items():
                rows.append(i)
                cols.append(self.term_to_index[term])
                values.append(self.idf[term] * count * (self.k1 + 1) / (count + self.k1 * length_norm))

        return csr_matrix((values, (rows, cols)), shape=(n_documents, n_terms), dtype=np.float32)

    def CalculateScores(self, queries: list[str]):
        """
        Function to calculate scores for a query

        Args:
            query (list of str): The query for which to calculate scores

        Returns:
            np.array: An array of scores for each document
        """
        query_vector = self.QueryToVector(queries)
        scores = query_vector.dot(self.bm25_matrix.T).toarray()
        
        return scores
