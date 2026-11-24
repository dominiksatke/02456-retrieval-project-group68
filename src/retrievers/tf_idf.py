import re
import numpy as np

from collections import Counter
from scipy.sparse import csr_matrix, lil_matrix
from src.data_utils.document import Document
from src.retrievers.base_retriever import Retriever

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class TFIDF(Retriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None, stopwords=None) -> None:
        """
        Args:
            stopwords (iterable of str): Words to drop from documents and queries
        """

        # with default value
        self.stopwords = frozenset(stopwords) if stopwords else frozenset()
        super(TFIDF, self).__init__(documents, index_path)
        self.corpus_vocabulary = self.GetCorpusVocabulary()
        self.idf = self.GetInverseDocumentFrequencies()
        self.BuildMatrix()

    def BuildMatrix(self) -> None:
        """
        Function to build the document matrix used by CalculateScores
        """
        self.tfidf_matrix = self.GetDocumentsTFIDFVectors()

    def PreprocessText(self, text: str) -> list[str]:
        """
        Function to clean a text

        Args:
            text (str): The text to clean

        Returns:
            list of str: Cleaned out text (lower case, no numbers or punctuation marks, no stop words)
        """
        tokens = TOKEN_PATTERN.findall(text.lower())
        if self.stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        return tokens
        
    def GetQueryVocabulary(self, query: str) -> list[str]:
        """
        Function to collect all unique words in a query

        Args:
            query (str): The query

        Returns:
            list of str: Distinct words in a query
        """

        return list(set(self.PreprocessText(query)))

    def GetDocumentVocabulary(self, document: Document) -> list[str]:
        """
        Function to collect all unique words in a document

        Args:
            document (Document): The document

        Returns:
            list of str: Distinct words in a document
        """
        return list(set(self.PreprocessText(document.GetText())))
    
    def GetCorpusVocabulary(self) -> set[str]:
        """
        Function to collect all unique words in all documents of the corpus

        Returns:
            set of str: Distinct words in the corpus
        """
        processed_texts = map(lambda doc: set(self.PreprocessText(doc.GetText())), self.index.GetDocuments())
        corpus_vocabulary = set().union(*processed_texts)
        return corpus_vocabulary

    def GetDocumentFrequencies(self) -> dict[str, int]:
        """
        Function to count in how many documents each term occurs (counted once per document)

        Returns:
            dict: Term with the number of documents that contain it
        """
        document_frequencies = Counter()
        for document in self.index.GetDocuments():
            document_frequencies.update(self.GetDocumentVocabulary(document))
        return dict(document_frequencies)
       
    
    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """
        Function to compute TF-IDF IDF:
            IDF(q) = log(N / n_q)

        Returns:
            dict: Term with its IDF score
        """
        n_documents = len(self.index.GetDocuments())
        return {term: np.log(n_documents / n_term)
                for term, n_term in self.GetDocumentFrequencies().items()}
    
    
    def GetDocumentTermCounts(self, document: Document):
        """
        Function to count how many times each term occurs in a document

        Args:
            document (Document): The document

        Returns:
            dict: Term with its number of occurrences in the document
        """
        return dict(Counter(self.PreprocessText(document.GetText())))

    def GetDocumentsTFIDFVectors(self):
        """
        Function to compute the TF-IDF matrix:
            Element (i, j) = f(q_j, D_i) / max_f(D_i) * IDF(q_j)

        Returns:
            csr_matrix: n_documents x n_terms TF-IDF matrix
        """
        # Create a mapping from terms to indices
        self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}

        n_documents = len(self.index.GetDocuments())
        n_terms = len(self.corpus_vocabulary)

        # Collect non-zero entries as COO triplets; much faster than filling a lil_matrix cell by cell
        rows, cols, values = [], [], []
        for i, document in enumerate(self.index.GetDocuments()):
            term_counts = self.GetDocumentTermCounts(document)
            if not term_counts:
                continue
            max_count = max(term_counts.values())
            for term, count in term_counts.items():
                rows.append(i)
                cols.append(self.term_to_index[term])
                values.append((count / max_count) * self.idf[term])  # TF(q, D) * IDF(q)

        tfidf_matrix = csr_matrix((values, (rows, cols)), shape=(n_documents, n_terms), dtype=np.float32)
        
        return tfidf_matrix
    
    def QueryToVector(self, queries: list[str]):
        """
        Function to convert queries into sparse vectors of term counts over the corpus vocabulary

        Args:
            queries (list of str): The queries to convert (a single str is also accepted)

        Returns:
            csr_matrix: n_queries x n_terms matrix of term counts
        """
        if isinstance(queries, str):
            queries = [queries]
        if not hasattr(self, 'term_to_index'):
            self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}

        query_vector = lil_matrix((len(queries), len(self.corpus_vocabulary)), dtype=np.float32)
        for i, query in enumerate(queries):
            # Terms outside the corpus vocabulary have no document to match and are skipped
            for term, count in Counter(self.PreprocessText(query)).items():
                idx = self.term_to_index.get(term)
                if idx is not None:
                    query_vector[i, idx] = count

        return query_vector.tocsr()

    #@time_func
    def CalculateScores(self, queries: list[str]):
        """
        Function to calculate scores for a query

        Args:
            queries (list of str): The queries for which to calculate scores

        Returns:
            np.array: n_queries x n_documents array of scores
        """

        # Convert the queries to a vector
        query_vectors = self.QueryToVector(queries) # n_queries x vocab size
        scores = query_vectors.dot(self.tfidf_matrix.T).toarray() # n_queries x n_documents
        
        return scores
    