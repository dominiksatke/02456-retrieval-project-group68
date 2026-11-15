import random
from transformers import AutoTokenizer

class TextTokenizer:
    """Turns lists of texts into the token tensors that the encoders are expexting. 
    Padding and truncation are handled on the right, and [CLS] is at position 0."""

    def __init__(self, name: str = 'bert-base-uncased', query_max_len: int = 64, passage_max_len: int = 256, tokenizer=None):
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(name)
        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'right'
        self.query_max_len = query_max_len
        self.passage_max_len = passage_max_len

    def __encode(self, texts: list[str], max_len: int):
        """Returns input_ids and attention_mask, both [B, L] integer tensors, with L <= max_len."""
        tokens = self.tokenizer(texts, padding='longest', truncation=True, max_length=max_len, return_tensors='pt')
        return tokens['input_ids'], tokens['attention_mask']
    
    def encode_queries(self, texts: list[str]):
        return self.__encode(texts, self.query_max_len)

    def encode_passages(self, texts: list[str]):
        return self.__encode(texts, self.passage_max_len)

    def token_lengths(self, texts: list[str]) -> list[int]:
        """Untruncated token counts (including [CLS] and [SEP]), for choosing the max lengths."""
        return [len(ids) for ids in self.tokenizer(texts, truncation=False)['input_ids']]

def build_training_pairs(split) -> list[tuple[str, str, str, str]]:
    """ @param split: A corpus split (e.g. corpus.train) with columns question, query_id, text and document_id.
    Returns one (query_id, query_text, document_id, document_text) tuple per relevant passage.
    A FiQA query with three relevant passages gives three pairs."""
    pairs = []
    for question, query_id, texts, document_ids in zip(split['question'], split['query_id'], split['text'], split['document_id']):
        for text, document_id in zip(texts, document_ids):
            pairs.append((query_id, question, document_id, text))
    return pairs

class DPRBatchSampler:
    """ Yields training batches as the agreed dict: query_ids, query_mask [B, L <= 64] and doc_ids, doc_mask [B, L <= 256].
    Row i of the passages is the positive for query i.
    No query and no passage appears twice in a batch. Otherwise a correct passage would also count as a
    negative for another row (e.g. SQuAD questions that share a paragraph, or a FiQA query with several answers)."""

    def __init__(self, pairs: list[tuple[str, str, str, str]], tokenizer: TextTokenizer, batch_size: int = 64, shuffle: bool = True, drop_last: bool = False, seed: int = 0):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = random.Random(seed)

    def __group_into_batches(self) -> list[list[tuple]]:
        pairs = list(self.pairs)
        if self.shuffle:
            self.rng.shuffle(pairs)

        batches = []
        current, seen_queries, seen_documents = [], set(), set()
        waiting = []  #pairs that clashed with the current batch are tried again in the next one

        def try_add(pair) -> bool:
            query_id, _, document_id, _ = pair
            if len(current) == self.batch_size or query_id in seen_queries or document_id in seen_documents:
                return False
            current.append(pair)
            seen_queries.add(query_id)
            seen_documents.add(document_id)
            return True

        for pair in pairs:
            if not try_add(pair):
                waiting.append(pair)
            if len(current) == self.batch_size:
                batches.append(current)
                current, seen_queries, seen_documents = [], set(), set()
                waiting = [p for p in waiting if not try_add(p)]

        while waiting:
            waiting = [p for p in waiting if not try_add(p)]
            if waiting or len(current) == self.batch_size:
                batches.append(current)
                current, seen_queries, seen_documents = [], set(), set()
        if current:
            batches.append(current)

        if self.drop_last:
            batches = [b for b in batches if len(b) == self.batch_size]
        return batches

    def __iter__(self):
        #A new shuffle every time this is iterated, i.e. every epoch
        for batch in self.__group_into_batches():
            _, query_texts, _, document_texts = zip(*batch)
            query_ids, query_mask = self.tokenizer.encode_queries(list(query_texts))
            doc_ids, doc_mask = self.tokenizer.encode_passages(list(document_texts))
            yield {'query_ids': query_ids, 'query_mask': query_mask,
                   'doc_ids': doc_ids, 'doc_mask': doc_mask}