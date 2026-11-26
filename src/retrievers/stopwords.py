"""
English stop-word lists for the retrievers
"""

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Lucene english list
LUCENE_STOPWORDS = frozenset("""
a an and are as at be but by for if in into is it no not of on or such that the their then there these
they this to was will with
""".split())

# NLTK English list
NLTK_STOPWORDS = frozenset("""
i me my myself we our ours ourselves you you're you've you'll you'd your yours yourself yourselves he him
his himself she she's her hers herself it it's its itself they them their theirs themselves what which who
whom this that that'll these those am is are was were be been being have has had having do does did doing a
an the and but if or because as until while of at by for with about against between into through during
before after above below to from up down in out on off over under again further then once here there when
where why how all any both each few more most other some such no nor not only own same so than too very s t
can will just don don't should should've now d ll m o re ve y ain aren aren't couldn couldn't didn didn't
doesn doesn't hadn hadn't hasn hasn't haven haven't isn isn't ma mightn mightn't mustn mustn't needn needn't
shan shan't shouldn shouldn't wasn wasn't weren weren't won won't wouldn wouldn't
""".split())

# scikit-learn English list
SKLEARN_STOPWORDS = frozenset(ENGLISH_STOP_WORDS)

# Names used in config/experiment.yaml (stopwords: nltk); "none" or a missing key keeps every word
STOPWORD_LISTS = {
    'none': frozenset(),
    'lucene': LUCENE_STOPWORDS,
    'nltk': NLTK_STOPWORDS,
    'sklearn': SKLEARN_STOPWORDS,
}


def get_stopwords(name) -> frozenset:
    """Returns the stop-word list for a config name"""
    
    if name is None:
        return frozenset()

    return STOPWORD_LISTS[name]
