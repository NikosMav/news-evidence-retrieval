"""Evidence retrieval over a fixed corpus.

Sparse TF-IDF or BM25, dense MiniLM, and hybrid RRF.
Not a fact-checker. ISOT labels are source buckets, not claim-level truth.
The ISOT title-recovery table is a sanity check; SciFact qrels are the external comparison.
"""

__version__ = "0.1.0"
