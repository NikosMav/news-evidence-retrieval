# News Evidence Retrieval

Passage retrieval with TF-IDF, BM25, MiniLM embeddings, and hybrid reciprocal
rank fusion (RRF). The repository also keeps the original supervised
classification notebook as a documented historical study.

This project retrieves passages from a fixed corpus. It is **not a
fact-checker**. ISOT labels are source buckets, and a retrieved neighbor does
not establish whether a claim is true.

The comparison that can support a retrieval claim is SciFact document ranking
(nDCG@10 and Recall@100) next to the published BEIR BM25 anchor of **0.665**.
The ISOT table further down is a title-recovery sanity check. Do not cite it as
evidence retrieval.

## SciFact test

`beir/scifact/test` via `ir_datasets`: 300 claims, 5,183 abstracts, one document
per abstract. Metrics are pytrec_eval nDCG@10, Recall@100, and reciprocal rank.
BM25 is `bm25s` with Lucene scoring, k1=0.9, b=0.4, English stopwords, and a
Snowball stemmer. Dense retrieval is `sentence-transformers/all-MiniLM-L6-v2`.
Hybrid is RRF (k=60) of those two rank lists. The word-window chunker is not used.

The train split (809 queries) was run first. The test split was scored only
after that metric code was frozen. Regenerate with:

```bash
pip install -e ".[scifact]"
python scripts/run_scifact_eval.py --split train
python scripts/run_scifact_eval.py --split test
```

<!-- SCIFACT_TEST_TABLE_START -->
| Method | nDCG@10 | Recall@100 | Reciprocal rank |
| --- | ---: | ---: | ---: |
| bm25 | 0.6762 | 0.9127 | 0.6456 |
| dense | 0.6451 | 0.9250 | 0.6110 |
| hybrid | 0.7194 | 0.9550 | 0.6869 |
<!-- SCIFACT_TEST_TABLE_END -->

BM25 nDCG@10 is 0.6762. The published anchors are BEIR BM25 **0.665** and
Anserini flat BM25 **0.6789**. Recall@100 for this BM25 run is 0.9127; BEIR
reports 0.908 for BM25.

MiniLM does not beat BM25 on nDCG@10 (0.6451 vs 0.6762; paired sign test
p=0.73). Hybrid does (0.7194; 74 queries up, 21 down, 205 ties, p=4.3×10⁻⁸).

### Ablations on the same test qrels

The reranker uses hybrid as its first stage because that system had the higher
train nDCG@10 (0.719 vs BM25 0.694 vs MiniLM 0.660). The test split was not
used to choose it.

<!-- SCIFACT_ABLATION_TABLE_START -->
| Method | nDCG@10 | Recall@100 | Reciprocal rank |
| --- | ---: | ---: | ---: |
| hybrid_rerank | 0.6903 | 0.9550 | 0.6652 |
| bge | 0.7127 | 0.9417 | 0.6866 |
| bge_hybrid | 0.7189 | 0.9650 | 0.6865 |
<!-- SCIFACT_ABLATION_TABLE_END -->

Replacing MiniLM with `BAAI/bge-small-en-v1.5` moves dense nDCG@10 from 0.6451
to 0.7127 (sign test against BM25, p=0.018). Queries use BGE's retrieval
instruction; abstracts do not. Putting that encoder into RRF does not move the
hybrid number (0.7189 vs 0.7194).

Reranking the top 50 hybrid hits with `cross-encoder/ms-marco-MiniLM-L-6-v2`
does not help. nDCG@10 goes from 0.7194 to 0.6903, and the comparison with BM25
is not significant (p=0.10). Recall@100 stays 0.9550 because the reranker only
reorders those 50 hits.

`sentence-transformers` stays `>=2.6,<4`. Both ablations use the existing
`SentenceTransformer.encode` and `CrossEncoder` APIs. Regenerate the ablation
rows with `python scripts/run_scifact_ablations.py --split test`.

Three misses from the frozen three-way run are in
[`results/scifact_failures.md`](results/scifact_failures.md): a methionine-restriction
claim whose BM25 hit is a lifespan abstract while MiniLM returns a miRNA review;
a vCJD prevalence claim that BM25 ranks first and MiniLM replaces with a different
case report; and a low-birth-weight claim that none of the three systems place
in the top 10.

CI does not download BEIR or these models.

## ISOT title-recovery sanity check

The committed ISOT run uses 4,000 sampled articles and 300 queries. The query is the
article title. The gold set is every indexed body chunk of that same article.
High scores mean the index can find an article from a lightly edited copy of
its headline.

<!-- METRICS_TABLE_START -->
| Method | Article Hit@1 | Article Hit@5 | Article Hit@10 | Passage Recall@5 | nDCG@5 | nDCG@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf | 0.6533 | 0.8267 | 0.8700 | 0.4675 | 0.5181 | 0.5321 | 0.7290 |
| dense | 0.7767 | 0.8867 | 0.9033 | 0.5528 | 0.6163 | 0.6324 | 0.8205 |
| hybrid | 0.7833 | 0.9100 | 0.9400 | 0.5531 | 0.6258 | 0.6397 | 0.8415 |
<!-- METRICS_TABLE_END -->

Hybrid's article Hit@1 edge over dense is **0.0067** (0.7833 vs 0.7767, two
queries in 300). Article Hit@10 moves more: 0.9400 vs 0.9033.

Passage recall moves with chunk size because every chunk of the gold article is
relevant. In the committed ablation (2,000 articles, 150 queries), hybrid
article Hit@1 stays near 0.80 while passage Recall@5 goes from 0.397 at 60-word
chunks to 0.795 at 240-word chunks. Putting the title into the indexed passage
inflates article Hit@1 to 0.9867 (`results/retrieval_ablations.csv`). Ablation
defaults in the eval CLI match that committed run.

`article_mrr` was dropped. With one gold article it duplicated passage MRR on
every committed row.

## Paraphrase rewrite

The deterministic rewrite is not a robustness result. On the 300 committed
pairs, mean token Jaccard between the title and the rewrite is **0.806**, and
about 94% of content words are kept. TF-IDF article Hit@1 **rose** from 0.6533
to 0.6767. Hybrid MRR moved from 0.8415 to 0.8291. Details are in
[`results/paraphrase_eval_meta.json`](results/paraphrase_eval_meta.json).

## What this project demonstrates

- Reproducible ingestion and stratified sampling of ISOT articles
- Overlapping passage chunking with article and source metadata
- Sparse TF-IDF (ISOT default) and BM25 (`bm25s`) beside it
- Dense MiniLM and hybrid RRF retrieval
- Saved, reloadable indexes and a command-line interface
- A SciFact document-ranking script with pytrec_eval metrics
- Unit tests and CI without dataset or model downloads

## Retrieval pipeline

```text
ISOT CSVs -> article sample -> passages -> TF-IDF + MiniLM indexes
                                             |
query ---------------------------------------+-> RRF -> ranked passages

SciFact abstracts (ir_datasets) -> BM25 + MiniLM -> RRF -> nDCG@10, Recall@100
```

The default ISOT index contains body passages of about 120 words with 20-word
overlap. Index artifacts store passage metadata, the active sparse index, dense
vectors, and the configuration required to reload the same index. TF-IDF
remains the ISOT sparse arm so the title-recovery table can still be
regenerated. BM25 is selected with `sparse_backend="bm25"`.

## Quick start

The test suite uses synthetic data and mocked dense encoders. It does not
download ISOT, BEIR, or MiniLM:

```bash
git clone https://github.com/NikosMav/news-evidence-retrieval.git
cd news-evidence-retrieval
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
python -m evidence_retrieval -h
```

Build and query a real local ISOT index (TF-IDF + MiniLM):

```bash
python scripts/download_data.py
python -m evidence_retrieval build
python -m evidence_retrieval query "Federal Reserve raises interest rates" --top-k 5
```

The first build downloads `sentence-transformers/all-MiniLM-L6-v2` and writes the
index under `data/retrieval_index/default/`.

Regenerate the ISOT title-recovery evidence (ablation defaults: 2,000 articles,
150 queries):

```bash
python -m evidence_retrieval eval
python -m evidence_retrieval eval --paraphrase-only
```

## Repository map

| Path | Purpose |
| --- | --- |
| `evidence_retrieval/` | Chunking, TF-IDF, BM25, index, evaluation, and CLI |
| `tests/` | Unit tests (no BEIR, no torch) |
| `results/` | SciFact test table, ablations, and the ISOT title-recovery demo |
| `scripts/run_scifact_eval.py` | BM25 vs MiniLM vs RRF on SciFact |
| `scripts/run_scifact_ablations.py` | Cross-encoder rerank and BGE-small on the frozen test split |
| `scripts/` | Data download, ISOT demo evaluation, and notebook helpers |
| `evidence_retrieval.ipynb` | Closed-corpus ISOT demo |
| `fake_news_classification.ipynb` | Original classification case study |

## Why the 0.9963 SVM misleads

The classification notebook compares Count, TF-IDF, and Word2Vec features across
logistic regression, Naive Bayes, linear SVM, and random forest models. Its best
committed result is Count + linear SVM at `0.9963` test accuracy.

That number is not fact-check accuracy. A random ISOT article split leaks outlet
style between train and test. The retrieval project leaves the notebook in place
and does not retune it.

To run the notebook stack:

```bash
pip install -r requirements.txt
jupyter notebook fake_news_classification.ipynb
```

## Limitations

- The ISOT corpus is closed and historical. No web evidence is fetched.
- ISOT judgments are same-article title recovery, not independent qrels.
- The paraphrase rewrite does not remove enough lexical overlap to stress BM25.
- The committed SciFact test table is the BEIR comparison. Regenerating it needs `pip install -e ".[scifact]"` and the commands above. Tokenizer and BM25 parameters are named in `results/scifact_eval_meta.json`; they are not a guaranteed match to Elasticsearch's 0.665.
- The dense index is an in-memory brute-force cosine index. A few thousand vectors do not need an ANN service.
- Source-bucket labels can encode outlet and style artifacts.

## Data and license

`scripts/download_data.py` downloads the
[ISOT Fake News Dataset](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/).
SciFact is loaded on demand via `ir_datasets` (`beir/scifact`, CC BY-NC 2.0).
The large CSVs, BEIR cache, and generated indexes are not part of Git.

Code is released under the [MIT License](LICENSE.md). Dataset use remains subject
to the source dataset's terms.
