# Retrieval eval outputs

The headline result is the SciFact test table (`beir/scifact/test`, 300 queries).
ISOT files further down are a closed-corpus title-recovery demo.

| Method | nDCG@10 | Recall@100 |
| --- | ---: | ---: |
| bm25 | 0.6762 | 0.9127 |
| dense (MiniLM) | 0.6451 | 0.9250 |
| hybrid | 0.7194 | 0.9550 |

Published anchors: BEIR BM25 nDCG@10 **0.665**, Anserini flat BM25 **0.6789**.

`BAAI/bge-small-en-v1.5` moves dense nDCG@10 from 0.6451 to 0.7127. The same
encoder inside RRF does not move hybrid (0.7189 vs 0.7194). Reranking hybrid's
top 50 with `cross-encoder/ms-marco-MiniLM-L-6-v2` moves nDCG@10 from 0.7194
to 0.6903. Three misses from the frozen three-way run are in `scifact_failures.md`.

Regenerate SciFact, then refresh the README tables:

```bash
python scripts/run_scifact_eval.py --split train
python scripts/run_scifact_eval.py --split test
python scripts/run_scifact_ablations.py --split test
python scripts/update_readme_metrics.py
```

Those scripts do not rewrite the ISOT tables.

| File | Contents |
| --- | --- |
| `scifact_test_metrics.csv` | BM25 vs MiniLM vs RRF on `beir/scifact/test` |
| `scifact_test_per_query.csv` | Per-query scores for that frozen test run |
| `scifact_test_ablations.csv` | Cross-encoder rerank and `bge-small-en-v1.5` |
| `scifact_train_metrics.csv` | Same three-way comparison on `beir/scifact/train` |
| `scifact_train_per_query.csv` | Per-query scores for the train split |
| `scifact_failures.md` | Three test misses from the frozen three-way run |
| `scifact_eval_meta.json` | Tokenizer, anchors, sign tests, and regenerate commands |
| `retrieval_metrics.csv` | TF-IDF vs dense vs hybrid on title-recovery queries |
| `retrieval_eval_meta.json` | What those numbers mean (Hit@1 edge, chunk-size recall, dropped `article_mrr`) |
| `paraphrase_metrics.csv` | Same metrics on rule-based title rewrites |
| `paraphrase_queries.csv` | Original title vs paraphrase + gold ids |
| `paraphrase_eval_detail.csv` | Per-query detail for the paraphrase protocol |
| `paraphrase_eval_meta.json` | Overlap stats; the rewrite is not a robustness result |
| `retrieval_ablations.csv` | Chunk size / field / method ablations |
| `retrieval_eval_detail.csv` | Per-query ranks + leakage signals |
| `qualitative_failures.md` | Sampled title-recovery misses |
| `eval_queries.csv` | Query set used for the ISOT demo table |
| `workflow_comparison.csv` | Optional classify-then vs retrieve-first demo rows |

`article_mrr` is not a column. With one gold article it copied passage MRR.

Do not hand-edit metric CSVs. Regenerate the ISOT demo with
`python scripts/run_retrieval_eval.py` or `python scripts/run_paraphrase_eval.py`,
then `python scripts/update_readme_metrics.py` (also invoked at the end of `eval`).
Paraphrase eval never rewrites `retrieval_metrics.csv`.
