# Dataset directory

ISOT CSVs live here for the closed-corpus demo of the same passage index.
They are not the score. That table is SciFact document ranking in the root README.

Place `True.csv` and `Fake.csv` here (ISOT Fake News Dataset).

```bash
python scripts/download_data.py
```

These CSV files are intentionally not committed (large text dumps).

SciFact (`beir/scifact`) is loaded on demand by `ir_datasets` and is not stored
in this directory.

The evidence-retrieval notebook may also create `retrieval_index/` here (cached
embeddings + chunk metadata). That cache is gitignored and safe to delete; the notebook
will rebuild it.
