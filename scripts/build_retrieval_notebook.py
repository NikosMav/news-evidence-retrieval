#!/usr/bin/env python3
"""Rebuild evidence_retrieval.ipynb as a thin walkthrough over the Python package."""

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


def build() -> nbf.NotebookNode:
    cells = [
        md(
            """
# Evidence Retrieval Walkthrough

Closed-corpus ISOT demo over the `evidence_retrieval` package: build (or load) a
**tiny** local index, run one query, print ranked hits.

The retrieval claim is the SciFact test table in the root **README**.
Regenerate it with `python scripts/run_scifact_eval.py --split test`
(the script defaults to train). The ISOT
title-recovery script (`python scripts/run_retrieval_eval.py`) is a sanity
check, not that table.

**Not a fact-checker.** ISOT labels are source buckets; nearest neighbors ≠ verification.
"""
        ),
        md(
            """
## Build or load a tiny index

Uses a small article sample and `data/retrieval_index/tiny`. Needs ISOT CSVs in
`data/` (`python scripts/download_data.py`) and a one-time MiniLM download on first build.
"""
        ),
        code(
            """
from pathlib import Path
from evidence_retrieval.index import IndexConfig, PassageIndex

INDEX_DIR = Path("data/retrieval_index/tiny")

if (INDEX_DIR / "chunks.parquet").exists():
    index = PassageIndex.load(INDEX_DIR)
    print(f"Loaded tiny index: {len(index.chunks):,} passages")
else:
    config = IndexConfig(n_articles=200, chunk_words=120, fields=("body",))
    index = PassageIndex.build(data_dir="data", config=config, show_progress=True)
    index.save(INDEX_DIR)
    print(f"Built + saved tiny index → {INDEX_DIR} ({len(index.chunks):,} passages)")
"""
        ),
        md("## One query → ranked hits"),
        code(
            """
query = "Federal Reserve interest rate decision and markets"
hits = index.query_df(query, top_k=5, method="hybrid")
cols = ["rank", "score", "label_name", "title", "passage"]
print(f"query: {query!r} | method: hybrid | hits: {len(hits)}")
print(hits[cols].to_string(index=False, max_colwidth=72))
"""
        ),
        md(
            """
## CLI equivalent

```bash
python -m evidence_retrieval build --n-articles 200 --out data/retrieval_index/tiny
python -m evidence_retrieval query "Federal Reserve interest rate decision and markets" \\
  --index data/retrieval_index/tiny --top-k 5
```
"""
        ),
    ]
    nb = new_notebook(cells=cells)
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    return nb


def main():
    out = Path(__file__).resolve().parents[1] / "evidence_retrieval.ipynb"
    nb = build()
    nbf.write(nb, out)
    print(f"Wrote {out} with {len(nb.cells)} cells")


if __name__ == "__main__":
    main()
