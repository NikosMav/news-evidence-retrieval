"""BM25 encoder and index path. No BEIR download and no torch."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from evidence_retrieval.encoders import BM25Encoder
from evidence_retrieval.index import IndexConfig, PassageIndex
from tests.conftest import FakeDenseEncoder


def test_bm25_ranks_the_matching_document_first():
    corpus = [
        "school lunch menu vegetables allergy labeling cafeteria",
        "federal reserve raised interest rates amid persistent inflation",
        "local park concert schedule and outdoor seating",
    ]
    encoder = BM25Encoder().fit(corpus, show_progress=False)
    ranked = encoder.rank("interest rates and inflation", fetch=3)
    assert ranked[0][0] == 1
    assert ranked[0][1] > ranked[1][1]


def test_bm25_save_load_roundtrip(tmp_path: Path):
    corpus = [
        "mitochondria produce ATP in eukaryotic cells",
        "glaciers retreat when summer temperatures rise",
    ]
    encoder = BM25Encoder().fit(corpus, show_progress=False)
    encoder.save(tmp_path / "bm25")
    loaded = BM25Encoder.load(tmp_path / "bm25")
    original = encoder.rank("mitochondria ATP", fetch=2)
    restored = loaded.rank("mitochondria ATP", fetch=2)
    assert [idx for idx, _ in restored] == [idx for idx, _ in original]
    assert restored[0][0] == 0


def test_index_bm25_hybrid_and_reload(tmp_path: Path):
    passages = pd.DataFrame(
        [
            {
                "chunk_id": "1",
                "title": "Rates",
                "text": "the federal reserve raised interest rates to fight inflation",
            },
            {
                "chunk_id": "2",
                "title": "Lunch",
                "text": "the school board approved a lunch menu with more vegetables",
            },
            {
                "chunk_id": "3",
                "title": "Shares",
                "text": "tech shares climbed after inflation data cooled interest rate fears",
            },
        ]
    )
    # from_passages expects doc-style columns only when used via scifact helper.
    # Here we pass the passage schema directly.
    passages = passages.rename(columns={"text": "passage"})
    config = IndexConfig(
        n_articles=3,
        chunk_words=40,
        overlap=5,
        sparse_backend="bm25",
        dense_model="fake-minilm",
    )
    index = PassageIndex.from_passages(
        passages,
        config=config,
        dense_encoder=FakeDenseEncoder(),
        show_progress=False,
    )
    assert index.sparse is None
    assert index.bm25 is not None
    hits = index.query("federal reserve interest rates", top_k=2, method="bm25")
    assert hits[0].chunk_id == "1"
    assert hits[0].method == "bm25"
    hybrid = index.query("federal reserve interest rates", top_k=2, method="hybrid")
    assert hybrid[0].method == "hybrid"
    assert hybrid[0].chunk_id in {"1", "3"}
    with pytest.raises(RuntimeError, match="TF-IDF"):
        index.query("federal reserve", top_k=1, method="tfidf")

    out = index.save(tmp_path / "idx")
    loaded = PassageIndex.load(out)
    loaded.dense_encoder = FakeDenseEncoder()
    again = loaded.query("lunch menu vegetables", top_k=1, method="bm25")
    assert again[0].chunk_id == "2"


def test_sparse_backend_rejects_unknown_name():
    with pytest.raises(ValueError, match="sparse_backend"):
        IndexConfig(sparse_backend="splade")  # type: ignore[arg-type]
