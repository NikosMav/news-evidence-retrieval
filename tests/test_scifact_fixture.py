"""SciFact comparison on a synthetic corpus. Does not import ir_datasets or torch."""

from __future__ import annotations

import sys

import pandas as pd

from evidence_retrieval.eval.scifact import apply_rerank, run_document_comparison
from tests.conftest import FakeDenseEncoder


def _corpus() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    documents = pd.DataFrame(
        [
            {
                "doc_id": "1",
                "title": "Kinase inhibitors",
                "text": "Selective kinase inhibitors reduced tumor growth in the mouse model.",
            },
            {
                "doc_id": "2",
                "title": "Glacier mass",
                "text": "Glacier mass balance declined after warmer summer temperatures.",
            },
            {
                "doc_id": "3",
                "title": "Vaccine trial",
                "text": "A peptide vaccine increased antibody titers in healthy volunteers.",
            },
            {
                "doc_id": "4",
                "title": "Unrelated kinase mention",
                "text": "The city council discussed parking meters and street lighting.",
            },
            {
                "doc_id": "5",
                "title": "Ocean heat",
                "text": "Ocean heat content rose and glaciers continued to lose mass.",
            },
        ]
    )
    queries = pd.DataFrame(
        [
            {"query_id": "q1", "text": "Do kinase inhibitors reduce tumor growth?"},
            {"query_id": "q2", "text": "Are glaciers losing mass as summers warm?"},
            {"query_id": "q3", "text": "This claim has no judgment and must be ignored."},
        ]
    )
    qrels = {
        "q1": {"1": 1},
        "q2": {"2": 1, "5": 1},
    }
    return documents, queries, qrels


def test_fixture_comparison_does_not_download_beir():
    sys.modules.pop("ir_datasets", None)
    documents, queries, qrels = _corpus()
    result = run_document_comparison(
        documents,
        queries,
        qrels,
        dense_encoder=FakeDenseEncoder(dim=32),
        k=4,
        show_progress=False,
    )
    assert "ir_datasets" not in sys.modules
    summary = result["summary"].set_index("method")
    assert list(summary.index) == ["bm25", "dense", "hybrid"]
    assert summary.loc["bm25", "n_queries"] == 2
    assert set(summary.columns) >= {"ndcg@10", "recall@100", "recip_rank"}
    # The kinase abstract shares the claim's content words, so BM25 should hit it.
    assert summary.loc["bm25", "recall@100"] == 1.0
    assert summary.loc["bm25", "ndcg@10"] > 0.5
    assert "dense" in result["sign_tests"]
    assert result["sign_tests"]["dense"]["compared_to"] == "bm25"
    assert "torch" not in sys.modules


def test_rerank_reorders_only_the_head_and_keeps_ties():
    assert apply_rerank(["a", "b", "c", "d"], [0.1, 0.2, 0.9], depth=3) == ["c", "b", "a", "d"]
    assert apply_rerank(["a", "b", "c"], [1.0, 1.0, 0.0], depth=3) == ["a", "b", "c"]
