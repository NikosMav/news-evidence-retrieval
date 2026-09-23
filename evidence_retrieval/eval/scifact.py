"""SciFact document retrieval: BM25, MiniLM, and RRF.

Loads ``beir/scifact/train`` or ``beir/scifact/test`` through ``ir_datasets``
only when a split is requested. The comparison itself takes plain tables so
unit tests never download BEIR.

Dev on the train split. The test split is the number next to BEIR's BM25
nDCG@10 of 0.665; do not use it to choose a metric or a tokenizer.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from evidence_retrieval.encoders import DEFAULT_BM25_B, DEFAULT_BM25_K1, DEFAULT_DENSE_MODEL
from evidence_retrieval.eval.trec import (
    BEIR_MEASURES,
    MEASURE_COLUMNS,
    evaluate_run,
    mean_measures,
    paired_sign_test,
    ranking_to_run,
)
from evidence_retrieval.index import IndexConfig, PassageIndex

SCIFACT_DATASET_IDS = {
    "train": "beir/scifact/train",
    "test": "beir/scifact/test",
}
# Published anchors, not numbers this repo measured.
BEIR_BM25_NDCG_AT_10 = 0.665
ANSERINI_FLAT_BM25_NDCG_AT_10 = 0.6789


def load_scifact_split(split: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]]]:
    """Document corpus, queries, and binary-or-graded qrels for one BEIR split.

    ``ir_datasets`` is imported here so importing this module does not download
    anything and does not require the package in CI.
    """
    if split not in SCIFACT_DATASET_IDS:
        raise ValueError(f"split must be one of {sorted(SCIFACT_DATASET_IDS)}")
    import ir_datasets

    dataset = ir_datasets.load(SCIFACT_DATASET_IDS[split])
    documents = []
    for doc in dataset.docs_iter():
        doc_id = str(doc.doc_id)
        title = str(getattr(doc, "title", "") or "")
        text = str(getattr(doc, "text", "") or "")
        if hasattr(doc, "default_text"):
            passage = str(doc.default_text())
        else:
            passage = f"{title} {text}".strip()
        documents.append({"doc_id": doc_id, "title": title, "text": passage})

    queries = []
    for query in dataset.queries_iter():
        queries.append({"query_id": str(query.query_id), "text": str(query.text)})

    qrels: dict[str, dict[str, int]] = {}
    for qrel in dataset.qrels_iter():
        relevance = int(qrel.relevance)
        if relevance <= 0:
            continue
        qrels.setdefault(str(qrel.query_id), {})[str(qrel.doc_id)] = relevance

    return pd.DataFrame(documents), pd.DataFrame(queries), qrels


def run_document_comparison(
    documents: pd.DataFrame,
    queries: pd.DataFrame,
    qrels: dict[str, dict[str, int]],
    *,
    dense_encoder,
    k: int = 100,
    bm25_k1: float = DEFAULT_BM25_K1,
    bm25_b: float = DEFAULT_BM25_B,
    dense_model: str = DEFAULT_DENSE_MODEL,
    show_progress: bool = False,
    methods: Iterable[str] = ("bm25", "dense", "hybrid"),
) -> dict:
    """Rank whole documents with BM25, a dense encoder, and RRF of those two.

    ``documents`` columns: ``doc_id``, ``text`` (already the indexed string),
    optional ``title``. ``queries`` columns: ``query_id``, ``text``.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    method_list = tuple(methods)
    unknown = set(method_list) - {"bm25", "dense", "hybrid"}
    if unknown:
        raise ValueError(f"Unsupported SciFact methods: {sorted(unknown)}")

    passages = _documents_to_passages(documents)
    config = IndexConfig(
        n_articles=max(1, len(passages)),
        chunk_words=120,
        overlap=20,
        fields=("body",),
        dense_model=dense_model,
        sparse_backend="bm25",
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
    )
    index = PassageIndex.from_passages(
        passages,
        config=config,
        dense_encoder=dense_encoder,
        show_progress=show_progress,
    )

    judged = {qid: rels for qid, rels in qrels.items() if rels}
    query_rows = queries[queries["query_id"].astype(str).isin(judged)]
    runs: dict[str, dict[str, dict[str, float]]] = {method: {} for method in method_list}
    for row in query_rows.itertuples(index=False):
        query_id = str(row.query_id)
        text = str(row.text).strip()
        for method in method_list:
            if not text:
                runs[method][query_id] = {}
                continue
            hits = index.query(text, top_k=k, method=method)  # type: ignore[arg-type]
            runs[method][query_id] = ranking_to_run(hit.chunk_id for hit in hits)

    # Judged queries that produced no row still count as empty runs.
    for method in method_list:
        for query_id in judged:
            runs[method].setdefault(query_id, {})

    per_method = {}
    summary_rows = []
    per_query_rows = []
    for method in method_list:
        scores = evaluate_run(judged, runs[method], BEIR_MEASURES)
        per_method[method] = scores
        means = mean_measures(scores)
        summary_rows.append(
            {
                "method": method,
                "n_queries": len(scores),
                **{MEASURE_COLUMNS[name]: means.get(name, 0.0) for name in BEIR_MEASURES},
            }
        )
        for query_id, row in scores.items():
            per_query_rows.append(
                {
                    "query_id": query_id,
                    "method": method,
                    **{MEASURE_COLUMNS[name]: row[name] for name in BEIR_MEASURES if name in row},
                }
            )

    sign_tests = {}
    if "bm25" in per_method:
        bm25_ndcg = {
            query_id: row["ndcg_cut_10"] for query_id, row in per_method["bm25"].items()
        }
        for method in method_list:
            if method == "bm25":
                continue
            other = {
                query_id: row["ndcg_cut_10"] for query_id, row in per_method[method].items()
            }
            sign_tests[method] = paired_sign_test(other, bm25_ndcg)
            sign_tests[method]["metric"] = "ndcg@10"
            sign_tests[method]["compared_to"] = "bm25"

    summary = pd.DataFrame(summary_rows)
    return {
        "summary": summary,
        "per_query": pd.DataFrame(per_query_rows),
        "sign_tests": sign_tests,
        "k": k,
        "bm25_k1": bm25_k1,
        "bm25_b": bm25_b,
        "dense_model": dense_model,
        "n_documents": int(len(documents)),
        "n_judged_queries": len(judged),
    }


def _documents_to_passages(documents: pd.DataFrame) -> pd.DataFrame:
    if "doc_id" not in documents.columns or "text" not in documents.columns:
        raise ValueError("documents must include doc_id and text columns")
    frame = documents.copy()
    frame["chunk_id"] = frame["doc_id"].astype(str)
    frame["passage"] = frame["text"].astype(str)
    if "title" in frame.columns:
        frame["title"] = frame["title"].fillna("").astype(str)
    keep = ["chunk_id", "passage"] + (["title"] if "title" in frame.columns else [])
    return frame[keep]
