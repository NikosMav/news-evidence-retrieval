"""Committed numbers say what they mean. No model download."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd

from evidence_retrieval.cli import build_parser
from evidence_retrieval.eval.run import (
    _METRIC_COLS,
    build_isot_eval_meta,
    run_ablations,
)
from evidence_retrieval.eval.queries import content_word_retention, token_jaccard

ROOT = Path(__file__).resolve().parents[1]


def test_article_mrr_is_not_a_reported_column():
    assert "article_mrr" not in _METRIC_COLS
    for name in (
        "retrieval_metrics.csv",
        "retrieval_ablations.csv",
        "paraphrase_metrics.csv",
        "retrieval_eval_detail.csv",
        "paraphrase_eval_detail.csv",
    ):
        header = (ROOT / "results" / name).read_text(encoding="utf-8").splitlines()[0]
        assert "article_mrr" not in header.split(",")


def test_ablation_defaults_match_the_committed_run():
    assert inspect.signature(run_ablations).parameters["n_articles"].default == 2000
    assert inspect.signature(run_ablations).parameters["max_queries"].default == 150
    args = build_parser().parse_args(["eval"])
    assert args.ablation_articles == 2000
    assert args.ablation_queries == 150


def test_isot_meta_states_the_hit_edge_and_the_recall_confound():
    summary = pd.read_csv(ROOT / "results" / "retrieval_metrics.csv")
    ablations = pd.read_csv(ROOT / "results" / "retrieval_ablations.csv")
    meta = build_isot_eval_meta(
        summary,
        ablations,
        n_articles=4000,
        max_queries=300,
        ablation_articles=2000,
        ablation_queries=150,
    )
    assert "title-recovery sanity check" in meta["protocol"]
    assert "0.0067" in meta["interpretation"]["hit_at_1_note"]
    assert "Hit@10" in meta["interpretation"]["hit_at_1_note"]
    assert "every chunk of the gold article is relevant" in meta["interpretation"]["passage_recall_note"]
    assert "article_mrr was dropped" in meta["interpretation"]["article_mrr_note"]
    assert meta["ablations"]["n_articles"] == 2000
    assert meta["ablations"]["n_queries"] == 150
    committed = json.loads((ROOT / "results" / "retrieval_eval_meta.json").read_text(encoding="utf-8"))
    assert "0.0067" in committed["interpretation"]["hit_at_1_note"]
    assert "title-recovery" in committed["protocol"]


def test_committed_paraphrase_keeps_lexical_overlap():
    frame = pd.read_csv(ROOT / "results" / "paraphrase_queries.csv")
    jaccard = [
        token_jaccard(original, rewrite)
        for original, rewrite in zip(frame["original_title"], frame["paraphrase"])
    ]
    retention = [
        content_word_retention(original, rewrite)
        for original, rewrite in zip(frame["original_title"], frame["paraphrase"])
    ]
    assert sum(jaccard) / len(jaccard) > 0.75
    assert sum(retention) / len(retention) > 0.9
    meta = json.loads((ROOT / "results" / "paraphrase_eval_meta.json").read_text(encoding="utf-8"))
    assert meta["mean_token_jaccard"] > 0.75
    assert "rose" in meta["note"]
    assert "expected to drop" not in meta["note"]


def test_walkthrough_regenerates_the_scifact_test_split():
    """The SciFact script defaults to train. The test table needs --split test."""
    script = (ROOT / "scripts" / "run_scifact_eval.py").read_text(encoding="utf-8")
    assert 'default="train"' in script
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    notebook = (ROOT / "evidence_retrieval.ipynb").read_text(encoding="utf-8")
    builder = (ROOT / "scripts" / "build_retrieval_notebook.py").read_text(encoding="utf-8")
    command = "python scripts/run_scifact_eval.py --split test"
    assert command in readme
    for text in (notebook, builder):
        assert command in text
        assert "defaults to train" in text
        assert "Regenerate it with `python scripts/run_scifact_eval.py`." not in text


def test_scifact_failure_note_matches_the_per_query_csv():
    """The three README misses are the failure note, and those nDCG figures match the CSV."""
    import re

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    note = (ROOT / "results" / "scifact_failures.md").read_text(encoding="utf-8")
    assert "results/scifact_failures.md" in readme
    assert "methionine" in readme and "methionine" in note
    assert "vCJD" in readme and "vCJD" in note
    assert "low-birth-weight" in readme and "low birth weight" in note.lower()

    frame = pd.read_csv(ROOT / "results" / "scifact_test_per_query.csv", dtype={"query_id": str})
    claims = re.findall(r"Claim `(\d+)`:", note)
    assert claims == ["238", "48", "13"]
    sections = re.split(r"\n### ", note)
    for query_id, section in zip(claims, sections[1:]):
        subset = frame[frame["query_id"] == query_id].set_index("method")
        for method in ("bm25", "dense", "hybrid"):
            match = re.search(rf"- {method} nDCG@10=([0-9.]+);", section)
            assert match is not None, f"missing {method} line for claim {query_id}"
            reported = float(match.group(1))
            actual = float(subset.loc[method, "ndcg@10"])
            assert abs(reported - actual) < 5e-4


def test_readme_fusion_tail_matches_the_per_query_csv():
    """The README's fusion-tail counts are the committed test file, not a retelling."""
    frame = pd.read_csv(ROOT / "results" / "scifact_test_per_query.csv", dtype={"query_id": str})
    wide = frame.pivot(index="query_id", columns="method")
    ndcg = wide["ndcg@10"]
    recall = wide["recall@100"]
    reciprocal = wide["recip_rank"]
    drop = ndcg["hybrid"] - ndcg["bm25"]
    assert int((drop <= -0.5).sum()) == 10

    query = "785"
    assert float(ndcg.loc[query, "bm25"]) == 1.0
    assert float(ndcg.loc[query, "hybrid"]) == 0.0
    assert float(recall.loc[query, "hybrid"]) == 1.0
    assert abs(float(reciprocal.loc[query, "hybrid"]) - (1 / 18)) < 1e-12

    claim = "13"
    for method in ("dense", "hybrid"):
        assert float(ndcg.loc[claim, method]) == 0.0
        assert float(recall.loc[claim, method]) == 1.0
    assert int((recall["hybrid"] < recall["bm25"]).sum()) == 0
    worse_than_both = (ndcg["hybrid"] < ndcg["bm25"]) & (ndcg["hybrid"] < ndcg["dense"])
    assert int(worse_than_both.sum()) == 0

    ablations = pd.read_csv(ROOT / "results" / "scifact_test_ablations.csv").set_index("method")
    assert abs(float(ablations.loc["bge_hybrid", "recall@100"]) - 0.965) < 1e-12

    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    assert "10 queries drop by at least 0.5 nDCG@10 versus BM25" in readme
    assert "Query 785 goes from 1.0 to 0.0 while Recall@100 stays 1 (reciprocal rank 1/18)" in readme
    assert "outside the top 10 and inside the top 100 for MiniLM and hybrid" in readme
    assert "Hybrid loses Recall@100 to BM25 on 0 queries" in readme
    assert "worse than both parents on 0 test queries" in readme
    assert "BGE hybrid Recall@100 does move, to 0.9650" in readme
