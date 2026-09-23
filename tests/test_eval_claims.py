"""Committed ISOT numbers say what they mean. No model download."""

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
