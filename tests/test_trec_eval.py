"""pytrec_eval wiring. No dataset download."""

from __future__ import annotations

from evidence_retrieval.eval.trec import (
    evaluate_run,
    mean_measures,
    paired_sign_test,
    ranking_to_run,
)


def test_ranking_to_run_is_strictly_decreasing_and_dedupes():
    run = ranking_to_run(["a", "b", "a"])
    assert run["a"] > run["b"]
    assert set(run) == {"a", "b"}


def test_perfect_and_miss_match_pytrec_definitions():
    qrels = {
        "q1": {"d1": 1},
        "q2": {"d1": 1},
    }
    run = {
        "q1": ranking_to_run(["d1", "d9"]),
        "q2": ranking_to_run([f"other{i}" for i in range(100)]),
    }
    scores = evaluate_run(qrels, run)
    assert scores["q1"]["ndcg_cut_10"] == 1.0
    assert scores["q1"]["recall_100"] == 1.0
    assert scores["q2"]["ndcg_cut_10"] == 0.0
    assert scores["q2"]["recall_100"] == 0.0
    means = mean_measures(scores)
    assert means["ndcg_cut_10"] == 0.5
    assert means["recall_100"] == 0.5


def test_relevant_at_rank_2_is_not_perfect_ndcg():
    qrels = {"q": {"d1": 1}}
    scores = evaluate_run(qrels, {"q": ranking_to_run(["miss", "d1"])})
    assert 0.0 < scores["q"]["ndcg_cut_10"] < 1.0
    assert scores["q"]["recall_100"] == 1.0
    assert scores["q"]["recip_rank"] == 0.5


def test_missing_query_is_not_silently_dropped_when_present_as_empty():
    qrels = {"q1": {"d1": 1}, "q2": {"d1": 1}}
    scores = evaluate_run(qrels, {"q1": ranking_to_run(["d1"]), "q2": {}})
    assert set(scores) == {"q1", "q2"}
    assert scores["q2"]["ndcg_cut_10"] == 0.0


def test_paired_sign_test_detects_a_consistent_winner():
    left = {str(i): 1.0 for i in range(12)}
    right = {str(i): 0.0 for i in range(12)}
    result = paired_sign_test(left, right)
    assert result["n_pos"] == 12
    assert result["n_neg"] == 0
    assert result["pvalue"] < 0.01

    tied = paired_sign_test({"a": 1.0, "b": 1.0}, {"a": 1.0, "b": 1.0})
    assert tied["n_ties"] == 2
    assert tied["pvalue"] == 1.0
