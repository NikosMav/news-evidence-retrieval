"""BEIR-style metrics through pytrec_eval.

The in-repo formulas in ``metrics.py`` stay for unit tests of the ISOT proxy.
SciFact nDCG@10 and Recall@100 go through pytrec_eval so the number is the
same definition BEIR publishes.
"""

from __future__ import annotations

from typing import Iterable, Mapping

Qrels = Mapping[str, Mapping[str, int]]
Run = Mapping[str, Mapping[str, float]]

BEIR_MEASURES = ("ndcg_cut_10", "recall_100", "recip_rank")
MEASURE_COLUMNS = {
    "ndcg_cut_10": "ndcg@10",
    "recall_100": "recall@100",
    "recip_rank": "recip_rank",
}


def ranking_to_run(doc_ids: Iterable[str]) -> dict[str, float]:
    """Strictly decreasing scores so pytrec_eval keeps this order."""
    ordered = [str(doc_id) for doc_id in doc_ids]
    n = len(ordered)
    run: dict[str, float] = {}
    for rank, doc_id in enumerate(ordered):
        # First occurrence wins if a backend repeats an id.
        if doc_id not in run:
            run[doc_id] = float(n - rank)
    return run


def evaluate_run(
    qrels: Qrels,
    run: Run,
    measures: Iterable[str] = BEIR_MEASURES,
) -> dict[str, dict[str, float]]:
    """Per-query pytrec_eval scores for queries present in both qrels and run.

    Every judged query should appear in ``run``, even with an empty document
    map. pytrec_eval omits queries that are missing from the run, which would
    inflate the mean.
    """
    import pytrec_eval

    measure_set = set(measures)
    judged = {qid: dict(rels) for qid, rels in qrels.items() if rels}
    evaluator = pytrec_eval.RelevanceEvaluator(judged, measure_set)
    per_query = evaluator.evaluate({qid: dict(docs) for qid, docs in run.items()})
    return {qid: {name: float(scores[name]) for name in measure_set} for qid, scores in per_query.items()}


def mean_measures(per_query: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    if not per_query:
        return {}
    names = list(next(iter(per_query.values())).keys())
    return {
        name: float(sum(row[name] for row in per_query.values()) / len(per_query))
        for name in names
    }


def paired_sign_test(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> dict[str, float | int]:
    """Two-sided sign test on paired per-query scores (left minus right).

    Ties are dropped. ``n_pos`` counts queries where ``left`` is higher.
    """
    from scipy.stats import binomtest

    keys = sorted(set(left) & set(right))
    n_pos = 0
    n_neg = 0
    for key in keys:
        diff = float(left[key]) - float(right[key])
        if diff > 0:
            n_pos += 1
        elif diff < 0:
            n_neg += 1
    n = n_pos + n_neg
    ties = len(keys) - n
    if n == 0:
        pvalue = 1.0
    else:
        pvalue = float(binomtest(n_pos, n, 0.5, alternative="two-sided").pvalue)
    return {
        "n_queries": len(keys),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_ties": ties,
        "pvalue": pvalue,
    }
