"""ISOT closed-corpus title-recovery demo: metrics, ablations, and CSV output.

SciFact scoring lives in ``evidence_retrieval.eval.scifact``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from evidence_retrieval.data import load_isot, stratified_sample
from evidence_retrieval.eval.metrics import (
    aggregate_metrics,
    mrr,
    ndcg_at_k,
    recall_at_k,
    recall_at_k_binary,
)
from evidence_retrieval.eval.queries import (
    build_title_queries,
    content_word_retention,
    paraphrase_queries_to_frame,
    token_jaccard,
)
from evidence_retrieval.index import IndexConfig, Method, PassageIndex

KS = (1, 5, 10)

_METRIC_COLS = [
    "method",
    "n_queries",
    "article_hit@1",
    "article_hit@5",
    "article_hit@10",
    "passage_recall@1",
    "passage_recall@5",
    "passage_recall@10",
    "ndcg@1",
    "ndcg@5",
    "ndcg@10",
    "mrr",
]


def _evaluate_method(
    index: PassageIndex,
    queries,
    method: Method,
    ks: Sequence[int] = KS,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Return mean metrics + per-query detail frame."""
    per_query = []
    detail_rows = []
    max_k = max(ks)

    for q in queries:
        hits = index.query(
            q.query_text,
            top_k=max_k,
            method=method,
            # Keep self in the pool — gold passages are from this article.
            exclude_article_ids=None,
        )
        ranked_chunks = [h.chunk_id for h in hits]
        ranked_articles = [str(h.article_id) for h in hits]
        gold_chunks = set(q.gold_chunk_ids)
        gold_articles = {str(a) for a in q.gold_article_ids}

        row: dict[str, float] = {}
        for k in ks:
            row[f"passage_recall@{k}"] = recall_at_k(gold_chunks, ranked_chunks, k)
            row[f"article_hit@{k}"] = recall_at_k_binary(
                gold_articles, ranked_articles, k
            )
            row[f"ndcg@{k}"] = ndcg_at_k(gold_chunks, ranked_chunks, k)
        row["mrr"] = mrr(gold_chunks, ranked_chunks)
        # article_mrr is not recorded. With one gold article it is the rank of
        # the first gold chunk, which is already `mrr`.
        per_query.append(row)

        # Hard-negative / leakage signals among top-5
        top5 = hits[:5]
        opposite = sum(1 for h in top5 if h.label_name != q.label_name)
        same_subject_other = sum(
            1
            for h in top5
            if h.subject == q.subject and h.article_id != q.article_id
        )
        detail_rows.append(
            {
                "query_id": q.query_id,
                "method": method,
                "label_name": q.label_name,
                "subject": q.subject,
                "title": q.title,
                "top1_article_id": top5[0].article_id if top5 else None,
                "top1_label": top5[0].label_name if top5 else None,
                "top1_score": top5[0].score if top5 else None,
                "top1_title": top5[0].title if top5 else None,
                "gold_in_top5": row["article_hit@5"],
                "opposite_label_in_top5": opposite,
                "same_subject_other_in_top5": same_subject_other,
                "mrr": row["mrr"],
            }
        )

    means = aggregate_metrics(per_query)
    means["method"] = method  # type: ignore[assignment]
    means["n_queries"] = float(len(queries))
    return means, pd.DataFrame(detail_rows)


def run_main_comparison(
    data_dir: Path | str = "data",
    n_articles: int = 4000,
    chunk_words: int = 120,
    overlap: int = 20,
    max_queries: int = 300,
    random_state: int = 7,
    methods: Sequence[Method] = ("tfidf", "dense", "hybrid"),
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, PassageIndex]:
    """Build default body index; evaluate TF-IDF vs dense vs hybrid on SAME queries."""
    config = IndexConfig(
        n_articles=n_articles,
        chunk_words=chunk_words,
        overlap=overlap,
        fields=("body",),
        random_state=random_state,
    )
    index = PassageIndex.build(
        data_dir=data_dir, config=config, show_progress=show_progress
    )
    queries = build_title_queries(
        index.chunks, max_queries=max_queries, random_state=random_state
    )

    summary_rows = []
    details = []
    for method in methods:
        means, detail = _evaluate_method(index, queries, method)
        summary_rows.append(means)
        details.append(detail)

    summary = pd.DataFrame(summary_rows)
    detail = pd.concat(details, ignore_index=True)
    return summary, detail, index


def run_ablations(
    data_dir: Path | str = "data",
    n_articles: int = 2000,
    max_queries: int = 150,
    random_state: int = 7,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Chunk-size, field, and method ablations on a shared article sample."""
    df = load_isot(data_dir)
    articles = stratified_sample(df, n_articles=n_articles, random_state=random_state)

    experiments = [
        # chunk size (body, hybrid)
        {"name": "chunk60_hybrid", "chunk_words": 60, "fields": ("body",), "method": "hybrid"},
        {"name": "chunk120_hybrid", "chunk_words": 120, "fields": ("body",), "method": "hybrid"},
        {"name": "chunk240_hybrid", "chunk_words": 240, "fields": ("body",), "method": "hybrid"},
        # title vs body (dense)
        {"name": "title_only_dense", "chunk_words": 120, "fields": ("title",), "method": "dense"},
        {"name": "body_dense", "chunk_words": 120, "fields": ("body",), "method": "dense"},
        {"name": "title_body_dense", "chunk_words": 120, "fields": ("title_body",), "method": "dense"},
        # methods at fixed chunk120 body
        {"name": "body120_tfidf", "chunk_words": 120, "fields": ("body",), "method": "tfidf"},
        {"name": "body120_dense", "chunk_words": 120, "fields": ("body",), "method": "dense"},
        {"name": "body120_hybrid", "chunk_words": 120, "fields": ("body",), "method": "hybrid"},
    ]

    # Cache indexes by (chunk_words, fields)
    index_cache: dict[tuple, PassageIndex] = {}
    rows = []

    for exp in experiments:
        key = (exp["chunk_words"], exp["fields"])
        if key not in index_cache:
            cfg = IndexConfig(
                n_articles=n_articles,
                chunk_words=int(exp["chunk_words"]),
                overlap=20,
                fields=tuple(exp["fields"]),
                random_state=random_state,
            )
            index_cache[key] = PassageIndex.build(
                data_dir=data_dir,
                config=cfg,
                articles=articles,
                show_progress=show_progress,
            )
        index = index_cache[key]
        queries = build_title_queries(
            index.chunks, max_queries=max_queries, random_state=random_state
        )
        # Title-only index: gold = title chunk of same article
        if exp["fields"] == ("title",):
            # rebuild gold as the title chunk ids
            from evidence_retrieval.eval.queries import EvalQuery

            fixed = []
            for q in queries:
                gold = index.chunks.loc[
                    index.chunks["article_id"] == q.article_id, "chunk_id"
                ].astype(str).tolist()
                if not gold:
                    continue
                fixed.append(
                    EvalQuery(
                        query_id=q.query_id,
                        article_id=q.article_id,
                        query_text=q.query_text,
                        gold_chunk_ids=gold,
                        gold_article_ids=[q.article_id],
                        label_name=q.label_name,
                        subject=q.subject,
                        title=q.title,
                    )
                )
            queries = fixed

        means, _ = _evaluate_method(index, queries, exp["method"])  # type: ignore[arg-type]
        rows.append(
            {
                "experiment": exp["name"],
                "chunk_words": exp["chunk_words"],
                "fields": ",".join(exp["fields"]),
                "method": exp["method"],
                "n_queries": means["n_queries"],
                "article_hit@1": means["article_hit@1"],
                "article_hit@5": means["article_hit@5"],
                "article_hit@10": means["article_hit@10"],
                "passage_recall@5": means["passage_recall@5"],
                "ndcg@5": means["ndcg@5"],
                "ndcg@10": means["ndcg@10"],
                "mrr": means["mrr"],
            }
        )

    return pd.DataFrame(rows)


def run_paraphrase_comparison(
    data_dir: Path | str = "data",
    n_articles: int = 4000,
    chunk_words: int = 120,
    overlap: int = 20,
    max_queries: int = 300,
    random_state: int = 7,
    methods: Sequence[Method] = ("tfidf", "dense", "hybrid"),
    show_progress: bool = True,
    index: PassageIndex | None = None,
    index_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list, PassageIndex]:
    """Evaluate TF-IDF vs dense vs hybrid on paraphrased title queries.

    Uses the same body-index setup as the main title self-retrieval protocol when
    building from scratch. Pass ``index`` or ``index_dir`` to reuse an existing
    index (does not rewrite ``retrieval_metrics.csv``).
    """
    from evidence_retrieval.eval.queries import build_paraphrase_queries

    if index is None and index_dir is not None and Path(index_dir).exists():
        index = PassageIndex.load(index_dir)
    if index is None:
        config = IndexConfig(
            n_articles=n_articles,
            chunk_words=chunk_words,
            overlap=overlap,
            fields=("body",),
            random_state=random_state,
        )
        index = PassageIndex.build(
            data_dir=data_dir, config=config, show_progress=show_progress
        )

    queries = build_paraphrase_queries(
        index.chunks, max_queries=max_queries, random_state=random_state
    )

    summary_rows = []
    details = []
    for method in methods:
        means, detail = _evaluate_method(index, queries, method)
        summary_rows.append(means)
        details.append(detail)

    summary = pd.DataFrame(summary_rows)
    detail = pd.concat(details, ignore_index=True)
    return summary, detail, queries, index


def save_paraphrase_bundle(
    summary: pd.DataFrame,
    detail: pd.DataFrame,
    queries,
    results_dir: Path | str = "results",
) -> dict[str, Path]:
    """Write paraphrase metrics / queries / meta. Never touches retrieval_metrics.csv."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "metrics": results_dir / "paraphrase_metrics.csv",
        "detail": results_dir / "paraphrase_eval_detail.csv",
        "queries": results_dir / "paraphrase_queries.csv",
        "meta": results_dir / "paraphrase_eval_meta.json",
    }
    summary = summary[[c for c in _METRIC_COLS if c in summary.columns]]
    summary.to_csv(paths["metrics"], index=False)
    detail.to_csv(paths["detail"], index=False)
    paraphrase_queries_to_frame(queries).to_csv(paths["queries"], index=False)

    overlap = _paraphrase_overlap(queries)
    meta = {
        "protocol": (
            "Rule-based rewrite of an article title; gold is still every indexed "
            "chunk of that same article. Same-article title recovery, not claim "
            "verification, and not a lexical stress test."
        ),
        "main_comparison_ref": "results/retrieval_metrics.csv",
        "n_queries": int(len(queries)),
        "methods": summary["method"].tolist() if "method" in summary.columns else [],
        "metrics_file": str(paths["metrics"]),
        "queries_file": str(paths["queries"]),
        "mean_token_jaccard": overlap["mean_token_jaccard"],
        "content_word_retention": overlap["content_word_retention"],
        "note": _paraphrase_note(summary, overlap, results_dir / "retrieval_metrics.csv"),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return paths


def _paraphrase_overlap(queries) -> dict[str, float]:
    if not queries:
        return {"mean_token_jaccard": 0.0, "content_word_retention": 0.0}
    jaccard_values = [
        token_jaccard(q.title, q.query_text) for q in queries
    ]
    retention_values = [
        content_word_retention(q.title, q.query_text) for q in queries
    ]
    return {
        "mean_token_jaccard": float(sum(jaccard_values) / len(jaccard_values)),
        "content_word_retention": float(sum(retention_values) / len(retention_values)),
    }


def _paraphrase_note(
    summary: pd.DataFrame,
    overlap: dict[str, float],
    main_metrics_path: Path,
) -> str:
    note = (
        "The rewrite did not cut lexical overlap enough to be a stress test. "
        f"Mean token Jaccard is {overlap['mean_token_jaccard']:.3f} and "
        f"content-word retention is {overlap['content_word_retention']:.3f}."
    )
    rise = _tfidf_hit1_change(summary, main_metrics_path)
    if rise is not None:
        old, new = rise
        if new > old:
            note += (
                f" TF-IDF article Hit@1 rose from {old:.4f} to {new:.4f}."
            )
        elif new < old:
            note += (
                f" TF-IDF article Hit@1 moved from {old:.4f} to {new:.4f}."
            )
        else:
            note += f" TF-IDF article Hit@1 stayed at {new:.4f}."
    note += " Do not read this table as a robustness result."
    return note


def _tfidf_hit1_change(
    summary: pd.DataFrame,
    main_metrics_path: Path,
) -> tuple[float, float] | None:
    if summary is None or summary.empty or "method" not in summary.columns:
        return None
    if not main_metrics_path.exists():
        return None
    main = pd.read_csv(main_metrics_path)
    if "article_hit@1" not in summary.columns or "article_hit@1" not in main.columns:
        return None
    new_rows = summary[summary["method"] == "tfidf"]
    old_rows = main[main["method"] == "tfidf"]
    if new_rows.empty or old_rows.empty:
        return None
    return float(old_rows["article_hit@1"].iloc[0]), float(new_rows["article_hit@1"].iloc[0])


def write_qualitative_failures(
    detail: pd.DataFrame,
    out_path: Path | str,
    n_examples: int = 8,
) -> Path:
    """Document failure modes: misses, opposite-label leakage, same-subject collapse."""
    out_path = Path(out_path)
    # Prefer hybrid rows if present
    df = detail[detail["method"] == "hybrid"] if (detail["method"] == "hybrid").any() else detail

    misses = df[df["gold_in_top5"] < 1.0].head(n_examples)
    leakage = df[df["opposite_label_in_top5"] >= 2].head(n_examples)
    style = df[df["same_subject_other_in_top5"] >= 2].head(n_examples)

    lines = [
        "# Qualitative failure cases (auto-sampled from eval detail)",
        "",
        "These examples come from the ISOT title-recovery sanity check "
        "(query = article title, gold = that article's passages).",
        "They are misses against finding the source article, not against finding evidence for a claim.",
        "",
        "## Misses (gold article absent from top-5)",
        "",
    ]
    if misses.empty:
        lines.append("_No misses in the sampled hybrid detail (strong self-retrieval)._")
    else:
        for _, r in misses.iterrows():
            lines.append(
                f"- **Query title:** {r['title']!r}  \n"
                f"  Gold bucket: `{r['label_name']}` / subject `{r['subject']}`.  \n"
                f"  Top-1 returned: {r['top1_title']!r} "
                f"(bucket `{r['top1_label']}`, score={r['top1_score']:.3f})."
            )

    lines += ["", "## Opposite-label neighbors in top-5 (source-bucket leakage)", ""]
    if leakage.empty:
        lines.append("_Few/no queries with ≥2 opposite-label neighbors in top-5._")
    else:
        for _, r in leakage.iterrows():
            lines.append(
                f"- **Query:** {r['title']!r} (`{r['label_name']}`) → "
                f"{int(r['opposite_label_in_top5'])} opposite-label hits in top-5; "
                f"top-1 `{r['top1_label']}`: {r['top1_title']!r}."
            )
        lines.append(
            "\nRetrieved opposite-bucket neighbors show topical/style overlap across "
            "source classes. They do **not** prove or refute the query claim."
        )

    lines += ["", "## Same-subject / outlet-style collapse", ""]
    if style.empty:
        lines.append("_Few/no queries with ≥2 other same-subject articles in top-5._")
    else:
        for _, r in style.iterrows():
            lines.append(
                f"- **Query:** {r['title']!r} (subject `{r['subject']}`) → "
                f"{int(r['same_subject_other_in_top5'])} other same-subject articles in top-5."
            )
        lines.append(
            "\nSame-subject neighbors often share wire diction or political framing; "
            "the ranker can collapse to outlet/topic style rather than the specific claim."
        )

    lines += [
        "",
        "## Takeaway",
        "",
        "High scores here mean the index can find an article's own passages from its title. "
        "That is a pipeline sanity check, not evidence retrieval. "
        "ISOT labels remain source buckets.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def build_isot_eval_meta(
    summary: pd.DataFrame | None = None,
    ablations: pd.DataFrame | None = None,
    *,
    n_articles: int | None = None,
    max_queries: int | None = None,
    ablation_articles: int | None = None,
    ablation_queries: int | None = None,
    metrics_file: str = "results/retrieval_metrics.csv",
    ablations_file: str = "results/retrieval_ablations.csv",
) -> dict:
    """Protocol note for the committed ISOT title-recovery run.

    The numeric sentences are computed from the tables when those columns exist,
    so a regeneration does not keep a stale 'scores should drop' claim.
    """
    main = {
        "n_articles": n_articles if n_articles is not None else 4000,
        "chunk_words": 120,
        "fields": ["body"],
        "n_queries": max_queries if max_queries is not None else 300,
        "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
        "methods": ["tfidf", "dense", "hybrid"],
        "random_state": 7,
        "role": "title-recovery sanity check",
    }
    if summary is not None and not summary.empty and "n_queries" in summary.columns:
        main["n_queries"] = int(summary["n_queries"].iloc[0])

    ab_articles = 2000 if ablation_articles is None else int(ablation_articles)
    ab_queries = 150 if ablation_queries is None else int(ablation_queries)
    if ablations is not None and not ablations.empty and "n_queries" in ablations.columns:
        ab_queries = int(ablations["n_queries"].iloc[0])

    return {
        "protocol": (
            "ISOT title-recovery sanity check. The query is the article title and "
            "the gold set is every indexed chunk of that same article. This is not "
            "human relevance judgment, not claim verification, and not comparable "
            "to SciFact or BEIR."
        ),
        "interpretation": _isot_interpretation(summary, ablations),
        "main_comparison": main,
        "ablations": {
            "n_articles": ab_articles,
            "n_queries": ab_queries,
            "notes": (
                "Shared article sample across chunk-size / field / method experiments. "
                f"This file used {ab_articles} articles and {ab_queries} queries. "
                "The eval CLI defaults are 2000 articles and 150 queries, matching the committed run."
            ),
        },
        "metrics_file": metrics_file,
        "ablations_file": ablations_file,
    }


def _isot_interpretation(
    summary: pd.DataFrame | None,
    ablations: pd.DataFrame | None,
) -> dict[str, str]:
    hit_note = (
        "Hybrid's article Hit@1 edge over dense is small, while article Hit@10 moves more."
    )
    if summary is not None and not summary.empty and {"method", "article_hit@1", "article_hit@10"} <= set(summary.columns):
        by_method = summary.set_index("method")
        if {"hybrid", "dense"} <= set(by_method.index):
            hybrid_hit1 = float(by_method.loc["hybrid", "article_hit@1"])
            dense_hit1 = float(by_method.loc["dense", "article_hit@1"])
            hybrid_hit10 = float(by_method.loc["hybrid", "article_hit@10"])
            dense_hit10 = float(by_method.loc["dense", "article_hit@10"])
            edge = hybrid_hit1 - dense_hit1
            hit_note = (
                f"Hybrid's article Hit@1 edge over dense is {edge:.4f} "
                f"({hybrid_hit1:.4f} vs {dense_hit1:.4f}), while article Hit@10 "
                f"moves more ({hybrid_hit10:.4f} vs {dense_hit10:.4f})."
            )

    recall_note = (
        "Passage recall moves with chunk size because every chunk of the gold article is relevant."
    )
    if ablations is not None and not ablations.empty and "experiment" in ablations.columns:
        named = ablations.set_index("experiment")
        if {"chunk60_hybrid", "chunk240_hybrid"} <= set(named.index):
            short = float(named.loc["chunk60_hybrid", "passage_recall@5"])
            long = float(named.loc["chunk240_hybrid", "passage_recall@5"])
            short_hit = float(named.loc["chunk60_hybrid", "article_hit@1"])
            long_hit = float(named.loc["chunk240_hybrid", "article_hit@1"])
            recall_note = (
                "Passage recall moves with chunk size because every chunk of the gold "
                f"article is relevant. Hybrid passage Recall@5 goes from {short:.3f} at "
                f"60 words to {long:.3f} at 240 words, while article Hit@1 stays near "
                f"{short_hit:.3f}–{long_hit:.3f}."
            )

    return {
        "hit_at_1_note": hit_note,
        "passage_recall_note": recall_note,
        "article_mrr_note": (
            "article_mrr was dropped. With a single gold article it duplicated passage "
            "MRR on every committed row, so it was not a second result."
        ),
    }


def save_eval_bundle(
    summary: pd.DataFrame,
    detail: pd.DataFrame,
    ablations: pd.DataFrame,
    results_dir: Path | str = "results",
    index: PassageIndex | None = None,
    index_dir: Path | str = "data/retrieval_index/default",
    queries_path: Path | str | None = None,
    n_articles: int | None = None,
    max_queries: int | None = None,
    ablation_articles: int | None = None,
    ablation_queries: int | None = None,
) -> dict[str, Path]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "metrics": results_dir / "retrieval_metrics.csv",
        "detail": results_dir / "retrieval_eval_detail.csv",
        "ablations": results_dir / "retrieval_ablations.csv",
        "qualitative": results_dir / "qualitative_failures.md",
    }
    # Stable column order for README tables
    metric_cols = _METRIC_COLS
    summary = summary[[c for c in metric_cols if c in summary.columns]]
    summary.to_csv(paths["metrics"], index=False)
    detail.to_csv(paths["detail"], index=False)
    ablations.to_csv(paths["ablations"], index=False)
    write_qualitative_failures(detail, paths["qualitative"])

    if index is not None:
        index.save(index_dir)
        paths["index"] = Path(index_dir)

    if queries_path is None:
        queries_path = results_dir / "eval_queries.csv"
    # Rebuild query list from detail titles is lossy; caller may pass separately.
    paths["queries"] = Path(queries_path)

    meta = build_isot_eval_meta(
        summary,
        ablations,
        n_articles=n_articles,
        max_queries=max_queries,
        ablation_articles=ablation_articles,
        ablation_queries=ablation_queries,
        metrics_file=str(paths["metrics"]),
        ablations_file=str(paths["ablations"]),
    )
    meta_path = results_dir / "retrieval_eval_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    paths["meta"] = meta_path
    return paths
