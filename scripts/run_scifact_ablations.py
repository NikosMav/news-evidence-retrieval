#!/usr/bin/env python3
"""SciFact test ablations on top of the frozen BM25 / MiniLM / RRF table.

Run only after ``scripts/run_scifact_eval.py --split test`` has been committed.
The first stage for reranking is hybrid, chosen from the train split
(nDCG@10 hybrid 0.719 > BM25 0.694 > MiniLM 0.660), not from the test numbers.

  python scripts/run_scifact_ablations.py --split test

Writes results/scifact_test_ablations.csv and a short failure note.
Does not overwrite results/scifact_test_metrics.csv. Refuses to continue if a
recomputed three-way table disagrees with that file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from evidence_retrieval.encoders import DEFAULT_DENSE_MODEL, DenseEncoder
from evidence_retrieval.eval.scifact import (
    apply_rerank,
    load_scifact_split,
    order_from_run,
    run_document_comparison,
)
from evidence_retrieval.eval.trec import ranking_to_run, summarize_run, paired_sign_test

RERANK_DEPTH = 50
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BGE_MODEL = "BAAI/bge-small-en-v1.5"
# Official BGE v1.5 retrieval instruction. Applied to queries only.
BGE_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "
FIRST_STAGE = "hybrid"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    frozen_path = args.results_dir / "scifact_test_metrics.csv"
    if not frozen_path.exists():
        print(f"Missing {frozen_path}. Run the three-way test eval before ablations.", file=sys.stderr)
        return 1

    print("Loading beir/scifact/test...")
    documents, queries, qrels = load_scifact_split("test")
    text_by_id = {str(row.doc_id): str(row.text) for row in documents.itertuples(index=False)}
    title_by_id = {
        str(row.doc_id): str(getattr(row, "title", "") or "")
        for row in documents.itertuples(index=False)
    }
    query_text = {str(row.query_id): str(row.text) for row in queries.itertuples(index=False)}

    print(f"Recomputing the frozen three-way table with {DEFAULT_DENSE_MODEL}...")
    minilm = run_document_comparison(
        documents,
        queries,
        qrels,
        dense_encoder=DenseEncoder(model_name=DEFAULT_DENSE_MODEL),
        dense_model=DEFAULT_DENSE_MODEL,
        show_progress=not args.quiet,
    )
    _require_same_table(pd.read_csv(frozen_path), minilm["summary"])

    print(f"Reranking the top {RERANK_DEPTH} of {FIRST_STAGE} with {RERANK_MODEL}...")
    rerank_run = _rerank(
        queries,
        qrels,
        minilm["runs"][FIRST_STAGE],
        text_by_id,
        show_progress=not args.quiet,
    )
    rerank_row, rerank_scores = summarize_run("hybrid_rerank", qrels, rerank_run)

    print(f"Swapping the dense encoder for {BGE_MODEL}...")
    bge = run_document_comparison(
        documents,
        queries,
        qrels,
        dense_encoder=DenseEncoder(model_name=BGE_MODEL),
        dense_model=BGE_MODEL,
        methods=("dense", "hybrid"),
        query_prompt=BGE_QUERY_PROMPT,
        show_progress=not args.quiet,
    )

    rows = [rerank_row]
    for _, row in bge["summary"].iterrows():
        name = "bge" if row["method"] == "dense" else "bge_hybrid"
        rows.append({**row.to_dict(), "method": name})
    ablations = pd.DataFrame(rows)
    print(ablations.to_string(index=False))
    out = args.results_dir / "scifact_test_ablations.csv"
    ablations.to_csv(out, index=False)
    print(f"Wrote {out}")

    bm25_rows = minilm["per_query"]
    bm25_rows = bm25_rows[bm25_rows["method"] == "bm25"]
    bm25_ndcg = {
        str(query_id): float(score)
        for query_id, score in zip(bm25_rows["query_id"], bm25_rows["ndcg@10"])
    }
    sign_tests = {
        "hybrid_rerank": _sign(rerank_scores, bm25_ndcg),
        "bge": _sign_from_frame(bge["per_query"], "dense", bm25_ndcg),
        "bge_hybrid": _sign_from_frame(bge["per_query"], "hybrid", bm25_ndcg),
    }
    for name, test in sign_tests.items():
        print(
            f"  {name} vs bm25 nDCG@10: +{test['n_pos']} / -{test['n_neg']} "
            f"(ties {test['n_ties']}), p={test['pvalue']:.4g}"
        )

    failures = args.results_dir / "scifact_failures.md"
    _write_failures(
        failures,
        query_text,
        qrels,
        text_by_id,
        title_by_id,
        minilm["runs"],
        minilm["per_query"],
    )
    _update_meta(args.results_dir / "scifact_eval_meta.json", ablations, sign_tests, out, failures)
    print(f"Wrote {out}")
    print(f"Wrote {failures}")
    return 0


def _rerank(queries, qrels, first_stage, text_by_id, show_progress: bool):
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(RERANK_MODEL)
    judged = {qid for qid, rels in qrels.items() if rels}
    run = {}
    rows = [row for row in queries.itertuples(index=False) if str(row.query_id) in judged]
    for index, row in enumerate(rows, start=1):
        query_id = str(row.query_id)
        ordered = order_from_run(first_stage.get(query_id, {}))
        width = min(RERANK_DEPTH, len(ordered))
        if width == 0:
            run[query_id] = {}
            continue
        pairs = [(str(row.text), text_by_id.get(doc_id, "")) for doc_id in ordered[:width]]
        scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
        reranked = apply_rerank(ordered, scores, depth=RERANK_DEPTH)
        run[query_id] = ranking_to_run(reranked)
        if show_progress and index % 50 == 0:
            print(f"  reranked {index}/{len(rows)} queries")
    for query_id in judged:
        run.setdefault(query_id, {})
    return run


def _require_same_table(frozen: pd.DataFrame, fresh: pd.DataFrame) -> None:
    fresh_by_method = fresh.set_index("method")
    for _, row in frozen.iterrows():
        method = row["method"]
        for column in ("ndcg@10", "recall@100", "recip_rank"):
            old = float(row[column])
            new = float(fresh_by_method.loc[method, column])
            if abs(old - new) > 1e-8:
                raise SystemExit(
                    f"Recomputed {method} {column}={new} disagrees with frozen {old}. "
                    "Not writing ablations."
                )
    print("Recomputed three-way table matches results/scifact_test_metrics.csv.")


def _sign(per_query: dict, bm25_ndcg: dict) -> dict:
    left = {qid: row["ndcg_cut_10"] for qid, row in per_query.items()}
    result = paired_sign_test(left, bm25_ndcg)
    result["metric"] = "ndcg@10"
    result["compared_to"] = "bm25"
    return result


def _sign_from_frame(frame: pd.DataFrame, method: str, bm25_ndcg: dict) -> dict:
    subset = frame[frame["method"] == method]
    left = {
        str(query_id): float(score)
        for query_id, score in zip(subset["query_id"], subset["ndcg@10"])
    }
    result = paired_sign_test(left, bm25_ndcg)
    result["metric"] = "ndcg@10"
    result["compared_to"] = "bm25"
    return result


def _write_failures(path, query_text, qrels, text_by_id, title_by_id, runs, per_query) -> None:
    by_method = {
        method: frame.set_index("query_id")
        for method, frame in per_query.groupby("method")
    }

    def example(kind: str, query_id: str) -> str:
        gold = ", ".join(sorted(qrels.get(query_id, {})))
        lines = [f"### {kind}", "", f"Claim `{query_id}`: {query_text.get(query_id, '')}", "", f"Gold doc ids: {gold}", ""]
        for method in ("bm25", "dense", "hybrid"):
            ordered = order_from_run(runs[method].get(query_id, {}))
            top = ordered[0] if ordered else None
            title = title_by_id.get(top, "") if top else ""
            snippet = text_by_id.get(top, "").replace("\n", " ")[:320] if top else ""
            ndcg = float(by_method[method].loc[query_id, "ndcg@10"])
            lines.append(
                f"- {method} nDCG@10={ndcg:.3f}; top doc `{top}` {title}: {snippet}"
            )
        lines.append("")
        return "\n".join(lines)

    lexical = _first_where(
        by_method,
        lambda qid: float(by_method["bm25"].loc[qid, "ndcg@10"]) == 0.0
        and float(by_method["dense"].loc[qid, "ndcg@10"]) > 0.5,
    )
    dense_miss = _first_where(
        by_method,
        lambda qid: float(by_method["dense"].loc[qid, "ndcg@10"]) == 0.0
        and float(by_method["bm25"].loc[qid, "ndcg@10"]) > 0.9,
    )
    both_miss = _first_where(
        by_method,
        lambda qid: float(by_method["bm25"].loc[qid, "ndcg@10"]) == 0.0
        and float(by_method["dense"].loc[qid, "ndcg@10"]) == 0.0
        and float(by_method["hybrid"].loc[qid, "ndcg@10"]) == 0.0,
    )
    parts = [
        "# SciFact misses (beir/scifact/test)",
        "",
        "Drawn from the frozen BM25 / MiniLM / RRF run. A miss is nDCG@10 of 0: no relevant abstract in the top 10.",
        "",
    ]
    if lexical:
        parts.append(example("BM25 misses a claim that MiniLM ranks", lexical))
    if dense_miss:
        parts.append(example("MiniLM misses a claim that BM25 ranks first", dense_miss))
    if both_miss:
        parts.append(example("BM25, MiniLM, and hybrid all miss the top 10", both_miss))
    if not (lexical or dense_miss or both_miss):
        parts.append("No query matched the miss filters used by this sampler.")
        parts.append("")
    path.write_text("\n".join(parts), encoding="utf-8")


def _first_where(by_method, predicate) -> str | None:
    for query_id in by_method["bm25"].index.astype(str):
        if predicate(query_id):
            return query_id
    return None


def _update_meta(path: Path, ablations: pd.DataFrame, sign_tests: dict, table: Path, failures: Path) -> None:
    meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    meta["ablations"] = {
        "split": "test",
        "first_stage": FIRST_STAGE,
        "first_stage_choice": (
            "Hybrid was the better first stage on beir/scifact/train "
            "(nDCG@10 0.719 > BM25 0.694 > MiniLM 0.660). The test split was not used to choose it."
        ),
        "reranker": RERANK_MODEL,
        "rerank_depth": RERANK_DEPTH,
        "embedding": BGE_MODEL,
        "embedding_query_prompt": BGE_QUERY_PROMPT,
        "sentence_transformers_pin": "Kept sentence-transformers>=2.6,<4. Both models use the existing CrossEncoder and SentenceTransformer.encode APIs.",
        "metrics_file": str(table).replace("\\", "/"),
        "failures_file": str(failures).replace("\\", "/"),
        "command": "python scripts/run_scifact_ablations.py --split test",
        "rows": ablations.to_dict(orient="records"),
        "sign_test_ndcg@10_vs_bm25": sign_tests,
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
