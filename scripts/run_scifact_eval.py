#!/usr/bin/env python3
"""SciFact document retrieval: BM25 vs MiniLM vs RRF.

Dev on the train split. Touch the test split only after the metric code is frozen.

  pip install -e ".[scifact]"
  python scripts/run_scifact_eval.py --split train
  python scripts/run_scifact_eval.py --split test

Writes results/scifact_<split>_metrics.csv, a per-query file, and
results/scifact_eval_meta.json. Does not rewrite the ISOT title-recovery tables.

Metrics are pytrec_eval nDCG@10 and Recall@100. BM25 is bm25s (Lucene scoring,
k1=0.9, b=0.4, English stopwords, Snowball stemmer). Published anchors, not
measured here: BEIR BM25 nDCG@10 0.665; Anserini flat BM25 nDCG@10 0.6789.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence_retrieval.encoders import DEFAULT_DENSE_MODEL, DenseEncoder
from evidence_retrieval.eval.scifact import (
    ANSERINI_FLAT_BM25_NDCG_AT_10,
    BEIR_BM25_NDCG_AT_10,
    SCIFACT_DATASET_IDS,
    load_scifact_split,
    run_document_comparison,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--split",
        choices=sorted(SCIFACT_DATASET_IDS),
        default="train",
        help="beir/scifact split. Default is train. Pass test only for the reported table.",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.split == "test":
        print(
            "Scoring beir/scifact/test. Metric code should already be frozen; "
            "do not change tokenization or metrics from this run.",
            file=sys.stderr,
        )

    print(f"Loading {SCIFACT_DATASET_IDS[args.split]} via ir_datasets...")
    documents, queries, qrels = load_scifact_split(args.split)
    print(
        f"  documents={len(documents):,}  queries={len(queries):,}  "
        f"judged={sum(1 for rels in qrels.values() if rels):,}"
    )
    print(f"Encoding with {args.dense_model} and bm25s (k1=0.9, b=0.4)...")
    encoder = DenseEncoder(model_name=args.dense_model)
    result = run_document_comparison(
        documents,
        queries,
        qrels,
        dense_encoder=encoder,
        k=args.k,
        dense_model=args.dense_model,
        show_progress=not args.quiet,
    )
    summary = result["summary"]
    print(summary.to_string(index=False))
    if result["sign_tests"]:
        print("Sign test on nDCG@10 versus BM25 (ties dropped):")
        for method, test in result["sign_tests"].items():
            print(
                f"  {method}: +{test['n_pos']} / -{test['n_neg']} "
                f"(ties {test['n_ties']}), p={test['pvalue']:.4g}"
            )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.results_dir / f"scifact_{args.split}_metrics.csv"
    per_query_path = args.results_dir / f"scifact_{args.split}_per_query.csv"
    summary.to_csv(metrics_path, index=False)
    result["per_query"].to_csv(per_query_path, index=False)
    meta_path = _update_meta(args.results_dir / "scifact_eval_meta.json", args, result)
    print(f"Wrote {metrics_path}")
    print(f"Wrote {per_query_path}")
    print(f"Wrote {meta_path}")
    return 0


def _update_meta(path: Path, args, result: dict) -> Path:
    if path.exists():
        meta = json.loads(path.read_text(encoding="utf-8"))
    else:
        meta = {
            "task": "SciFact document retrieval (BEIR qrels). Not the ISOT title-recovery check.",
            "loader": "ir_datasets",
            "metrics": ["ndcg_cut_10", "recall_100", "recip_rank"],
            "metrics_library": "pytrec_eval",
            "bm25": {
                "library": "bm25s",
                "method": "lucene",
                "k1": result["bm25_k1"],
                "b": result["bm25_b"],
                "stopwords": "english",
                "stemmer": "PyStemmer english (Snowball)",
            },
            "hybrid": "RRF k=60 over BM25 and dense ranks; raw scores ignored",
            "chunking": "Whole abstracts. The word-window chunker is not used.",
            "anchors": {
                "beir_bm25_ndcg@10": BEIR_BM25_NDCG_AT_10,
                "anserini_flat_bm25_ndcg@10": ANSERINI_FLAT_BM25_NDCG_AT_10,
                "note": (
                    "Anchors are published numbers. k1=0.9, b=0.4 matches Anserini's "
                    "BEIR flat BM25 setting more closely than the Elasticsearch defaults "
                    "behind BEIR's 0.665."
                ),
            },
            "command_train": "python scripts/run_scifact_eval.py --split train",
            "command_test": "python scripts/run_scifact_eval.py --split test",
            "discipline": (
                "Develop on beir/scifact/train. Score beir/scifact/test only after "
                "metric code is frozen."
            ),
            "splits": {},
        }
    meta.setdefault("splits", {})
    meta["splits"][args.split] = {
        "dataset": SCIFACT_DATASET_IDS[args.split],
        "n_documents": result["n_documents"],
        "n_judged_queries": result["n_judged_queries"],
        "k": result["k"],
        "dense_model": result["dense_model"],
        "metrics_file": f"results/scifact_{args.split}_metrics.csv",
        "per_query_file": f"results/scifact_{args.split}_per_query.csv",
        "sign_test_ndcg@10_vs_bm25": result["sign_tests"],
    }
    if "test" in meta["splits"]:
        meta["headline_split"] = "test"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
