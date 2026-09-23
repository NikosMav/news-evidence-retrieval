#!/usr/bin/env python3
"""Reproducible retrieval eval — regenerates results/*.csv from a clean checkout.

Usage:
  python scripts/download_data.py
  python scripts/run_retrieval_eval.py

Writes:
  results/retrieval_metrics.csv
  results/retrieval_ablations.csv
  results/retrieval_eval_detail.csv
  results/qualitative_failures.md
  results/eval_queries.csv
  data/retrieval_index/default/   (unless --no-save-index)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without editable install when repo root is CWD.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence_retrieval.cli import main as cli_main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--n-articles", type=int, default=4000)
    parser.add_argument("--max-queries", type=int, default=300)
    parser.add_argument("--ablation-articles", type=int, default=2000)
    parser.add_argument("--ablation-queries", type=int, default=150)
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-save-index", action="store_true")
    args, unknown = parser.parse_known_args()

    argv = [
        "eval",
        "--data-dir",
        str(args.data_dir),
        "--results-dir",
        str(args.results_dir),
        "--n-articles",
        str(args.n_articles),
        "--max-queries",
        str(args.max_queries),
        "--ablation-articles",
        str(args.ablation_articles),
        "--ablation-queries",
        str(args.ablation_queries),
    ]
    if args.skip_ablations:
        argv.append("--skip-ablations")
    if args.quiet:
        argv.append("--quiet")
    if args.no_save_index:
        argv.append("--no-save-index")
    argv.extend(unknown)
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
