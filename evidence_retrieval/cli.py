"""CLI: build index, query passages, run evaluation.

Examples:
  python -m evidence_retrieval build --data-dir data --out data/retrieval_index/default
  python -m evidence_retrieval query "Federal Reserve raises interest rates" --top-k 5
  python -m evidence_retrieval eval --data-dir data --results-dir results
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from evidence_retrieval.index import Hit, IndexConfig, PassageIndex


def format_query_hits(
    hits: Iterable[Hit] | pd.DataFrame,
    *,
    passage_chars: int = 280,
    query_text: str | None = None,
    method: str | None = None,
) -> str:
    """Human-readable query output for the CLI (and README fixture examples).

    Labels are ISOT source buckets, not claim-level truth.
    """
    if isinstance(hits, pd.DataFrame):
        rows = hits.to_dict(orient="records")
    else:
        rows = [
            h.__dict__ if hasattr(h, "__dict__") else dict(h)  # type: ignore[arg-type]
            for h in hits
        ]

    lines: list[str] = []
    if query_text is not None or method is not None:
        header_bits = []
        if query_text is not None:
            header_bits.append(f'query: "{query_text}"')
        if method is not None:
            header_bits.append(f"method: {method}")
        lines.append(" | ".join(header_bits))
        lines.append(
            "note: label is an ISOT source bucket (true ~ Reuters-style, "
            "fake ~ unreliable outlet) - not a fact-check verdict."
        )
        lines.append("")

    if not rows:
        lines.append("No hits.")
        return "\n".join(lines)

    for row in rows:
        rank = int(row["rank"])
        score = float(row["score"])
        label_name = row["label_name"]
        article_id = row["article_id"]
        title = row["title"]
        passage = str(row["passage"])
        if len(passage) > passage_chars:
            passage = passage[: passage_chars - 3] + "..."
        lines.append(
            f"#{rank}  score={score:.4f}  label={label_name}  article_id={article_id}"
        )
        lines.append(f"  title: {title}")
        lines.append(f"  passage: {passage}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _add_build_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("data/retrieval_index/default"))
    p.add_argument("--n-articles", type=int, default=4000)
    p.add_argument("--chunk-words", type=int, default=120)
    p.add_argument("--overlap", type=int, default=20)
    p.add_argument(
        "--fields",
        default="body",
        help="Comma-separated: body | title | title_body",
    )
    p.add_argument("--random-state", type=int, default=7)


def cmd_build(args: argparse.Namespace) -> int:
    fields = tuple(f.strip() for f in args.fields.split(",") if f.strip())
    config = IndexConfig(
        n_articles=args.n_articles,
        chunk_words=args.chunk_words,
        overlap=args.overlap,
        fields=fields,
        random_state=args.random_state,
    )
    print(
        f"Building index: n_articles={config.n_articles}, "
        f"chunk_words={config.chunk_words}, fields={config.fields}"
    )
    index = PassageIndex.build(
        data_dir=args.data_dir, config=config, show_progress=True
    )
    out = index.save(args.out)
    print(
        f"Saved {len(index.chunks):,} passages "
        f"({index.chunks['article_id'].nunique():,} articles) -> {out}"
    )
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    index = PassageIndex.load(args.index)
    hits = index.query_df(
        args.text, top_k=args.top_k, method=args.method
    )
    if args.json:
        print(hits.to_json(orient="records", force_ascii=False, indent=2))
    else:
        print(
            format_query_hits(
                hits,
                query_text=args.text,
                method=args.method,
            ),
            end="",
        )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from evidence_retrieval.eval.queries import (
        build_title_queries,
        queries_to_frame,
    )
    from evidence_retrieval.eval.run import (
        run_ablations,
        run_main_comparison,
        run_paraphrase_comparison,
        save_eval_bundle,
        save_paraphrase_bundle,
    )

    if args.paraphrase_only:
        return _cmd_paraphrase_eval(args)

    print("Running main comparison (TF-IDF vs dense vs hybrid)...")
    summary, detail, index = run_main_comparison(
        data_dir=args.data_dir,
        n_articles=args.n_articles,
        chunk_words=args.chunk_words,
        max_queries=args.max_queries,
        random_state=args.random_state,
        show_progress=not args.quiet,
    )
    print(summary.to_string(index=False))

    import pandas as pd

    if not args.skip_ablations:
        print("\nRunning ablations (chunk size / fields / method)...")
        ablations = run_ablations(
            data_dir=args.data_dir,
            n_articles=min(args.n_articles, args.ablation_articles),
            max_queries=min(args.max_queries, args.ablation_queries),
            random_state=args.random_state,
            show_progress=not args.quiet,
        )
        print(ablations.to_string(index=False))
    else:
        ablations = pd.DataFrame()

    queries = build_title_queries(
        index.chunks, max_queries=args.max_queries, random_state=args.random_state
    )
    qpath = Path(args.results_dir) / "eval_queries.csv"
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    queries_to_frame(queries).to_csv(qpath, index=False)

    paths = save_eval_bundle(
        summary=summary,
        detail=detail,
        ablations=ablations,
        results_dir=args.results_dir,
        index=index if args.save_index else None,
        index_dir=args.index_out,
        queries_path=qpath,
        n_articles=args.n_articles,
        max_queries=args.max_queries,
        ablation_articles=None if args.skip_ablations else min(args.n_articles, args.ablation_articles),
        ablation_queries=None if args.skip_ablations else min(args.max_queries, args.ablation_queries),
    )
    print("\nWrote:")
    for k, p in paths.items():
        print(f"  {k}: {p}")

    if args.with_paraphrase:
        print("\nRunning paraphrase-title comparison (harder proxy)...")
        p_summary, p_detail, p_queries, _ = run_paraphrase_comparison(
            data_dir=args.data_dir,
            n_articles=args.n_articles,
            chunk_words=args.chunk_words,
            max_queries=args.max_queries,
            random_state=args.random_state,
            show_progress=not args.quiet,
            index=index,
        )
        print(p_summary.to_string(index=False))
        p_paths = save_paraphrase_bundle(
            summary=p_summary,
            detail=p_detail,
            queries=p_queries,
            results_dir=args.results_dir,
        )
        print("\nWrote paraphrase bundle:")
        for k, p in p_paths.items():
            print(f"  {k}: {p}")

    _maybe_update_readme()
    return 0


def _cmd_paraphrase_eval(args: argparse.Namespace) -> int:
    from evidence_retrieval.eval.run import (
        run_paraphrase_comparison,
        save_paraphrase_bundle,
    )
    from evidence_retrieval.index import PassageIndex

    index = None
    index_dir = args.index if args.index is not None else args.index_out
    if index_dir is not None and Path(index_dir).exists():
        print(f"Loading existing index from {index_dir}...")
        index = PassageIndex.load(index_dir)
    else:
        print(
            "No existing index found; building body index for paraphrase eval "
            f"(n_articles={args.n_articles})..."
        )

    print("Running paraphrase-title comparison (harder proxy)...")
    summary, detail, queries, index = run_paraphrase_comparison(
        data_dir=args.data_dir,
        n_articles=args.n_articles,
        chunk_words=args.chunk_words,
        max_queries=args.max_queries,
        random_state=args.random_state,
        show_progress=not args.quiet,
        index=index,
    )
    print(summary.to_string(index=False))

    if args.save_index and index_dir is not None:
        index.save(index_dir)
        print(f"Saved index -> {index_dir}")

    paths = save_paraphrase_bundle(
        summary=summary,
        detail=detail,
        queries=queries,
        results_dir=args.results_dir,
    )
    print("\nWrote (does not modify retrieval_metrics.csv):")
    for k, p in paths.items():
        print(f"  {k}: {p}")

    _maybe_update_readme()
    return 0


def _maybe_update_readme() -> None:
    updater = Path(__file__).resolve().parents[1] / "scripts" / "update_readme_metrics.py"
    if not updater.exists():
        return
    import runpy

    try:
        runpy.run_path(str(updater), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(f"(README metrics table not updated: exit {exc.code})")
    except Exception as exc:  # noqa: BLE001
        print(f"(README metrics table not updated: {exc})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evidence_retrieval",
        description=(
            "Closed-corpus demo over the ISOT news corpus.\n\n"
            "Chunk articles -> TF-IDF or BM25 + MiniLM -> hybrid RRF ranking.\n"
            "Returns ranked passages with title, source-bucket label, and score.\n\n"
            "The evaluation claim is the SciFact table in the README,\n"
            "not this title-recovery demo.\n\n"
            "This is NOT a fact-checker. ISOT labels are source buckets\n"
            "(Reuters-style vs unreliable outlets), not claim-level truth.\n"
            'A retrieved "fake" neighbor does not prove a claim is false.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m evidence_retrieval build\n"
            '  python -m evidence_retrieval query "Federal Reserve raises rates" --top-k 5\n'
            "  python -m evidence_retrieval eval\n"
            "  python -m evidence_retrieval eval --paraphrase-only\n"
            "\n"
            "quick install:\n"
            "  pip install -e .\n"
            "  pip install -r requirements.txt   # + walkthrough notebook\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser(
        "build",
        help="Chunk + embed + save a local passage index (needs ISOT CSVs)",
    )
    _add_build_args(p_build)
    p_build.set_defaults(func=cmd_build)

    p_query = sub.add_parser(
        "query",
        help="Retrieve ranked passages for a claim / article text",
    )
    p_query.add_argument("text", help="Query / claim / article text")
    p_query.add_argument(
        "--index",
        type=Path,
        default=Path("data/retrieval_index/default"),
        help="Index directory from `build` (default: data/retrieval_index/default)",
    )
    p_query.add_argument("--top-k", type=int, default=5, help="Number of hits to return")
    p_query.add_argument(
        "--method",
        choices=["tfidf", "bm25", "dense", "hybrid"],
        default="hybrid",
        help="Ranking backend (default: hybrid RRF; bm25 needs a BM25 index)",
    )
    p_query.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON records instead of the human-readable text format",
    )
    p_query.set_defaults(func=cmd_query)

    p_eval = sub.add_parser(
        "eval",
        help="ISOT title-recovery sanity check -> results/*.csv (not claim verification)",
    )
    p_eval.add_argument("--data-dir", type=Path, default=Path("data"))
    p_eval.add_argument("--results-dir", type=Path, default=Path("results"))
    p_eval.add_argument("--n-articles", type=int, default=4000)
    p_eval.add_argument("--chunk-words", type=int, default=120)
    p_eval.add_argument("--max-queries", type=int, default=300)
    p_eval.add_argument("--ablation-articles", type=int, default=2000)
    p_eval.add_argument("--ablation-queries", type=int, default=150)
    p_eval.add_argument("--random-state", type=int, default=7)
    p_eval.add_argument("--skip-ablations", action="store_true")
    p_eval.add_argument("--save-index", action="store_true", default=True)
    p_eval.add_argument("--no-save-index", action="store_false", dest="save_index")
    p_eval.add_argument(
        "--index-out",
        type=Path,
        default=Path("data/retrieval_index/default"),
    )
    p_eval.add_argument("--quiet", action="store_true")
    p_eval.add_argument(
        "--paraphrase-only",
        action="store_true",
        help=(
            "Run paraphrase-title eval only -> results/paraphrase_*.csv "
            "(does not rewrite retrieval_metrics.csv)"
        ),
    )
    p_eval.add_argument(
        "--with-paraphrase",
        action="store_true",
        help="After the main title eval, also run paraphrase-title comparison",
    )
    p_eval.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Existing index dir for --paraphrase-only (default: --index-out if present)",
    )
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
