#!/usr/bin/env python3
"""Insert SciFact metric CSVs into README table markers.

ISOT title-recovery markers are refreshed only when the README still has them.
The root README keeps that demo out of the scored table.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
METRICS = ROOT / "results" / "retrieval_metrics.csv"
PARA_METRICS = ROOT / "results" / "paraphrase_metrics.csv"
START = "<!-- METRICS_TABLE_START -->"
END = "<!-- METRICS_TABLE_END -->"
PARA_START = "<!-- PARAPHRASE_METRICS_TABLE_START -->"
PARA_END = "<!-- PARAPHRASE_METRICS_TABLE_END -->"
SCIFACT = ROOT / "results" / "scifact_test_metrics.csv"
SCIFACT_ABLATION = ROOT / "results" / "scifact_test_ablations.csv"
SCIFACT_START = "<!-- SCIFACT_TEST_TABLE_START -->"
SCIFACT_END = "<!-- SCIFACT_TEST_TABLE_END -->"
SCIFACT_ABLATION_START = "<!-- SCIFACT_ABLATION_TABLE_START -->"
SCIFACT_ABLATION_END = "<!-- SCIFACT_ABLATION_TABLE_END -->"


def fmt(x: float) -> str:
    return f"{x:.4f}"


def build_scifact_table(df: pd.DataFrame) -> str:
    cols = [
        ("method", "Method", None),
        ("ndcg@10", "nDCG@10", fmt),
        ("recall@100", "Recall@100", fmt),
        ("recip_rank", "Reciprocal rank", fmt),
    ]
    header = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "| " + " | ".join("---" if c[2] is None else "---:" for c in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for key, _, formatter in cols:
            val = row[key]
            cells.append(str(val) if formatter is None else formatter(float(val)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_table(df: pd.DataFrame) -> str:
    cols = [
        ("method", "Method", None),
        ("article_hit@1", "Article Hit@1", fmt),
        ("article_hit@5", "Article Hit@5", fmt),
        ("article_hit@10", "Article Hit@10", fmt),
        ("passage_recall@5", "Passage Recall@5", fmt),
        ("ndcg@5", "nDCG@5", fmt),
        ("ndcg@10", "nDCG@10", fmt),
        ("mrr", "MRR", fmt),
    ]
    header = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "| " + " | ".join("---" if c[2] is None else "---:" for c in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for key, _, formatter in cols:
            val = row[key]
            cells.append(str(val) if formatter is None else formatter(float(val)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _replace_block(text: str, start: str, end: str, body: str) -> str:
    if start not in text or end not in text:
        raise SystemExit(f"README missing markers {start} / {end}")
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    return before + start + "\n" + body + "\n" + end + after


def main() -> int:
    text = README.read_text(encoding="utf-8")

    if METRICS.exists() and START in text and END in text:
        table = build_table(pd.read_csv(METRICS))
        text = _replace_block(text, START, END, table)
        print(f"Updated README ISOT demo table from {METRICS}")
        print(table)
    else:
        print("README has no ISOT metrics markers; leaving the demo table in results/")

    if PARA_START in text and PARA_END in text:
        if PARA_METRICS.exists():
            para_table = build_table(pd.read_csv(PARA_METRICS))
            text = _replace_block(text, PARA_START, PARA_END, para_table)
            print(f"Updated README paraphrase metrics table from {PARA_METRICS}")
            print(para_table)
        else:
            placeholder = (
                "_Not measured in this checkout — regenerate with:_ "
                "`python scripts/run_paraphrase_eval.py` "
                "(needs ISOT CSVs + MiniLM; does not rewrite "
                "`results/retrieval_metrics.csv`)."
            )
            text = _replace_block(text, PARA_START, PARA_END, placeholder)
            print("Paraphrase metrics CSV missing; left honest placeholder in README")

    if SCIFACT_START in text and SCIFACT_END in text and SCIFACT.exists():
        text = _replace_block(text, SCIFACT_START, SCIFACT_END, build_scifact_table(pd.read_csv(SCIFACT)))
        print(f"Updated README SciFact test table from {SCIFACT}")
    if SCIFACT_ABLATION_START in text and SCIFACT_ABLATION_END in text and SCIFACT_ABLATION.exists():
        text = _replace_block(
            text,
            SCIFACT_ABLATION_START,
            SCIFACT_ABLATION_END,
            build_scifact_table(pd.read_csv(SCIFACT_ABLATION)),
        )
        print(f"Updated README SciFact ablation table from {SCIFACT_ABLATION}")

    text = text.replace(
        "**Placeholder until the eval run commits real numbers** — if you are reading a checkout\n"
        "before that run finishes, regenerate locally.\n\n",
        "",
    )
    README.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
