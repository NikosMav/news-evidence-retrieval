"""Labeled query construction for retrieval evaluation."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import pandas as pd


@dataclass
class EvalQuery:
    query_id: str
    article_id: int
    query_text: str
    gold_chunk_ids: list[str]
    gold_article_ids: list[int]
    label_name: str
    subject: str
    title: str


# ---------------------------------------------------------------------------
# Deterministic title paraphrases (no LLM / no paid API)
# ---------------------------------------------------------------------------

_LEADING_TAG_RE = re.compile(
    r"^(?:"
    r"BREAKING(?:\s+NEWS)?|OOPS|WATCH|VIDEO|LOOK|MUST\s+SEE|EXCLUSIVE|"
    r"UPDATE|DEVELOPING"
    r")[!,:.\s\-–—]*",
    re.IGNORECASE,
)
_TRAILING_TAG_RE = re.compile(
    r"[\s\-–—]*[\(\[]\s*(?:VIDEO|VIDEOS|TWEET|IMAGE|IMAGES|DETAILS?)\s*[\)\]]\s*$",
    re.IGNORECASE,
)
_OUTLET_SUFFIX_RE = re.compile(
    r"\s*:\s*(?:"
    r"Reuters|AP|AFP|Treasury|minister|officials?|report|sources?|"
    r"White House|Poll|Opinion|Analysis|Live|Update"
    r")\s*$",
    re.IGNORECASE,
)
_DATELINE_RE = re.compile(
    r"^[A-Z][A-Za-z./\s]{2,40}\s*\((?:Reuters|AP|AFP)\)\s*[-–—:]\s*",
)
_PUNCT_RE = re.compile(r"[\"“”‘’…]+")
_SPACE_RE = re.compile(r"\s+")

_SYNONYMS: dict[str, str] = {
    "says": "states",
    "said": "stated",
    "raises": "increases",
    "raised": "increased",
    "hits": "strikes",
    "slams": "criticizes",
    "blasts": "criticizes",
    "rips": "criticizes",
    "urges": "calls on",
    "warns": "cautions",
    "vows": "promises",
    " amid ": " during ",
    " after ": " following ",
    " before ": " ahead of ",
    " over ": " regarding ",
    " against ": " versus ",
    "needs": "requires",
    "faces": "confronts",
    "sees": "observes",
    "calls": "urges",
    "talks": "discussions",
    "deal": "agreement",
    "poll": "survey",
    "probe": "investigation",
    "row": "dispute",
    "bid": "attempt",
}

_FUNCTION_WORDS = frozenset(
    {"a", "an", "the", "of", "to", "in", "on", "for", "and", "or"}
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")

_FALLBACK_TEMPLATES = (
    "What happened with {core}",
    "Reports concerning {core}",
    "Coverage of {core}",
)


def token_set(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(str(text).lower()))


def token_jaccard(left: str, right: str) -> float:
    """Token Jaccard used to show the title rewrite barely changes the query."""
    a = token_set(left)
    b = token_set(right)
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def content_word_retention(original: str, rewrite: str) -> float:
    """Share of non-function tokens from ``original`` that survive in ``rewrite``."""
    content = {tok for tok in token_set(original) if tok not in _FUNCTION_WORDS}
    if not content:
        return 1.0
    return len(content & token_set(rewrite)) / len(content)


def paraphrase_title(title: str) -> str:
    """Deterministic, rule-based paraphrase of a news title.

    Drops dateline/outlet/clickbait tags, light punctuation cleanup, a closed
    synonym map, mild function-word stripping, and a hash-selected template
    fallback when the cleaned string would otherwise match the original.
    No LLM / paid API. Reproducible across processes (no builtin ``hash()``).
    """
    original = (title or "").strip()
    if not original:
        return "What happened with this story"

    text = original
    text = _DATELINE_RE.sub("", text)
    text = _LEADING_TAG_RE.sub("", text)
    text = _TRAILING_TAG_RE.sub("", text)
    text = _OUTLET_SUFFIX_RE.sub("", text)
    text = _PUNCT_RE.sub(" ", text)
    text = text.replace("!", " ").replace("?", " ")
    text = _SPACE_RE.sub(" ", text).strip(" -–—:,.")

    lowered = f" {text.lower()} "
    for src, dst in sorted(_SYNONYMS.items(), key=lambda kv: -len(kv[0])):
        if src.startswith(" ") and src.endswith(" "):
            lowered = lowered.replace(src, dst)
    words = lowered.strip().split()
    replaced: list[str] = []
    for w in words:
        key = w.lower()
        if key in _SYNONYMS and not key.startswith(" "):
            replaced.append(_SYNONYMS[key])
        else:
            replaced.append(w)
    content = [w for w in replaced if w.lower() not in _FUNCTION_WORDS]
    if len(content) >= 4:
        replaced = content
    core = _SPACE_RE.sub(" ", " ".join(replaced)).strip(" -–—:,.")

    if not core:
        core = _SPACE_RE.sub(" ", original).strip()

    if core.casefold() == original.casefold() or _nearly_same(core, original):
        tmpl = _FALLBACK_TEMPLATES[_stable_index(original, len(_FALLBACK_TEMPLATES))]
        core = tmpl.format(core=core)

    if core.casefold() == original.casefold():
        core = f"What happened with {original}"

    return core


def _nearly_same(a: str, b: str) -> bool:
    def norm(s: str) -> str:
        return _SPACE_RE.sub(" ", _PUNCT_RE.sub(" ", s).lower()).strip(" -–—:,.!")

    return norm(a) == norm(b)


def _stable_index(text: str, modulus: int) -> int:
    digest = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).digest()
    return int.from_bytes(digest[:4], "little") % modulus


def build_title_queries(
    chunks: pd.DataFrame,
    max_queries: int = 300,
    min_title_words: int = 4,
    random_state: int = 7,
) -> list[EvalQuery]:
    """Title-as-claim → gold = same article's indexed body passages.

    Justified synthetic protocol when human qrels are unavailable. See package docs.
    """
    body_chunks = chunks[chunks["field"].isin(["body", "title_body"])]
    if body_chunks.empty:
        body_chunks = chunks

    article_ids = body_chunks.groupby("article_id").size().reset_index(name="n_chunks")
    meta = chunks.drop_duplicates("article_id")[
        ["article_id", "title", "label_name", "subject"]
    ]
    candidates = article_ids.merge(meta, on="article_id")
    candidates["title_words"] = candidates["title"].str.split().str.len().fillna(0)
    candidates = candidates[candidates["title_words"] >= min_title_words]
    candidates = candidates[candidates["title"].str.strip().astype(bool)]

    if candidates.empty:
        raise RuntimeError("No eligible query articles — check chunk fields/titles.")

    n = min(max_queries, len(candidates))
    sampled = candidates.sample(n=n, random_state=random_state).reset_index(drop=True)
    gold_map = body_chunks.groupby("article_id")["chunk_id"].apply(list).to_dict()

    queries: list[EvalQuery] = []
    for _, row in sampled.iterrows():
        aid = int(row["article_id"])
        gold = [str(c) for c in gold_map.get(aid, [])]
        if not gold:
            continue
        queries.append(
            EvalQuery(
                query_id=f"title-{aid}",
                article_id=aid,
                query_text=str(row["title"]).strip(),
                gold_chunk_ids=gold,
                gold_article_ids=[aid],
                label_name=str(row["label_name"]),
                subject=str(row["subject"]),
                title=str(row["title"]),
            )
        )
    return queries


def build_paraphrase_queries(
    chunks: pd.DataFrame,
    max_queries: int = 300,
    min_title_words: int = 4,
    random_state: int = 7,
) -> list[EvalQuery]:
    """Rule-based title rewrite; gold is still that article's indexed passages.

    The rewrite keeps most content words (see ``token_jaccard``). It is not a
    lexical stress test. Gold ids are unchanged (same ``article_id``).
    """
    base = build_title_queries(
        chunks,
        max_queries=max_queries,
        min_title_words=min_title_words,
        random_state=random_state,
    )
    out: list[EvalQuery] = []
    for q in base:
        para = paraphrase_title(q.title)
        out.append(
            EvalQuery(
                query_id=f"para-{q.article_id}",
                article_id=q.article_id,
                query_text=para,
                gold_chunk_ids=list(q.gold_chunk_ids),
                gold_article_ids=list(q.gold_article_ids),
                label_name=q.label_name,
                subject=q.subject,
                title=q.title,
            )
        )
    return out


def queries_to_frame(queries: list[EvalQuery]) -> pd.DataFrame:
    rows = []
    for q in queries:
        rows.append(
            {
                "query_id": q.query_id,
                "article_id": q.article_id,
                "query_text": q.query_text,
                "gold_chunk_ids": "|".join(q.gold_chunk_ids),
                "gold_article_ids": "|".join(str(x) for x in q.gold_article_ids),
                "label_name": q.label_name,
                "subject": q.subject,
                "title": q.title,
                "n_gold_chunks": len(q.gold_chunk_ids),
            }
        )
    return pd.DataFrame(rows)


def paraphrase_queries_to_frame(queries: list[EvalQuery]) -> pd.DataFrame:
    """Frame with explicit original_title vs paraphrase columns for reviewers."""
    base = queries_to_frame(queries)
    if base.empty:
        return base
    out = base.rename(columns={"title": "original_title", "query_text": "paraphrase"})
    cols = [
        "query_id",
        "article_id",
        "original_title",
        "paraphrase",
        "label_name",
        "subject",
        "gold_chunk_ids",
        "gold_article_ids",
        "n_gold_chunks",
    ]
    return out[[c for c in cols if c in out.columns]]
