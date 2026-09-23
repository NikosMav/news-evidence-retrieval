"""Build, save, load, and query a multi-backend passage index."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from evidence_retrieval.chunking import articles_to_chunks
from evidence_retrieval.data import load_isot, stratified_sample
from evidence_retrieval.encoders import (
    DEFAULT_BM25_B,
    DEFAULT_BM25_K1,
    DEFAULT_DENSE_MODEL,
    BM25Encoder,
    DenseEncoder,
    SparseEncoder,
)

Method = Literal["tfidf", "bm25", "dense", "hybrid"]
SparseBackend = Literal["tfidf", "bm25"]


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[tuple[int, float]]],
    k_rrf: int = 60,
    fetch: int | None = None,
) -> list[tuple[int, float]]:
    """Fuse multiple ranked (index, score) lists with Reciprocal Rank Fusion.

    Only ranks matter; input scores are ignored. Matches the hybrid path used by
    ``PassageIndex._rank_hybrid``.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (idx, _) in enumerate(ranked, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k_rrf + rank)
    ordered = sorted(scores.items(), key=lambda x: -x[1])
    if fetch is not None:
        ordered = ordered[:fetch]
    return [(int(i), float(s)) for i, s in ordered]


def _coerce_passages(passages: pd.DataFrame) -> pd.DataFrame:
    """Fill the columns ``PassageIndex.query`` reads from a document table."""
    if "passage" not in passages.columns or "chunk_id" not in passages.columns:
        raise ValueError("passages must include chunk_id and passage columns")
    frame = passages.copy().reset_index(drop=True)
    if "article_id" not in frame.columns:
        article_ids = []
        for i, chunk_id in enumerate(frame["chunk_id"].astype(str)):
            article_ids.append(int(chunk_id) if chunk_id.isdigit() else i)
        frame["article_id"] = article_ids
    if "title" not in frame.columns:
        frame["title"] = frame["chunk_id"].astype(str)
    for column, default in (
        ("label", 0),
        ("label_name", "document"),
        ("subject", ""),
        ("date", ""),
    ):
        if column not in frame.columns:
            frame[column] = default
    frame["chunk_id"] = frame["chunk_id"].astype(str)
    frame["passage"] = frame["passage"].astype(str)
    frame["article_id"] = frame["article_id"].astype(int)
    return frame


@dataclass
class IndexConfig:
    n_articles: int = 4000
    chunk_words: int = 120
    overlap: int = 20
    fields: tuple[str, ...] = ("body",)
    dense_model: str = DEFAULT_DENSE_MODEL
    random_state: int = 7
    embed_batch_size: int = 64
    sparse_backend: SparseBackend = "tfidf"
    bm25_k1: float = DEFAULT_BM25_K1
    bm25_b: float = DEFAULT_BM25_B

    def __post_init__(self) -> None:
        if self.n_articles < 1:
            raise ValueError("n_articles must be >= 1")
        if self.chunk_words < 1:
            raise ValueError("chunk_words must be >= 1")
        if self.overlap < 0 or self.overlap >= self.chunk_words:
            raise ValueError("overlap must be >= 0 and smaller than chunk_words")
        if self.embed_batch_size < 1:
            raise ValueError("embed_batch_size must be >= 1")
        if not self.fields:
            raise ValueError("fields must contain body, title, or title_body")
        allowed = {"body", "title", "title_body"}
        unknown = set(self.fields) - allowed
        if unknown:
            raise ValueError(f"Unknown index fields: {sorted(unknown)}")
        if self.sparse_backend not in ("tfidf", "bm25"):
            raise ValueError("sparse_backend must be 'tfidf' or 'bm25'")
        if self.bm25_k1 <= 0:
            raise ValueError("bm25_k1 must be > 0")
        if not 0 <= self.bm25_b <= 1:
            raise ValueError("bm25_b must be between 0 and 1")


@dataclass
class Hit:
    rank: int
    score: float
    chunk_id: str
    article_id: int
    title: str
    label: int
    label_name: str
    subject: str
    date: str
    passage: str
    method: str


class PassageIndex:
    """In-memory sparse + dense index over chunked ISOT passages."""

    def __init__(
        self,
        chunks: pd.DataFrame,
        sparse: SparseEncoder | None,
        sparse_matrix,
        dense_embeddings: np.ndarray | None,
        dense_encoder: DenseEncoder | None,
        config: IndexConfig,
        bm25: BM25Encoder | None = None,
    ):
        self.chunks = chunks.reset_index(drop=True)
        self.sparse = sparse
        self.sparse_matrix = sparse_matrix
        self.dense_embeddings = dense_embeddings
        self.dense_encoder = dense_encoder
        self.bm25 = bm25
        self.config = config

        self._dense_nn: NearestNeighbors | None = None
        if dense_embeddings is not None:
            # Brute-force cosine. Cap is only the default; query() asks for the
            # depth it needs (100 is enough for SciFact Recall@100).
            self._dense_nn = NearestNeighbors(
                n_neighbors=min(1000, len(chunks)),
                metric="cosine",
                algorithm="brute",
            )
            self._dense_nn.fit(dense_embeddings)

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        data_dir: Path | str = "data",
        config: IndexConfig | None = None,
        articles: pd.DataFrame | None = None,
        show_progress: bool = True,
    ) -> "PassageIndex":
        config = config or IndexConfig()
        if articles is None:
            df = load_isot(data_dir)
            articles = stratified_sample(
                df, n_articles=config.n_articles, random_state=config.random_state
            )

        chunks = articles_to_chunks(
            articles,
            chunk_words=config.chunk_words,
            overlap=config.overlap,
            fields=config.fields,
        )
        if chunks.empty:
            raise RuntimeError("No passages produced — check data and chunk settings.")

        return cls._fit_chunks(chunks, config, show_progress=show_progress)

    @classmethod
    def from_passages(
        cls,
        passages: pd.DataFrame,
        config: IndexConfig | None = None,
        dense_encoder: DenseEncoder | None = None,
        show_progress: bool = False,
    ) -> "PassageIndex":
        """Index an already-built passage table (used for SciFact abstracts).

        ``passages`` must include ``passage`` and ``chunk_id``. Missing hit
        columns are filled so :meth:`query` can return :class:`Hit` rows.
        """
        config = config or IndexConfig(sparse_backend="bm25")
        chunks = _coerce_passages(passages)
        return cls._fit_chunks(
            chunks,
            config,
            show_progress=show_progress,
            dense_encoder=dense_encoder,
        )

    @classmethod
    def _fit_chunks(
        cls,
        chunks: pd.DataFrame,
        config: IndexConfig,
        show_progress: bool = True,
        dense_encoder: DenseEncoder | None = None,
    ) -> "PassageIndex":
        texts = chunks["passage"].tolist()
        sparse: SparseEncoder | None = None
        sparse_matrix = None
        bm25: BM25Encoder | None = None
        if config.sparse_backend == "tfidf":
            sparse = SparseEncoder().fit(texts)
            sparse_matrix = sparse.encode(texts)
        elif config.sparse_backend == "bm25":
            bm25 = BM25Encoder(k1=config.bm25_k1, b=config.bm25_b).fit(
                texts, show_progress=show_progress
            )
        else:
            raise ValueError(f"Unknown sparse_backend: {config.sparse_backend}")

        if dense_encoder is None:
            dense_encoder = DenseEncoder(
                model_name=config.dense_model,
                batch_size=config.embed_batch_size,
            )
        dense_embeddings = dense_encoder.encode(texts, show_progress=show_progress)

        return cls(
            chunks=chunks,
            sparse=sparse,
            sparse_matrix=sparse_matrix,
            dense_embeddings=dense_embeddings,
            dense_encoder=dense_encoder,
            config=config,
            bm25=bm25,
        )

    # --------------------------------------------------------------- persist
    def save(self, out_dir: Path | str) -> Path:
        import joblib
        from scipy import sparse as sp

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.chunks.to_parquet(out_dir / "chunks.parquet", index=False)
        if self.dense_embeddings is None:
            raise RuntimeError("Cannot save an index without dense embeddings.")
        np.save(out_dir / "dense_embeddings.npy", self.dense_embeddings)
        if self.sparse_matrix is not None and self.sparse is not None:
            sp.save_npz(out_dir / "sparse_matrix.npz", self.sparse_matrix)
            joblib.dump(self.sparse.vectorizer, out_dir / "tfidf_vectorizer.joblib")
        if self.bm25 is not None:
            self.bm25.save(out_dir / "bm25")
        with (out_dir / "config.json").open("w", encoding="utf-8") as f:
            cfg = asdict(self.config)
            cfg["fields"] = list(self.config.fields)
            json.dump(cfg, f, indent=2)
        meta = {
            "n_chunks": int(len(self.chunks)),
            "n_articles": int(self.chunks["article_id"].nunique()),
            "dense_dim": int(self.dense_embeddings.shape[1])
            if self.dense_embeddings is not None
            else None,
        }
        with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return out_dir

    @classmethod
    def load(cls, index_dir: Path | str) -> "PassageIndex":
        import joblib
        from scipy import sparse as sp

        index_dir = Path(index_dir)
        chunks = pd.read_parquet(index_dir / "chunks.parquet")
        dense_embeddings = np.load(index_dir / "dense_embeddings.npy").astype(np.float32)
        with (index_dir / "config.json").open(encoding="utf-8") as f:
            raw = json.load(f)
        config = IndexConfig(
            n_articles=int(raw["n_articles"]),
            chunk_words=int(raw["chunk_words"]),
            overlap=int(raw["overlap"]),
            fields=tuple(raw.get("fields", ["body"])),
            dense_model=raw.get("dense_model", DEFAULT_DENSE_MODEL),
            random_state=int(raw.get("random_state", 7)),
            embed_batch_size=int(raw.get("embed_batch_size", 64)),
            sparse_backend=raw.get("sparse_backend", "tfidf"),
            bm25_k1=float(raw.get("bm25_k1", DEFAULT_BM25_K1)),
            bm25_b=float(raw.get("bm25_b", DEFAULT_BM25_B)),
        )

        sparse = None
        sparse_matrix = None
        vectorizer_path = index_dir / "tfidf_vectorizer.joblib"
        matrix_path = index_dir / "sparse_matrix.npz"
        if vectorizer_path.exists() and matrix_path.exists():
            sparse_matrix = sp.load_npz(matrix_path).tocsr()
            sparse = SparseEncoder()
            sparse.vectorizer = joblib.load(vectorizer_path)

        bm25 = None
        bm25_dir = index_dir / "bm25"
        if bm25_dir.exists():
            bm25 = BM25Encoder.load(bm25_dir)
        if config.sparse_backend == "tfidf" and sparse is None:
            raise FileNotFoundError(f"TF-IDF artifacts missing in {index_dir}")
        if config.sparse_backend == "bm25" and bm25 is None:
            raise FileNotFoundError(f"BM25 artifacts missing in {index_dir}")

        dense_encoder = DenseEncoder(model_name=config.dense_model)

        return cls(
            chunks=chunks,
            sparse=sparse,
            sparse_matrix=sparse_matrix,
            dense_embeddings=dense_embeddings,
            dense_encoder=dense_encoder,
            config=config,
            bm25=bm25,
        )

    # ---------------------------------------------------------------- query
    def query(
        self,
        text: str,
        top_k: int = 5,
        method: Method = "hybrid",
        exclude_article_ids: set[int] | None = None,
    ) -> list[Hit]:
        text = (text or "").strip()
        if not text:
            raise ValueError("query text must be non-empty")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        fetch = min(len(self.chunks), max(top_k * 15, top_k + 30))
        exclude = exclude_article_ids or set()

        if method == "tfidf":
            ranked = self._rank_sparse(text, fetch)
        elif method == "bm25":
            ranked = self._rank_bm25(text, fetch)
        elif method == "dense":
            ranked = self._rank_dense(text, fetch)
        elif method == "hybrid":
            ranked = self._rank_hybrid(text, fetch)
        else:
            raise ValueError(f"Unknown method: {method}")

        hits: list[Hit] = []
        for chunk_idx, score in ranked:
            row = self.chunks.iloc[int(chunk_idx)]
            if int(row["article_id"]) in exclude:
                continue
            hits.append(
                Hit(
                    rank=len(hits) + 1,
                    score=float(score),
                    chunk_id=str(row["chunk_id"]),
                    article_id=int(row["article_id"]),
                    title=str(row["title"]),
                    label=int(row["label"]),
                    label_name=str(row["label_name"]),
                    subject=str(row["subject"]),
                    date=str(row["date"]),
                    passage=str(row["passage"]),
                    method=method,
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def query_df(self, text: str, **kwargs) -> pd.DataFrame:
        hits = self.query(text, **kwargs)
        if not hits:
            return pd.DataFrame(
                columns=[
                    "rank",
                    "score",
                    "title",
                    "label_name",
                    "label",
                    "subject",
                    "date",
                    "article_id",
                    "chunk_id",
                    "passage",
                    "method",
                ]
            )
        return pd.DataFrame([asdict(h) for h in hits])

    def _rank_sparse(self, text: str, fetch: int) -> list[tuple[int, float]]:
        if self.sparse is None or self.sparse_matrix is None:
            raise RuntimeError(
                "This index has no TF-IDF backend. Rebuild with sparse_backend='tfidf'."
            )
        q = self.sparse.encode([text])
        # cosine via dot product; sklearn TfidfVectorizer L2-normalizes rows by default.
        scores = (self.sparse_matrix @ q.T)
        scores = np.asarray(scores.todense() if hasattr(scores, "todense") else scores).reshape(-1)
        if fetch >= len(scores):
            order = np.argsort(-scores)
        else:
            order = np.argpartition(-scores, fetch)[:fetch]
            order = order[np.argsort(-scores[order])]
        return [(int(i), float(scores[i])) for i in order[:fetch]]

    def _rank_bm25(self, text: str, fetch: int) -> list[tuple[int, float]]:
        if self.bm25 is None:
            raise RuntimeError(
                "This index has no BM25 backend. Rebuild with sparse_backend='bm25'."
            )
        return self.bm25.rank(text, fetch)

    def _sparse_rank_for_hybrid(self, text: str, fetch: int) -> list[tuple[int, float]]:
        """TF-IDF when that is the configured arm; BM25 on a BM25 index."""
        if self.config.sparse_backend == "bm25":
            return self._rank_bm25(text, fetch)
        return self._rank_sparse(text, fetch)

    def _rank_dense(self, text: str, fetch: int) -> list[tuple[int, float]]:
        if self.dense_encoder is None or self._dense_nn is None:
            raise RuntimeError("Dense index is not available.")
        q = self.dense_encoder.encode([text])
        distances, indices = self._dense_nn.kneighbors(q, n_neighbors=fetch)
        out = []
        for dist, idx in zip(distances[0], indices[0]):
            out.append((int(idx), float(1.0 - dist)))
        return out

    def _rank_hybrid(self, text: str, fetch: int, k_rrf: int = 60) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion of sparse + dense lists."""
        sparse_rank = self._sparse_rank_for_hybrid(text, fetch)
        dense_rank = self._rank_dense(text, fetch)
        return reciprocal_rank_fusion(
            [sparse_rank, dense_rank], k_rrf=k_rrf, fetch=fetch
        )
