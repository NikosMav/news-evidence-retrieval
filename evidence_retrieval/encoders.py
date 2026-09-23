"""Sparse TF-IDF, BM25, and dense sentence-transformer encoders."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Anserini's BEIR flat-BM25 setting (k1=0.9, b=0.4), not Elasticsearch's default.
DEFAULT_BM25_K1 = 0.9
DEFAULT_BM25_B = 0.4


@dataclass
class SparseEncoder:
    """Fit-once TF-IDF encoder over passage text."""

    max_features: int = 50_000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    vectorizer: TfidfVectorizer | None = field(default=None, repr=False)

    def fit(self, texts: Sequence[str]) -> "SparseEncoder":
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=True,
        )
        self.vectorizer.fit(list(texts))
        return self

    def encode(self, texts: Sequence[str]):
        if self.vectorizer is None:
            raise RuntimeError("SparseEncoder.fit() must be called before encode().")
        return self.vectorizer.transform(list(texts))


@dataclass
class BM25Encoder:
    """Lucene-compatible BM25 via ``bm25s``, beside :class:`SparseEncoder`.

    TF-IDF stays the default sparse arm so the ISOT title-recovery table can
    still be regenerated. SciFact uses this encoder. Tokenization is lowercase,
    English stopwords, and a Snowball English stemmer (PyStemmer).
    """

    k1: float = DEFAULT_BM25_K1
    b: float = DEFAULT_BM25_B
    method: str = "lucene"
    stopwords: str = "english"
    stem: bool = True
    n_docs: int = 0
    _retriever: object | None = field(default=None, repr=False)
    _stemmer: object | None = field(default=None, repr=False)

    def _stemmer_obj(self):
        if not self.stem:
            return None
        if self._stemmer is None:
            from Stemmer import Stemmer

            self._stemmer = Stemmer("english")
        return self._stemmer

    def tokenize(self, texts: Sequence[str], show_progress: bool = False):
        import bm25s

        return bm25s.tokenize(
            list(texts),
            stopwords=self.stopwords,
            stemmer=self._stemmer_obj(),
            show_progress=show_progress,
            leave=False,
        )

    def fit(self, texts: Sequence[str], show_progress: bool = False) -> "BM25Encoder":
        import bm25s

        documents = list(texts)
        self.n_docs = len(documents)
        if self.n_docs < 1:
            raise ValueError("BM25Encoder.fit() requires at least one document")
        tokens = self.tokenize(documents, show_progress=show_progress)
        self._retriever = bm25s.BM25(k1=self.k1, b=self.b, method=self.method)
        self._retriever.index(tokens, show_progress=show_progress, leave_progress=False)
        return self

    def rank(self, text: str, fetch: int) -> list[tuple[int, float]]:
        if self._retriever is None or self.n_docs < 1:
            raise RuntimeError("BM25Encoder.fit() must be called before rank().")
        if fetch < 1:
            raise ValueError("fetch must be >= 1")
        k = min(int(fetch), self.n_docs)
        tokens = self.tokenize([text], show_progress=False)
        result = self._retriever.retrieve(tokens, k=k, sorted=True, show_progress=False)
        indices = np.asarray(result.documents[0]).reshape(-1)
        scores = np.asarray(result.scores[0]).reshape(-1)
        ranked: list[tuple[int, float]] = []
        for idx, score in zip(indices, scores):
            doc_idx = int(idx)
            if doc_idx < 0:
                continue
            ranked.append((doc_idx, float(score)))
            if len(ranked) >= k:
                break
        return ranked

    def save(self, out_dir: Path | str) -> Path:
        if self._retriever is None:
            raise RuntimeError("BM25Encoder.fit() must be called before save().")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._retriever.save(str(out_dir), show_progress=False)
        meta = {
            "k1": self.k1,
            "b": self.b,
            "method": self.method,
            "stopwords": self.stopwords,
            "stem": self.stem,
            "n_docs": self.n_docs,
            "library": "bm25s",
        }
        (out_dir / "bm25_encoder.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return out_dir

    @classmethod
    def load(cls, index_dir: Path | str) -> "BM25Encoder":
        import bm25s

        index_dir = Path(index_dir)
        meta_path = index_dir / "bm25_encoder.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        encoder = cls(
            k1=float(meta.get("k1", DEFAULT_BM25_K1)),
            b=float(meta.get("b", DEFAULT_BM25_B)),
            method=str(meta.get("method", "lucene")),
            stopwords=str(meta.get("stopwords", "english")),
            stem=bool(meta.get("stem", True)),
            n_docs=int(meta.get("n_docs", 0)),
        )
        encoder._retriever = bm25s.BM25.load(
            str(index_dir), load_corpus=False, show_progress=False
        )
        if encoder.n_docs < 1:
            scores = getattr(encoder._retriever, "scores", None)
            if scores is not None and getattr(scores, "shape", None):
                encoder.n_docs = int(scores.shape[1])
        return encoder


@dataclass
class DenseEncoder:
    """Local sentence-transformers encoder (CPU-friendly by default)."""

    model_name: str = DEFAULT_DENSE_MODEL
    batch_size: int = 64
    _model: object | None = field(default=None, repr=False)

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(
        self,
        texts: Sequence[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        model = self._ensure_model()
        emb = model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return normalize(np.asarray(emb, dtype=np.float32), norm="l2")
