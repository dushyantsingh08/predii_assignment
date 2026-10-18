"""Embedding backends behind one tiny interface.

* ``sentence-transformers`` (default) - local, free, e.g. ``BAAI/bge-small-en-v1.5``
* ``openai``                          - ``text-embedding-3-small`` via the OpenAI API
* ``tfidf``                           - TF-IDF + truncated SVD (LSA). Needs no model
  download and no network; used as an offline fallback and in the test-suite.

All backends return L2-normalised float32 matrices so a FAISS inner-product
index is a cosine-similarity index.
"""
from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import Settings

TOKEN_RE = r"(?u)\b[\w][\w\-./]*\b"


def _l2(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype="float32")
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class Embedder(Protocol):
    name: str

    def fit(self, texts: list[str]) -> None: ...
    def encode_passages(self, texts: list[str]) -> np.ndarray: ...
    def encode_query(self, text: str) -> np.ndarray: ...
    def save(self, directory: Path) -> None: ...


# ---------------------------------------------------------------------------
class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is not installed. Run `pip install sentence-transformers`, "
                "or use `--embed tfidf` / `--embed openai`."
            ) from exc
        self.name = f"sentence-transformers:{model_name}"
        self.model = SentenceTransformer(model_name)
        low = model_name.lower()
        if "bge" in low:
            self.q_prefix, self.p_prefix = "Represent this sentence for searching relevant passages: ", ""
        elif "e5" in low:
            self.q_prefix, self.p_prefix = "query: ", "passage: "
        else:
            self.q_prefix = self.p_prefix = ""

    def fit(self, texts):  # nothing to fit
        return None

    def encode_passages(self, texts):
        vecs = self.model.encode(
            [self.p_prefix + t for t in texts], batch_size=32, normalize_embeddings=True, show_progress_bar=len(texts) > 64
        )
        return _l2(vecs)

    def encode_query(self, text):
        return _l2(self.model.encode([self.q_prefix + text], normalize_embeddings=True))[0]

    def save(self, directory):
        return None


class OpenAIEmbedder:
    def __init__(self, model_name: str):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("`pip install openai` to use the OpenAI embedding backend.") from exc
        self.name = f"openai:{model_name}"
        self.model_name = model_name
        self.client = OpenAI()

    def fit(self, texts):
        return None

    def _embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 96):
            resp = self.client.embeddings.create(model=self.model_name, input=texts[i: i + 96])
            out.extend(d.embedding for d in resp.data)
        return _l2(np.array(out))

    def encode_passages(self, texts):
        return self._embed(texts)

    def encode_query(self, text):
        return self._embed([text])[0]

    def save(self, directory):
        return None


class TfidfSvdEmbedder:
    """Offline fallback: TF-IDF (1-2 grams) reduced with truncated SVD."""

    def __init__(self, dim: int = 192):
        self.name = f"tfidf-svd:{dim}"
        self.dim = dim
        self.vec = None
        self.svd = None

    def fit(self, texts):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, token_pattern=TOKEN_RE, strip_accents="unicode")
        x = self.vec.fit_transform(texts)
        n_comp = max(2, min(self.dim, x.shape[0] - 1, x.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=0)
        self.svd.fit(x)

    def _enc(self, texts):
        return _l2(self.svd.transform(self.vec.transform(texts)))

    def encode_passages(self, texts):
        return self._enc(texts)

    def encode_query(self, text):
        return self._enc([text])[0]

    def save(self, directory: Path):
        with open(Path(directory) / "tfidf.pkl", "wb") as fh:
            pickle.dump((self.vec, self.svd, self.dim), fh)

    @classmethod
    def load(cls, directory: Path) -> "TfidfSvdEmbedder":
        with open(Path(directory) / "tfidf.pkl", "rb") as fh:
            vec, svd, dim = pickle.load(fh)
        e = cls(dim)
        e.vec, e.svd = vec, svd
        return e


# ---------------------------------------------------------------------------
def make_embedder(settings: Settings, directory: Path | None = None, load: bool = False) -> Embedder:
    backend = settings.embed_backend.lower()
    if backend in {"tfidf", "tfidf-svd", "lsa"}:
        return TfidfSvdEmbedder.load(directory) if load and directory else TfidfSvdEmbedder()
    if backend == "openai":
        return OpenAIEmbedder(settings.openai_embed_model)
    if backend in {"sentence-transformers", "st", "hf", "huggingface"}:
        return SentenceTransformerEmbedder(settings.embed_model)
    raise ValueError(f"Unknown embedding backend: {settings.embed_backend}")
