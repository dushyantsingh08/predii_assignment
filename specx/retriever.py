"""Hybrid retrieval: dense (FAISS) + sparse (BM25) fused with Reciprocal Rank Fusion.

Why hybrid?  Dense embeddings capture meaning ("tightening value" ~ "torque")
but blur exact tokens; BM25 is exact on codes such as ``WSS-M6C62-A`` or
``XY-75W140-QL`` and on component words.  Service manuals need both.

After fusion, chunks whose *type* matches the query intent (e.g. a torque table
for a torque question) get a multiplicative boost, and authoritative
``SPECIFICATIONS`` documents get a small one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .chunker import Chunk
from .cleaning import stem
from .config import Settings
from .embeddings import Embedder
from .query import QueryInfo, analyze_query, overlap_score

_TOK = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lower-case, keep hyphenated codes whole *and* as pieces, light stemming."""
    out: list[str] = []
    for t in _TOK.findall(text.lower()):
        out.append(stem(t))
        if re.search(r"[-./]", t):
            out.extend(stem(p) for p in re.split(r"[-./]", t) if len(p) > 1)
    return out


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    boost: float = 1.0


class HybridRetriever:
    def __init__(self, chunks: Sequence[Chunk], embeddings: np.ndarray, embedder: Embedder, settings: Settings):
        import faiss
        from rank_bm25 import BM25Okapi

        self.chunks = list(chunks)
        self.embedder = embedder
        self.s = settings
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.ascontiguousarray(embeddings, dtype="float32"))
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])

    # ------------------------------------------------------------------
    def search(self, query: str, k: Optional[int] = None, info: Optional[QueryInfo] = None) -> list[Hit]:
        k = k or self.s.top_k
        info = info or analyze_query(query)
        cand = min(self.s.candidate_k, len(self.chunks))

        # dense
        qvec = np.ascontiguousarray(self.embedder.encode_query(query)[None, :], dtype="float32")
        _, d_idx = self.index.search(qvec, cand)
        dense_rank = {int(i): r for r, i in enumerate(d_idx[0]) if i >= 0}

        # sparse (query expanded with intent words so "tightening" finds "Torque Specifications")
        present = set(tokenize(query))
        extra = " ".join(w for w in info.expansion.split() if stem(w) not in present)   # no double-counting
        scores = self.bm25.get_scores(tokenize(f"{query} {extra}"))
        s_idx = np.argsort(-scores)[:cand]
        sparse_rank = {int(i): r for r, i in enumerate(s_idx) if scores[i] > 0}

        fused: dict[int, float] = {}
        for i, r in dense_rank.items():
            fused[i] = fused.get(i, 0.0) + self.s.dense_weight / (self.s.rrf_k + r + 1)
        for i, r in sparse_rank.items():
            fused[i] = fused.get(i, 0.0) + self.s.sparse_weight / (self.s.rrf_k + r + 1)

        hits: list[Hit] = []
        for i, base in fused.items():
            c = self.chunks[i]
            boost = 1.0
            if c.chunk_type in info.chunk_types:
                boost *= self.s.intent_boost
            elif info.intent == "torque" and c.has_torque:
                boost *= 1 + (self.s.intent_boost - 1) / 2          # procedure text with inline torque
            if c.doc_type == "SPECIFICATIONS":
                boost *= self.s.spec_doc_boost
            # term coverage: a chunk that mentions *all* the query's content words beats one that mentions some
            w = self.s.coverage_weight
            boost *= 1 - w / 2 + w * overlap_score(info, c.text)
            hits.append(Hit(c, base * boost, dense_rank.get(i), sparse_rank.get(i), boost))
        hits.sort(key=lambda h: -h.score)
        return hits[:k]
