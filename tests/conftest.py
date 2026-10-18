import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specx.chunker import Chunk  # noqa: E402


class HashEmbedder:
    """Deterministic bag-of-words embedder so retrieval tests need no model downloads."""
    name = "hash-test"
    dim = 256

    def fit(self, texts): ...
    def save(self, d): ...

    def _vec(self, text):
        import re
        v = np.zeros(self.dim, dtype="float32")
        for t in re.findall(r"[a-z0-9]+", text.lower()):
            v[hash(t) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_passages(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def encode_query(self, text):
        return self._vec(text)


def make_chunk(cid, chunk_type, header, rows, section_id="206-03", section_title="Front Disc Brake",
               doc_type="SPECIFICATIONS", page=636, footnote="", heading_path=None, body=None, text=None):
    md = body or ""
    return Chunk(
        id=cid, text=text or f"[{section_id} {section_title} | {doc_type} | p.{page}]\n{md}", body=md,
        chunk_type=chunk_type, file_id="SE000001", section_id=section_id, section_title=section_title,
        doc_type=doc_type, heading_path=heading_path or [], page_start=page, page_end=page,
        table_kind=chunk_type.replace("_table", "") if chunk_type.endswith("_table") else "",
        header=header, rows=rows, footnote=footnote, segments=[(page, md)],
    )


@pytest.fixture
def hash_embedder():
    return HashEmbedder()


@pytest.fixture(scope="session")
def pdf_path():
    for cand in (os.getenv("SPECX_PDF"), "data/service-manual.pdf", "/mnt/user-data/uploads/sample-service-manual_1.pdf"):
        if cand and Path(cand).exists():
            return cand
    pytest.skip("service manual PDF not available")
