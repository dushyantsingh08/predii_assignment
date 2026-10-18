"""Structure-aware chunking.

Fixed-size sliding windows would cut spec tables in half and mix unrelated
components.  Instead the chunker follows the manual's own structure:

* every **spec table** (torque / material / general / part numbers) becomes its
  own chunk - long ones are split by row groups with the header repeated;
* **procedures** are split at component / phase headings (``Brake Disc > Removal``)
  and then at paragraph boundaries once they exceed ``max_chunk_words``;
* tiny fragments are merged into their neighbour.

Each chunk starts with a one-line *breadcrumb* such as::

    [206-03 Front Disc Brake | Specifications | Torque Specifications | p.636]

That "contextual header" is embedded together with the text, which is what lets
a query like "front brake caliper bolts" find the right table even though the
table body never says "front".
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Optional

from .cleaning import word_count
from .config import Settings
from .pdf_loader import Element, LogicalDoc, ManualData, table_to_markdown

SPEC_TABLE_KINDS = {"torque", "material", "general", "parts"}

TABLE_LABELS = {
    "torque": "Torque Specifications (Nm, lb-ft, lb-in)",
    "material": "Material: fluids, lubricants, sealants - part numbers, specification codes, fill capacity",
    "general": "General Specifications",
    "parts": "Part numbers",
}
CHUNK_TYPE_BY_TABLE = {
    "torque": "torque_table",
    "material": "material_table",
    "general": "general_table",
    "parts": "parts_table",
    "other": "other_table",
}
_TORQUE_RE = re.compile(r"\d\s*(?:Nm|lb-ft|lb-in)\b")
_FOOTNOTE_RE = re.compile(r"^[a-z]\s+\S")


@dataclass
class Chunk:
    id: str
    text: str                                   # breadcrumb + body: what is embedded and shown to the LLM
    body: str
    chunk_type: str                             # torque_table | material_table | general_table | parts_table | text
    file_id: str
    section_id: str
    section_title: str
    doc_type: str
    heading_path: list[str]
    page_start: int
    page_end: int
    table_kind: str = ""
    header: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    footnote: str = ""
    segments: list[tuple[int, str]] = field(default_factory=list)    # (page, text) for page attribution
    has_torque: bool = False

    @property
    def section_label(self) -> str:
        return f"{self.section_id} {self.section_title}".strip()

    def page_of(self, needle: str) -> int:
        """Best-effort page that contains ``needle`` (falls back to the first page)."""
        n = re.sub(r"\s+", " ", needle).strip().lower()
        if n:
            for page, text in self.segments:
                if n in re.sub(r"\s+", " ", text).lower():
                    return page
        return self.page_start

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        d = dict(d)
        d["segments"] = [tuple(s) for s in d.get("segments", [])]
        return cls(**d)


def _split_long(text: str, max_words: int) -> list[str]:
    """Split an oversized paragraph at line boundaries, preferring step starts ("3. ...")."""
    if word_count(text) <= max_words:
        return [text]
    pieces, cur, cur_words = [], [], 0
    for line in text.split("\n"):
        lw = word_count(line)
        starts_step = bool(re.match(r"^\d+\.\s", line))
        if cur and cur_words + lw > max_words and (starts_step or cur_words > max_words * 0.8 or lw > max_words):
            pieces.append("\n".join(cur))
            cur, cur_words = [], 0
        cur.append(line)
        cur_words += lw
    if cur:
        pieces.append("\n".join(cur))
    return pieces


def _breadcrumb(doc: LogicalDoc, path: list[str], p0: int, p1: int) -> str:
    parts = [doc.section_label or "Unknown section", doc.doc_type.title() if doc.doc_type else ""]
    parts += [p for p in path if p]
    pages = f"p.{p0}" if p0 == p1 else f"p.{p0}-{p1}"
    return "[" + " | ".join(x for x in parts if x) + f" | {pages}]"


class _DocChunker:
    def __init__(self, doc: LogicalDoc, doc_idx: int, settings: Settings):
        self.doc = doc
        self.idx = doc_idx
        self.s = settings
        self.chunks: list[Chunk] = []
        self.h1 = ""
        self.h2 = ""
        self.buf: list[tuple[int, str]] = []
        self.buf_words = 0               # words of real content (headings excluded)
        self.buf_path: list[str] = []
        self.last_table_chunk: Optional[Chunk] = None

    # -- helpers ---------------------------------------------------------
    def _new_id(self) -> str:
        return f"d{self.idx:03d}-{len(self.chunks):02d}"

    def _add(self, page: int, text: str, content: bool = True) -> None:
        if not self.buf:
            self.buf_path = [self.h1, self.h2]
        self.buf.append((page, text))
        if content:
            self.buf_words += word_count(text)

    def _add_prose(self, page: int, text: str) -> None:
        """Add (possibly oversized) prose, flushing so chunks stay near ``max_chunk_words``."""
        limit = self.s.max_chunk_words
        for piece in _split_long(text, limit):
            if self.buf_words and self.buf_words + word_count(piece) > limit * 1.2:
                self._flush()
            self._add(page, piece)

    def _flush(self) -> None:
        if self.buf_words == 0:           # only heading markers: nothing worth indexing
            self.buf, self.buf_words = [], 0
            return
        body = "\n\n".join(t for _, t in self.buf)
        p0, p1 = self.buf[0][0], self.buf[-1][0]
        prev = self.chunks[-1] if self.chunks else None

        # merge tiny fragments into the previous text chunk of the same component
        if (
            self.buf_words < self.s.min_chunk_words
            and prev is not None
            and prev.chunk_type == "text"
            and prev.heading_path[:1] == self.buf_path[:1]
            and word_count(prev.body) + self.buf_words <= self.s.max_chunk_words * 1.3
        ):
            prev.body += "\n\n" + body
            prev.segments += list(self.buf)
            prev.page_end = max(prev.page_end, p1)
            prev.has_torque = prev.has_torque or bool(_TORQUE_RE.search(body))
            prev.text = _breadcrumb(self.doc, prev.heading_path, prev.page_start, prev.page_end) + "\n" + prev.body
        else:
            path = [p for p in self.buf_path if p]
            self.chunks.append(
                Chunk(
                    id=self._new_id(), text=_breadcrumb(self.doc, path, p0, p1) + "\n" + body, body=body,
                    chunk_type="text", file_id=self.doc.file_id, section_id=self.doc.section_id,
                    section_title=self.doc.section_title, doc_type=self.doc.doc_type, heading_path=path,
                    page_start=p0, page_end=p1, segments=list(self.buf),
                    has_torque=bool(_TORQUE_RE.search(body)),
                )
            )
        self.buf, self.buf_words = [], 0
        self.last_table_chunk = None

    def _emit_table(self, el: Element) -> None:
        label = TABLE_LABELS.get(el.table_kind, "Table")
        # drop a phase heading that the table label already says ("Material" / "Torque Specifications")
        h2 = "" if self.h2 and self.h2.lower() in label.lower() else self.h2
        path = [p for p in (self.h1, h2, label) if p]
        step = self.s.table_max_rows
        groups = [el.rows[i: i + step] for i in range(0, len(el.rows), step)] or [[]]
        for g in groups:
            md = table_to_markdown(el.table_kind, el.header, g)
            chunk = Chunk(
                id=self._new_id(), text=_breadcrumb(self.doc, path, el.page, el.page) + "\n" + md, body=md,
                chunk_type=CHUNK_TYPE_BY_TABLE[el.table_kind], file_id=self.doc.file_id,
                section_id=self.doc.section_id, section_title=self.doc.section_title, doc_type=self.doc.doc_type,
                heading_path=path, page_start=el.page, page_end=el.page, table_kind=el.table_kind,
                header=list(el.header), rows=[list(r) for r in g], segments=[(el.page, md)],
                has_torque=el.table_kind == "torque",
            )
            self.chunks.append(chunk)
            self.last_table_chunk = chunk

    def _attach_footnote(self, el: Element) -> bool:
        c = self.last_table_chunk
        if c is None or self.buf_words or not _FOOTNOTE_RE.match(el.text) or word_count(el.text) > 20:
            return False
        c.footnote = el.text.strip()
        c.body += "\n" + c.footnote
        c.text += "\n" + c.footnote
        c.segments.append((el.page, c.footnote))
        return True

    # -- main loop ---------------------------------------------------------
    def run(self) -> list[Chunk]:
        for el in self.doc.elements:
            if el.kind == "heading":
                if el.level == 1:
                    self._flush()
                    self.h1, self.h2 = el.text, ""
                else:
                    if self.buf_words >= self.s.min_chunk_words:
                        self._flush()
                    self.h2 = el.text
                    self._add(el.page, f"## {el.text}", content=False)
            elif el.kind == "table" and el.table_kind in SPEC_TABLE_KINDS:
                self._flush()
                self._emit_table(el)
            elif el.kind == "table":                      # diagnostic / tool tables: keep as prose, but size-limited
                self._add_prose(el.page, el.text)
            else:
                if self._attach_footnote(el):
                    continue
                self._add_prose(el.page, el.text)
        self._flush()
        return self.chunks


def build_chunks(manual: ManualData, settings: Optional[Settings] = None) -> list[Chunk]:
    settings = settings or Settings()
    out: list[Chunk] = []
    for i, doc in enumerate(manual.docs):
        out.extend(_DocChunker(doc, i, settings).run())
    return out
