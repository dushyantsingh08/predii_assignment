"""Layout-aware PDF extraction.

Plain ``page.get_text()`` is a poor fit for this manual because

* the text stream is *not* in reading order (the page header is emitted last),
* spec tables come out one cell per line, so ``Nm / lb-ft / lb-in`` columns
  lose their alignment,
* every page carries print noise (``Page 1 sur 2``, ``file:///...``, watermark),
* the ``SECTION 206-03: ...`` header only appears on the *first* page of each
  printed document, so continuation pages have no idea which section they are in.

The loader therefore works in two stages:

1. **Per page** - find tables with PyMuPDF, rebuild reading order from line
   geometry, detect headings from font weight/size, strip header/footer noise,
   and pull out metadata (file id, page n of m, section, document type).
2. **Per logical document** - pages that share the same footer file id
   (e.g. ``SE263001.HTM``) are stitched back into one document, section
   metadata is carried onto continuation pages, and tables are classified
   (torque / material / general / parts / other).  Header-less continuation
   tables inherit the previous table's header.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from . import config as cfg
from .cleaning import clean_line, is_dash, normalize_cell

try:  # PyMuPDF >= 1.24 exposes ``pymupdf``; older releases only ``fitz``
    import pymupdf  # type: ignore
except ImportError:  # pragma: no cover
    import fitz as pymupdf  # type: ignore

log = logging.getLogger(__name__)

SECTION_RE = re.compile(r"SECTION\s+(\d{3}-\d{2}[A-Z]?)\s*:\s*(.+)")
FILE_RE = re.compile(r"file=([A-Za-z0-9_]+)\.HTM", re.I)
PAGE_RE = re.compile(r"Page\s+(\d+)\s+(?:sur|of)\s+(\d+)")
REV_RE = re.compile(r"Procedure revision date:\s*([\d/]+)")
VEHICLE_RE = re.compile(r"^(\d{4}\s+.+?)\s+Workshop Manual$", re.I)
DOCTYPE_RE = re.compile(r"^[A-Z][A-Z ,&/\-]{5,}$")
NOISE_RES = [re.compile(p) for p in cfg.NOISE_LINE_PATTERNS]
STEP_ONLY_RE = re.compile(r"^\d+\.$")

TORQUE_UNIT_HEADERS = {"nm", "lb-ft", "lb-in"}


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------

@dataclass
class Element:
    kind: str                       # "heading" | "text" | "table"
    page: int
    text: str = ""                  # heading/paragraph text, or markdown for tables
    level: int = 0                  # heading level (1 = component title, 2 = phase)
    table_kind: str = ""            # torque | material | general | parts | other
    header: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)


@dataclass
class LogicalDoc:
    file_id: str
    section_id: str = ""
    section_title: str = ""
    doc_type: str = ""
    revision: str = ""
    page_start: int = 0
    page_end: int = 0
    elements: list[Element] = field(default_factory=list)

    @property
    def section_label(self) -> str:
        return f"{self.section_id} {self.section_title}".strip()


@dataclass
class ManualData:
    source: str
    n_pages: int
    vehicle: str = ""
    docs: list[LogicalDoc] = field(default_factory=list)

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ManualData":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        docs = []
        for d in raw["docs"]:
            els = [Element(**e) for e in d.pop("elements")]
            docs.append(LogicalDoc(**d, elements=els))
        return cls(source=raw["source"], n_pages=raw["n_pages"], vehicle=raw.get("vehicle", ""), docs=docs)


# ---------------------------------------------------------------------------
# stage 1: one page
# ---------------------------------------------------------------------------

@dataclass
class _Line:
    text: str
    x0: float
    x1: float
    y0: float
    y1: float
    size: float
    bold: bool

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class _Row:
    """Lines that sit on the same visual baseline, joined left to right."""
    text: str
    x0: float
    y0: float
    y1: float
    size: float
    bold: bool
    parts: list[str] = field(default_factory=list)   # the individual lines, left to right


@dataclass
class _RawPage:
    number: int
    file_id: Optional[str] = None
    page_idx: Optional[int] = None
    page_total: Optional[int] = None
    header: dict = field(default_factory=dict)
    items: list[tuple] = field(default_factory=list)   # (y, kind, payload)
    ocr: bool = False


def _is_bold(span: dict) -> bool:
    return bool(span["flags"] & 16) or "bold" in span["font"].lower()


def _extract_lines(page, table_boxes: list[tuple[float, float, float, float]]) -> list[_Line]:
    lines: list[_Line] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for ln in block["lines"]:
            spans = [s for s in ln["spans"] if s["text"].strip()]
            if not spans:
                continue
            x0, y0, x1, y1 = ln["bbox"]
            if y1 < 0 or y0 > page.rect.height:       # off-page ghost text
                continue
            yc = (y0 + y1) / 2
            xc = (x0 + x1) / 2
            if any(bx0 - 2 <= xc <= bx1 + 2 and by0 - 2 <= yc <= by1 + 2 for bx0, by0, bx1, by1 in table_boxes):
                continue
            text = clean_line("".join(s["text"] for s in ln["spans"]))
            if not text:
                continue
            dominant = max(spans, key=lambda s: len(s["text"].strip()))
            bold = all(_is_bold(s) for s in spans if s["text"].strip() != "z")
            lines.append(_Line(text, x0, x1, y0, y1, dominant["size"], bold))
    return lines


def _group_rows(lines: list[_Line], tol: float) -> list[_Row]:
    """Rebuild reading order: cluster lines into visual rows, then sort."""
    rows: list[list[_Line]] = []
    for ln in sorted(lines, key=lambda l: (l.yc, l.x0)):
        if rows and abs(ln.yc - sum(x.yc for x in rows[-1]) / len(rows[-1])) <= tol:
            rows[-1].append(ln)
        else:
            rows.append([ln])
    out: list[_Row] = []
    for r in rows:
        r.sort(key=lambda l: l.x0)
        dominant = max(r, key=lambda l: len(l.text))
        out.append(
            _Row(
                text=" ".join(l.text for l in r),
                x0=min(l.x0 for l in r),
                y0=min(l.y0 for l in r),
                y1=max(l.y1 for l in r),
                size=dominant.size,
                bold=all(l.bold for l in r),
                parts=[l.text for l in r],
            )
        )
    return out


def _heading_level(row: _Row) -> int:
    """0 = not a heading, 1 = component title, 2 = phase / sub-heading."""
    t = row.text.strip()
    if not row.bold or row.x0 > 90 or len(t) < 3 or len(t.split()) > 9:
        return 0
    if t.endswith((".", ":", ";", ",")):
        return 0
    if re.match(r"^(\d+\.|NOTICE|WARNING|CAUTION|NOTE)\b", t, re.I):
        return 0
    return 1 if row.size >= 10.4 else 2


def _parse_header_lines(texts: list[str]) -> dict:
    """Pull section / document-type / revision out of the page-1 header block."""
    hdr: dict = {}
    for i, t in enumerate(texts):
        if m := SECTION_RE.search(t):
            hdr["section_id"], hdr["section_title"] = m.group(1), m.group(2).strip()
            # the document type is the all-caps line that follows
            for nxt in texts[i + 1: i + 3]:
                if DOCTYPE_RE.match(nxt.strip()):
                    hdr["doc_type"] = nxt.strip()
                    break
        if m := REV_RE.search(t):
            hdr["revision"] = m.group(1)
        if m := VEHICLE_RE.match(t.strip()):
            hdr["vehicle"] = m.group(1)
    return hdr


def _ocr_text(page) -> str:
    """Optional OCR for pages without a text layer."""
    try:
        import io

        import pytesseract
        from PIL import Image

        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img)
    except Exception as exc:  # pragma: no cover - depends on local tesseract
        log.warning("OCR failed on page %s: %s", page.number + 1, exc)
        return ""


def _extract_page(page, settings) -> _RawPage:
    number = page.number + 1
    raw = _RawPage(number=number)

    # --- OCR path (scanned page) ---------------------------------------
    if settings.ocr_fallback and len(page.get_text("text").strip()) < settings.ocr_min_chars:
        text = _ocr_text(page)
        if text.strip():
            raw.ocr = True
            for i, para in enumerate(p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()):
                raw.items.append((float(i), "text", clean_line(para) if "\n" not in para else para))
        return raw

    # --- tables first, so their text is not duplicated in the prose -----
    tables = []
    try:
        for tb in page.find_tables().tables:
            rows = tb.extract()
            if rows:
                tables.append((tb.bbox, rows))
    except Exception as exc:  # pragma: no cover
        log.debug("find_tables failed on page %s: %s", number, exc)
    boxes = [t[0] for t in tables]

    # --- body text in reading order ------------------------------------
    rows = _group_rows(_extract_lines(page, boxes), settings.row_tolerance_pt)

    noise_texts: list[str] = []
    header_texts: list[str] = []
    body: list[_Row] = []
    for r in rows:
        if r.y1 < cfg.TOP_NOISE_Y or r.y0 > cfg.BOTTOM_NOISE_Y:
            noise_texts.extend(r.parts)
        elif r.y0 < cfg.HEADER_BLOCK_Y and r.size <= cfg.HEADER_BLOCK_MAX_FONT:
            header_texts.extend(r.parts)           # left/right columns share a baseline
        elif any(p.match(r.text) for p in NOISE_RES):
            noise_texts.extend(r.parts)
        else:
            body.append(r)

    # --- metadata -------------------------------------------------------
    joined = "\n".join(noise_texts)
    if m := FILE_RE.search(joined):
        raw.file_id = m.group(1)
    if m := PAGE_RE.search(joined):
        raw.page_idx, raw.page_total = int(m.group(1)), int(m.group(2))
    raw.header = _parse_header_lines(header_texts + noise_texts)

    # --- interleave body rows and tables by vertical position ----------
    items: list[tuple] = [(r.y0, "row", r) for r in body]
    items += [(bbox[1], "table", rows_) for bbox, rows_ in tables]
    items.sort(key=lambda it: it[0])

    para: list[str] = []
    last_y1 = None

    def flush():
        nonlocal para
        text = "\n".join(para).strip()
        if text and not STEP_ONLY_RE.match(text):
            raw.items.append((last_y1 or 0.0, "text", text))
        para = []

    for y, kind, payload in items:
        if kind == "table":
            flush()
            raw.items.append((y, "table", payload))
            last_y1 = None
            continue
        row: _Row = payload
        lvl = _heading_level(row)
        if lvl:
            flush()
            raw.items.append((y, "heading", (lvl, row.text.strip())))
            last_y1 = None
            continue
        if para and last_y1 is not None and row.y0 - last_y1 > settings.para_gap_pt:
            flush()
        para.append(row.text)
        last_y1 = row.y1
    flush()
    return raw


# ---------------------------------------------------------------------------
# stage 2: logical documents and table classification
# ---------------------------------------------------------------------------

def _norm_table(raw_rows: list[list[str | None]]) -> list[list[str]]:
    rows = [[normalize_cell(c) for c in r] for r in raw_rows]
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    keep = [i for i in range(width) if any(r[i] for r in rows)]       # drop empty columns
    return [[r[i] for i in keep] for r in rows]


def _looks_like_torque_row(row: list[str]) -> bool:
    if len(row) < 3 or not re.search(r"[A-Za-z]", row[0]):
        return False
    return all(re.fullmatch(r"\d+(?:\.\d+)?", c) or is_dash(c) for c in row[1:])


def classify_table(rows: list[list[str]], last_torque_header: Optional[list[str]]):
    """Return ``(kind, header, body_rows)`` for a normalised table."""
    if not rows:
        return "other", [], rows
    hdr = [c.lower() for c in rows[0]]
    if hdr[0] in {"description", "item", "fastener", "component"} and any(c in TORQUE_UNIT_HEADERS for c in hdr):
        return "torque", rows[0], rows[1:]
    if "part number" in hdr:
        return "parts", rows[0], rows[1:]
    if hdr[0] == "item" and "specification" in hdr and any("capacity" in c for c in hdr):
        return "material", rows[0], rows[1:]
    if hdr[0] == "item" and len(hdr) == 2 and hdr[1] == "specification":
        return "general", rows[0], rows[1:]
    if last_torque_header and len(rows[0]) == len(last_torque_header) and _looks_like_torque_row(rows[0]):
        return "torque", last_torque_header, rows           # continuation of a table from the previous page
    return "other", [], rows


def table_to_markdown(kind: str, header: list[str], rows: list[list[str]]) -> str:
    def cells(r):
        return " | ".join(c.replace("|", "/") for c in r)

    if header and kind != "other":
        width = len(header)
        out = [f"| {cells(header)} |", "|" + "---|" * width]
        out += [f"| {cells((r + [''] * width)[:width])} |" for r in rows]
        return "\n".join(out)
    return "\n".join(cells(r) for r in rows)


def _last_heading(doc: LogicalDoc) -> str:
    for el in reversed(doc.elements):
        if el.kind == "heading":
            return el.text.strip().lower()
        if el.kind == "table":
            break
    return ""


def _assemble(raw_pages: list[_RawPage]) -> tuple[list[LogicalDoc], str]:
    meta_by_file: dict[str, dict] = {}
    vehicle = ""
    for rp in raw_pages:
        if rp.header.get("vehicle") and not vehicle:
            vehicle = rp.header["vehicle"]
        if rp.header.get("section_id") and rp.file_id:
            meta_by_file.setdefault(rp.file_id, rp.header)

    docs: list[LogicalDoc] = []
    cur: Optional[LogicalDoc] = None
    last_torque_header: Optional[list[str]] = None
    last_meta: dict = {}

    for rp in raw_pages:
        fid = rp.file_id or (cur.file_id if cur else "unknown")
        if cur is None or fid != cur.file_id:
            meta = meta_by_file.get(fid) or last_meta
            cur = LogicalDoc(
                file_id=fid,
                section_id=meta.get("section_id", ""),
                section_title=meta.get("section_title", ""),
                doc_type=meta.get("doc_type", ""),
                revision=meta.get("revision", ""),
                page_start=rp.number,
                page_end=rp.number,
            )
            docs.append(cur)
            last_meta = meta
            last_torque_header = None
        cur.page_end = rp.number

        for _y, kind, payload in rp.items:
            if kind == "heading":
                lvl, text = payload
                cur.elements.append(Element("heading", rp.number, text=text, level=lvl))
            elif kind == "text":
                cur.elements.append(Element("text", rp.number, text=payload))
            elif kind == "table":
                rows = _norm_table(payload)
                if not rows:
                    continue
                tkind, header, body = classify_table(rows, last_torque_header)
                if tkind == "general" and _last_heading(cur) == "material":
                    tkind = "material"            # name + code/spec table without a capacity column
                if tkind == "torque":
                    last_torque_header = header
                cur.elements.append(
                    Element(
                        "table", rp.number,
                        text=table_to_markdown(tkind, header, body),
                        table_kind=tkind, header=header, rows=body,
                    )
                )
    return docs, vehicle


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def load_manual(pdf_path: str | Path, settings=None, pages: Optional[range] = None, progress: bool = True) -> ManualData:
    """Parse the PDF into :class:`ManualData` (logical documents with typed elements)."""
    from .config import Settings

    settings = settings or Settings()
    doc = pymupdf.open(str(pdf_path))
    page_numbers = list(pages) if pages is not None else list(range(len(doc)))

    iterator = page_numbers
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(page_numbers, desc="Parsing PDF", unit="page")
        except ImportError:  # pragma: no cover
            pass

    raw_pages = [_extract_page(doc[i], settings) for i in iterator]
    docs, vehicle = _assemble(raw_pages)
    return ManualData(source=str(pdf_path), n_pages=len(doc), vehicle=vehicle, docs=docs)
