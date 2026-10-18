"""End-to-end pipeline:  PDF -> logical docs -> chunks -> embeddings -> retrieve -> extract."""
from __future__ import annotations

import csv
import io
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .chunker import Chunk, build_chunks
from .config import Settings
from .embeddings import make_embedder
from .extractor import ExtractionError, extract_with_llm
from .llm import LLM, get_llm
from .pdf_loader import ManualData, load_manual
from .query import QueryInfo, analyze_query, overlap_score, position_conflict
from .retriever import Hit, HybridRetriever
from .rule_parser import RuleExtractor
from .schema import CSV_COLUMNS, SpecRecord

log = logging.getLogger(__name__)


@dataclass
class QueryResult:
    query: str
    intent: str
    mode: str                                   # "llm" | "rules"
    records: list[SpecRecord]
    hits: list[Hit] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    llm_name: str = ""

    def to_json(self, full: bool = False, indent: int = 2) -> str:
        """Default: the list of records. ``full=True`` also includes the retrieval trace."""
        recs = [r.model_dump(mode="json", exclude_none=True, exclude_defaults=False) for r in self.records]
        for r in recs:                                    # keep the output compact
            if not r.get("alt_values"):
                r.pop("alt_values", None)
        if not full:
            return json.dumps(recs, indent=indent, ensure_ascii=False)
        trace = [
            {"chunk": h.chunk.id, "pages": [h.chunk.page_start, h.chunk.page_end], "type": h.chunk.chunk_type,
             "section": h.chunk.section_label, "score": round(h.score, 5),
             "dense_rank": h.dense_rank, "sparse_rank": h.sparse_rank}
            for h in self.hits
        ]
        return json.dumps(
            {"query": self.query, "intent": self.intent, "mode": self.mode, "llm": self.llm_name,
             "warnings": self.warnings, "results": recs, "retrieved": trace},
            indent=indent, ensure_ascii=False,
        )

    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in self.records:
            w.writerow(r.to_flat_dict())
        return buf.getvalue()


class Pipeline:
    def __init__(self, settings: Optional[Settings] = None, llm: Optional[LLM] = None):
        self.s = settings or Settings()
        self._llm = llm
        self.manual: Optional[ManualData] = None
        self.chunks: list[Chunk] = []
        self.retriever: Optional[HybridRetriever] = None
        self.rules: Optional[RuleExtractor] = None

    # ---- paths ------------------------------------------------------------
    @property
    def cache(self) -> Path:
        return Path(self.s.cache_dir)

    def _meta(self, pdf: Path, embedder_name: str) -> dict:
        st = pdf.stat() if pdf.exists() else None
        return {"pdf": str(pdf), "size": st.st_size if st else None, "embedder": embedder_name}

    # ---- build ------------------------------------------------------------
    def build(self, pdf_path: Optional[str] = None, force: bool = False, pages: Optional[range] = None) -> None:
        """Parse, chunk and embed the manual; cache everything under ``cache_dir``."""
        pdf = Path(pdf_path or self.s.pdf_path)
        if not pdf.exists():
            raise FileNotFoundError(f"PDF not found: {pdf}  (set SPECX_PDF or pass --pdf)")
        self.cache.mkdir(parents=True, exist_ok=True)
        manual_path = self.cache / "manual.json"

        t0 = time.time()
        if manual_path.exists() and not force and pages is None:
            self.manual = ManualData.load(manual_path)
        else:
            self.manual = load_manual(pdf, self.s, pages=pages)
            if pages is None:
                self.manual.save(manual_path)
        log.info("parsed %d logical documents in %.0fs", len(self.manual.docs), time.time() - t0)

        self.chunks = build_chunks(self.manual, self.s)
        embedder = make_embedder(self.s)
        embedder.fit([c.text for c in self.chunks])
        vecs = embedder.encode_passages([c.text for c in self.chunks])

        with open(self.cache / "chunks.jsonl", "w", encoding="utf-8") as fh:
            for c in self.chunks:
                fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        np.save(self.cache / "embeddings.npy", vecs)
        embedder.save(self.cache)
        (self.cache / "meta.json").write_text(json.dumps(self._meta(pdf, embedder.name)), encoding="utf-8")
        self._attach(embedder, vecs)

    def load(self) -> bool:
        """Load a previously built cache. Returns False if none matches the current settings."""
        needed = ["chunks.jsonl", "embeddings.npy", "meta.json"]
        if not all((self.cache / n).exists() for n in needed):
            return False
        meta = json.loads((self.cache / "meta.json").read_text(encoding="utf-8"))
        try:
            embedder = make_embedder(self.s, self.cache, load=True)
        except FileNotFoundError:                       # cache was built with a different backend
            return False
        if meta.get("embedder") != embedder.name:
            log.info("cache was built with %s, current is %s: rebuilding", meta.get("embedder"), embedder.name)
            return False
        with open(self.cache / "chunks.jsonl", encoding="utf-8") as fh:
            self.chunks = [Chunk.from_dict(json.loads(line)) for line in fh]
        vecs = np.load(self.cache / "embeddings.npy")
        self._attach(embedder, vecs)
        return True

    def ensure_ready(self, pdf_path: Optional[str] = None) -> None:
        if self.retriever is None and not self.load():
            self.build(pdf_path)

    def _attach(self, embedder, vecs: np.ndarray) -> None:
        self.retriever = HybridRetriever(self.chunks, vecs, embedder, self.s)
        self.rules = RuleExtractor(self.chunks)

    # ---- query ------------------------------------------------------------
    @property
    def llm(self) -> Optional[LLM]:
        if self._llm is None:
            self._llm = get_llm(self.s)
        return self._llm

    def query(self, question: str, mode: str = "auto", k: Optional[int] = None, strict: bool = False) -> QueryResult:
        """``mode``: auto (LLM if configured, else rules) | llm | rules."""
        self.ensure_ready()
        info = analyze_query(question)
        hits = self.retriever.search(question, k=k, info=info)
        warnings: list[str] = []

        use_llm = mode == "llm" or (mode == "auto" and self.llm is not None)
        records: list[SpecRecord] = []
        used = "rules"
        if use_llm:
            if self.llm is None:
                warnings.append("No LLM configured (set ANTHROPIC_API_KEY or OPENAI_API_KEY); used rule-based extraction.")
            else:
                try:
                    records = extract_with_llm(self.llm, question, [h.chunk for h in hits])
                    used = "llm"
                except Exception as exc:                          # network / JSON failure -> graceful fallback
                    warnings.append(f"LLM extraction failed ({type(exc).__name__}: {exc}); used rule-based extraction.")
        if used == "rules":
            records = self._rule_records(info, hits)

        records = self._dedupe(records)
        if strict:
            dropped = len([r for r in records if not r.verified])
            records = [r for r in records if r.verified]
            if dropped:
                warnings.append(f"strict mode dropped {dropped} unverified record(s)")
        return QueryResult(
            question, info.intent, used, records[: self.s.max_results], hits, warnings,
            llm_name=self.llm.name if used == "llm" and self.llm else "",
        )

    # ---- rule-based path ------------------------------------------------
    def _rule_records(self, info: QueryInfo, hits: list[Hit]) -> list[SpecRecord]:
        scored: list[tuple[float, int, SpecRecord]] = []
        for rank, h in enumerate(hits):
            for rec in self.rules.extract(h.chunk):
                if info.spec_types and rec.spec_type not in info.spec_types:
                    continue
                if rec.value is None and info.intent != "torque":
                    continue
                # context = surrounding headings (the generated table label is not part of the manual's text)
                path = h.chunk.heading_path[:-1] if h.chunk.table_kind else h.chunk.heading_path
                ctx = (rec.component, rec.condition, rec.notes, h.chunk.section_title, " ".join(path))
                if position_conflict(info, *ctx):
                    continue
                score = overlap_score(info, *ctx)
                # few terms -> require all of them; more terms -> allow one miss
                need = 1.0 if len(info.terms) <= 3 else 0.75
                if score + 1e-9 >= need:
                    # reward matches in the record's *own* name over matches in surrounding context
                    own = overlap_score(info, rec.component, rec.condition)
                    scored.append((score + 0.5 * own, rank, rec))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [r for _, _, r in scored]

    @staticmethod
    def _dedupe(records: list[SpecRecord]) -> list[SpecRecord]:
        """Merge identical facts printed in several sections; keep the first as primary."""
        by_key: dict[tuple, SpecRecord] = {}
        for r in records:
            k = r.key()
            if k not in by_key:
                by_key[k] = r
            else:
                ref = f"{r.section} p.{r.page}"
                prim = by_key[k]
                if ref != f"{prim.section} p.{prim.page}" and ref not in prim.also_found_in:
                    prim.also_found_in.append(ref)
        return list(by_key.values())
