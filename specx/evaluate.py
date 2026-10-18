"""Tiny evaluation harness.

Two independent metrics per gold item:

* **retrieval hit** - is the gold page inside one of the retrieved chunks?
  (isolates the retriever from the extractor)
* **extraction hit** - does some returned record carry the expected value (+unit)
  for the expected component?

Queries with ``"expected": []`` are *negative* tests (the manual does not cover
them): they pass only if nothing is returned - this measures hallucination.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .pipeline import Pipeline, QueryResult
from .schema import SpecRecord


@dataclass
class EvalReport:
    n_items: int = 0
    retrieval_hits: int = 0
    extraction_hits: int = 0
    n_negative: int = 0
    negative_pass: int = 0
    total_records: int = 0
    n_queries: int = 0
    failures: list[str] = field(default_factory=list)

    def summary(self) -> str:
        pos_q = self.n_queries - self.n_negative
        return (
            f"retrieval hit-rate : {self.retrieval_hits}/{self.n_items} = {self.retrieval_hits / max(1, self.n_items):.0%}\n"
            f"extraction recall  : {self.extraction_hits}/{self.n_items} = {self.extraction_hits / max(1, self.n_items):.0%}\n"
            f"negative queries   : {self.negative_pass}/{self.n_negative} correctly empty\n"
            f"avg records/query  : {self.total_records / max(1, pos_q):.1f}"
        )


def _matches(rec: SpecRecord, exp: dict) -> bool:
    hay = f"{rec.component} {rec.condition or ''} {rec.section or ''}".lower()
    if not all(tok.lower() in hay for tok in exp.get("match", [])):
        return False
    val = rec.value or ""
    if "value" in exp and val.lower() != str(exp["value"]).lower():
        return False
    if "value_contains" in exp and exp["value_contains"].lower() not in val.lower():
        return False
    if exp.get("unit") and (rec.unit or "").lower() != exp["unit"].lower():
        return False
    return True


def evaluate(pipe: Pipeline, gold_path: str | Path, mode: str = "auto", k: int | None = None, verbose: bool = False) -> EvalReport:
    gold = json.loads(Path(gold_path).read_text(encoding="utf-8"))["queries"]
    rep = EvalReport(n_queries=len(gold))
    for item in gold:
        res: QueryResult = pipe.query(item["query"], mode=mode, k=k)
        expected = item["expected"]
        if not expected:
            rep.n_negative += 1
            ok = len(res.records) == 0
            rep.negative_pass += ok
            if not ok:
                rep.failures.append(f"[negative] {item['query']!r} returned {len(res.records)} record(s)")
            continue
        rep.total_records += len(res.records)
        pages = {p for h in res.hits for p in range(h.chunk.page_start, h.chunk.page_end + 1)}
        for exp in expected:
            rep.n_items += 1
            exp_pages = exp["page"] if isinstance(exp["page"], list) else [exp["page"]]
            r_ok = any(pg in pages for pg in exp_pages)
            e_ok = any(_matches(r, exp) for r in res.records)
            rep.retrieval_hits += r_ok
            rep.extraction_hits += e_ok
            if verbose or not (r_ok and e_ok):
                tag = "ok " if (r_ok and e_ok) else "MISS"
                if not (r_ok and e_ok):
                    rep.failures.append(
                        f"[{tag}] {item['query']!r}: expected {exp.get('match')} {exp.get('value', exp.get('value_contains'))} "
                        f"{exp.get('unit', '')} p{exp['page']} (retrieved={'y' if r_ok else 'n'}, extracted={'y' if e_ok else 'n'})"
                    )
    return rep
