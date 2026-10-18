"""LLM-based structured extraction from retrieved chunks.

The prompt gives the model the retrieved excerpts (each with its breadcrumb and
a ``[C#]`` handle) and demands a strict JSON object.  Afterwards every record is

1. validated against the :class:`SpecRecord` schema,
2. **grounded**: the returned value must literally occur in the cited excerpt
   (numbers are compared with thousands separators removed).  Records that fail
   are kept but flagged ``verified=False`` so they can be dropped or reviewed.
3. attributed to a PDF page using the chunk's per-page segments.
"""
from __future__ import annotations

import json
import re
from typing import Optional, Sequence

from .chunker import Chunk
from .llm import LLM
from .schema import Quantity, SpecRecord, normalize_spec_type
from .units import numbers_in

SYSTEM_PROMPT = """You extract vehicle specifications from excerpts of an automotive workshop (service) manual.

Rules
1. Use ONLY values that appear in the excerpts. Never calculate, convert units, estimate, or fill gaps from memory.
2. Return one record per component / specification / variant that answers the question. If the question is broad
   (e.g. "brake caliper bolts") return every matching component, including front AND rear systems when both appear.
3. Torque tables have columns Nm | lb-ft | lb-in. A dash (—) means that unit is not given. Put the Nm figure in
   "value"/"unit" when present; put the other units in "alt_values". If only lb-in or lb-ft is given, use that as the primary.
4. Variants (e.g. "(SVT Raptor)", "4WD", an axle size, an engine) go in "condition"; emit separate records per variant.
5. Footnote rows such as "Wheel nuts a" with only dashes mean "refer to the procedure": return value null and say so in "notes".
6. Procedure text often says only "To install, tighten to 37 Nm (27 lb-ft)" without naming the fastener (the name is in a
   figure). If the same Nm value appears in a torque table of the same section, use that table's component name. Otherwise
   name the component from the nearest heading / sentence and explain in "notes". Prefer table values over prose.
7. spec_type must be one of: "Torque", "Fluid Capacity", "Fluid Specification", "Part Number", "Specification".
   - Torque: fastener tightening torque.            - Fluid Capacity: fill volume (e.g. 1,420 ml).
   - Fluid Specification: Ford spec code (e.g. WSS-M6C62-A).   - Part Number: Motorcraft/Ford part code (e.g. PM-1-C, W506545).
   - Specification: any other limit / clearance / thickness / preload.
8. "value" is a string with no thousands separators ("1420"); ranges stay as written ("1.5-3.0"). Part and spec codes are copied exactly.
9. "source" is the excerpt handle ("C1", "C2" ...) the value came from. "evidence" is a short verbatim snippet (<= 160 chars).
10. If nothing in the excerpts answers the question, return {"results": []}.

Output JSON only, no prose, in exactly this shape:
{"results": [{"component": str, "spec_type": str, "value": str|null, "unit": str|null,
              "alt_values": [{"value": str, "unit": str}], "condition": str|null, "notes": str|null,
              "source": "C1", "evidence": str}]}"""


class ExtractionError(RuntimeError):
    pass


def build_user_prompt(question: str, chunks: Sequence[Chunk]) -> str:
    parts = [f"Question: {question}", "", "Excerpts from the manual:"]
    for i, c in enumerate(chunks, 1):
        parts.append(f"\n[C{i}]\n{c.text}")
    return "\n".join(parts)


def parse_json_object(raw: str) -> dict:
    """Parse the model reply, tolerating code fences or stray prose around the JSON."""
    txt = raw.strip()
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.I)
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        start, end = txt.find("{"), txt.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(txt[start: end + 1])
            except json.JSONDecodeError:
                pass
    raise ExtractionError(f"LLM did not return valid JSON: {raw[:200]!r}")


def _norm_text(s: str) -> str:
    return re.sub(r"[\s,]+", "", s.replace("\u2014", "-").replace("\u2013", "-")).lower()


def is_grounded(value: Optional[str], chunk_text: str) -> bool:
    """True if ``value`` literally occurs in ``chunk_text`` (format-tolerant).

    Plain numbers / ranges are matched on token boundaries, so "99" is not found inside "p.599"
    or "1,299". Codes ("WSS-M6C62-A") are matched as whitespace-insensitive substrings.
    """
    if value is None or not str(value).strip():
        return False
    v = str(value).strip()
    if re.fullmatch(r"[\d.,\s\-\u2013]+", v):
        nums = numbers_in(v)
        flat = chunk_text.replace(",", "")
        return bool(nums) and all(re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", flat) for n in nums)
    return _norm_text(v) in _norm_text(chunk_text)


def find_support(value: Optional[str], component: str, cited: Optional[Chunk], chunks: Sequence[Chunk]) -> Optional[Chunk]:
    """Chunk that backs ``value``: the cited one, else any chunk with the value on a line naming the component."""
    if cited is not None and is_grounded(value, cited.text):
        return cited
    words = re.findall(r"[a-z]{4,}", component.lower())
    for c in chunks:
        for line in c.text.split("\n"):
            if is_grounded(value, line) and any(w in line.lower() for w in words):
                return c
    return None


def _clean_value(raw) -> Optional[str]:
    """Normalise the model's value: drop thousands separators from plain numbers, keep codes verbatim."""
    if raw in (None, ""):
        return None
    v = str(raw).strip()
    return v.replace(",", "") if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?|\d+(\.\d+)?", v) else v


def extract_with_llm(llm: LLM, question: str, chunks: Sequence[Chunk]) -> list[SpecRecord]:
    if not chunks:
        return []
    raw = llm.complete(SYSTEM_PROMPT, build_user_prompt(question, chunks))
    try:
        payload = parse_json_object(raw)
    except ExtractionError:                               # one repair attempt
        raw = llm.complete(SYSTEM_PROMPT, build_user_prompt(question, chunks) + "\n\nReturn ONLY valid JSON.")
        payload = parse_json_object(raw)

    items = payload.get("results", payload) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ExtractionError("JSON has no 'results' list")

    records: list[SpecRecord] = []
    for it in items:
        if not isinstance(it, dict) or not it.get("component"):
            continue
        handle = str(it.get("source") or "")
        m = re.search(r"\d+", handle)
        idx = int(m.group()) - 1 if m else -1
        chunk = chunks[idx] if 0 <= idx < len(chunks) else None
        value = _clean_value(it.get("value"))

        # grounding: the value must be in the cited excerpt, or on a line that also names the component
        support = find_support(value, str(it["component"]), chunk, chunks)
        anchor = support or chunk or chunks[0]
        verified = support is not None or (value is None and chunk is not None)

        alts = []
        for a in it.get("alt_values") or []:
            if isinstance(a, dict) and a.get("value") not in (None, ""):
                alts.append(Quantity(value=str(a["value"]), unit=a.get("unit")))
        try:
            records.append(
                SpecRecord(
                    component=str(it["component"]).strip(), spec_type=normalize_spec_type(it.get("spec_type")),
                    value=value, unit=it.get("unit") or None, alt_values=alts,
                    condition=it.get("condition") or None, notes=it.get("notes") or None,
                    section=anchor.section_label, page=anchor.page_of(str(value or it.get("evidence") or "")),
                    source="llm", verified=verified, evidence=(it.get("evidence") or None), chunk_id=anchor.id,
                )
            )
        except Exception:      # malformed item: skip rather than fail the whole answer
            continue
    return records
