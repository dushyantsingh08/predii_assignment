"""Deterministic extraction from the manual's table families and procedure text.

This module has three jobs:

1. It is the **offline mode**: with no LLM configured the pipeline still returns
   structured results straight from the tables.
2. It is a **baseline** for evaluation: the LLM path should match or beat it.
3. Its helper functions power the LLM path's **grounding checks**.

Table families handled (see ``pdf_loader.classify_table``):

====================  =================================================  ===============================
kind                  columns                                            records produced
====================  =================================================  ===============================
``torque``            Description | Nm | lb-ft | lb-in                   Torque
``material``          Item | Specification | Fill Capacity               Part Number, Fluid Spec, Capacity
``general``           Item | Specification                               Specification (limits, clearances)
``parts``             Item | Part Number | Description                   Part Number
====================  =================================================  ===============================

Plus inline torque statements in procedures (``To install, tighten to 37 Nm
(27 lb-ft)``).  Those rarely name the fastener (it is in a figure), so the
component is recovered by looking up the unique torque-table row with the same
Nm value in the same section.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from .chunker import Chunk
from .cleaning import is_dash
from .schema import Quantity, SpecRecord
from .units import canon_unit, norm_number, parse_quantities

# Motorcraft-style part codes: PM-1-C, CPM-1-C, XL-3, XY-75W140-QL, TA-30 ...
PART_CODE_RE = re.compile(r"\b(?:C?[A-Z]{1,3}|[A-Z]{3})-\d+[A-Z0-9]*(?:-[A-Z0-9]+)*\b")
INLINE_TORQUE_RE = re.compile(
    r"(?P<val>\d+(?:\.\d+)?)\s*Nm\s*\(\s*(?P<alt>\d+(?:\.\d+)?)\s*(?P<altu>lb-ft|lb-in)\s*\)", re.I
)
FASTENER_RE = re.compile(
    r"(?:tighten|install|remove|torque)\s+(?:the\s+)?(?:\d+\s+)?(?:new\s+)?"
    r"(?P<obj>[A-Za-z][A-Za-z\-/ ]{2,70}?(?:bolts?|nuts?|screws?|fittings?|plugs?|studs?|clamps?))\b",
    re.I,
)
_ACRONYM_RE = re.compile(r"^[A-Z0-9/\-]{2,6}$")
_PAREN_TAIL = re.compile(r"^(?P<name>.*?\S)\s*\((?P<inner>[^()]+)\)\s*$")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def split_condition(name: str) -> tuple[str, str | None]:
    """'Shock absorber lower nut (SVT Raptor)' -> ('Shock absorber lower nut', 'SVT Raptor').

    Short upper-case parentheticals such as ``(TPMS)`` are acronyms, not variants.
    """
    m = _PAREN_TAIL.match(name)
    if not m or _ACRONYM_RE.match(m.group("inner").strip()):
        return name, None
    return m.group("name").strip(), m.group("inner").strip()


def split_item_and_codes(item: str) -> tuple[str, str]:
    """Split a Material-table item cell into (name, part codes)."""
    m = PART_CODE_RE.search(item)
    if not m:
        return item.strip(), ""
    return item[: m.start()].strip(" -"), item[m.start():].strip()


def _base(chunk: Chunk, page: int | None = None) -> dict:
    return dict(section=chunk.section_label, page=page or chunk.page_start, chunk_id=chunk.id)


def _primary(quantities: list[Quantity], prefer: tuple[str, ...] = ()) -> tuple[Quantity | None, list[Quantity]]:
    if not quantities:
        return None, []
    pick = next((q for u in prefer for q in quantities if q.unit == u), quantities[0])
    return pick, [q for q in quantities if q is not pick]


# ---------------------------------------------------------------------------
# table parsers
# ---------------------------------------------------------------------------

def parse_torque_table(chunk: Chunk) -> list[SpecRecord]:
    units = [canon_unit(h) for h in chunk.header[1:]]
    note_letter, note_text = "", ""
    if chunk.footnote:
        m = re.match(r"^([a-z])\s+(.*)$", chunk.footnote.strip())
        if m:
            note_letter, note_text = m.group(1), m.group(2)
    out: list[SpecRecord] = []
    for row in chunk.rows:
        name, vals = row[0], row[1:]
        footnoted = False
        if note_letter and (m := re.match(rf"^(.*\S)\s+{note_letter}$", name)):
            name, footnoted = m.group(1), True
        quantities = [Quantity(value=norm_number(v), unit=u) for v, u in zip(vals, units) if not is_dash(v)]
        component, condition = split_condition(name)
        prim, alts = _primary(quantities, prefer=("Nm",))
        out.append(
            SpecRecord(
                component=component, spec_type="Torque",
                value=prim.value if prim else None, unit=prim.unit if prim else None,
                alt_values=alts, condition=condition,
                notes=(note_text if footnoted else None) or (None if prim else "No torque value tabulated"),
                source="table", verified=True, evidence=" | ".join(row), **_base(chunk),
            )
        )
    return out


def parse_material_table(chunk: Chunk) -> list[SpecRecord]:
    hdr = [h.lower() for h in chunk.header]
    spec_i = hdr.index("specification") if "specification" in hdr else None
    cap_i = next((i for i, h in enumerate(hdr) if "capacity" in h), None)
    out: list[SpecRecord] = []
    for row in chunk.rows:
        row = row + [""] * (len(hdr) - len(row))
        name, codes = split_item_and_codes(row[0])
        if not name:
            continue
        common = dict(component=name, source="table", verified=True, evidence=" | ".join(row), **_base(chunk))
        if codes:
            out.append(SpecRecord(spec_type="Part Number", value=codes, **common))
        if spec_i is not None and not is_dash(row[spec_i]):
            out.append(SpecRecord(spec_type="Fluid Specification", value=row[spec_i], **common))
        if cap_i is not None and not is_dash(row[cap_i]):
            quantities, leftover = parse_quantities(row[cap_i])
            prim, alts = _primary(quantities)
            out.append(
                SpecRecord(
                    spec_type="Fluid Capacity",
                    value=prim.value if prim else row[cap_i], unit=prim.unit if prim else None,
                    alt_values=alts, notes=leftover or None, **common,
                )
            )
    return out


def parse_general_table(chunk: Chunk) -> list[SpecRecord]:
    out: list[SpecRecord] = []
    category = ""
    for row in chunk.rows:
        name = row[0]
        cell = row[1] if len(row) > 1 else ""
        if not cell:                      # category row such as "Brake Pads"
            category = name
            continue
        quantities, leftover = parse_quantities(cell)
        prim, alts = _primary(quantities)
        notes = "; ".join(x for x in (category if category.lower() not in name.lower() else "", leftover if prim else "") if x)
        out.append(
            SpecRecord(
                component=name, spec_type="Specification",
                value=prim.value if prim else cell, unit=prim.unit if prim else None,
                alt_values=alts, notes=notes or None,
                source="table", verified=True, evidence=f"{name} | {cell}", **_base(chunk),
            )
        )
    return out


def parse_parts_table(chunk: Chunk) -> list[SpecRecord]:
    hdr = [h.lower() for h in chunk.header]
    pn_i = hdr.index("part number")
    desc_i = next((i for i, h in enumerate(hdr) if h == "description"), None)
    item_i = next((i for i, h in enumerate(hdr) if h == "item"), None)
    context = chunk.heading_path[0] if chunk.heading_path else ""
    out: list[SpecRecord] = []
    for row in chunk.rows:
        row = row + [""] * (len(hdr) - len(row))
        if is_dash(row[pn_i]):
            continue
        desc = row[desc_i] if desc_i is not None else ""
        item = row[item_i] if item_i is not None else ""
        notes = "; ".join(x for x in (context, f"callout {item}" if item else "") if x)
        out.append(
            SpecRecord(
                component=desc or "Unnamed part", spec_type="Part Number", value=row[pn_i],
                notes=notes or None, source="table", verified=True, evidence=" | ".join(row), **_base(chunk),
            )
        )
    return out


TABLE_PARSERS = {
    "torque_table": parse_torque_table,
    "material_table": parse_material_table,
    "general_table": parse_general_table,
    "parts_table": parse_parts_table,
}


# ---------------------------------------------------------------------------
# inline torque statements in procedure text
# ---------------------------------------------------------------------------

class RuleExtractor:
    """Runs the table parsers and the inline-torque parser over chunks."""

    def __init__(self, all_chunks: Iterable[Chunk] = ()):
        # (section_id, Nm value) -> [(component, condition)] from torque tables
        self.torque_index: dict[tuple[str, str], list[tuple[str, str | None]]] = defaultdict(list)
        for c in all_chunks:
            if c.chunk_type == "torque_table":
                for r in parse_torque_table(c):
                    if r.value and r.unit == "Nm":
                        self.torque_index[(c.section_id, r.value)].append((r.component, r.condition))

    # -- public -----------------------------------------------------------
    def extract(self, chunk: Chunk) -> list[SpecRecord]:
        parser = TABLE_PARSERS.get(chunk.chunk_type)
        if parser:
            return parser(chunk)
        return self._inline_torque(chunk)

    # -- inline torque ------------------------------------------------------
    def _inline_torque(self, chunk: Chunk) -> list[SpecRecord]:
        out: list[SpecRecord] = []
        lines = chunk.body.split("\n")
        heading = chunk.heading_path[0] if chunk.heading_path else chunk.section_title
        for i, line in enumerate(lines):
            for m in INLINE_TORQUE_RE.finditer(line):
                context = " ".join(lines[max(0, i - 2): i + 1])
                nm = norm_number(m.group("val"))
                alt = Quantity(value=norm_number(m.group("alt")), unit=canon_unit(m.group("altu")))
                component, how = self._name_fastener(chunk, nm, context, heading)
                out.append(
                    SpecRecord(
                        component=component, spec_type="Torque", value=nm, unit="Nm", alt_values=[alt],
                        notes=f"Procedure text ({chunk.heading_path[-1] if chunk.heading_path else 'procedure'}); {how}",
                        source="text", verified=True, evidence=m.group(0),
                        **_base(chunk, chunk.page_of(m.group(0))),
                    )
                )
        return out

    def _name_fastener(self, chunk: Chunk, nm: str, context: str, heading: str) -> tuple[str, str]:
        # 1) a specific fastener named in the sentence ("tighten the lower brake caliper guide pin bolt")
        objs = [m.group("obj").strip() for m in FASTENER_RE.finditer(context)]
        obj = objs[-1] if objs else ""
        if len(obj.split()) >= 3:
            return obj[0].upper() + obj[1:], "component named in procedure text"
        # 2) unique torque-table row with the same Nm value in this section
        matches = self.torque_index.get((chunk.section_id, nm), [])
        names = sorted({n for n, _ in matches})
        if len(names) == 1:
            return names[0], "component matched by value to this section's torque table"
        if names:
            return f"{heading}: " + " / ".join(names), "value matches several rows of the section's torque table"
        # 3) fall back to the heading
        label = f"{obj[0].upper() + obj[1:]} ({heading})" if obj else f"{heading} fastener"
        return label, "fastener not named in text (shown in a figure)"
