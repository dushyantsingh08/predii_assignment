"""Quantity parsing for strings such as ``32 mm (1.259 in)`` or ``2.84L (6.0 pt)``."""
from __future__ import annotations

import re

from .schema import Quantity

_NUM = r"\d[\d,]*(?:\.\d+)?"
_RANGE = rf"{_NUM}(?:\s*(?:-|\u2013|\u2014|to)\s*{_NUM})?"

_UNIT_ALTERNATION = (
    r"N[\u00b7.\- ]?m|lb[\- ]?ft|lb[\- ]?in|mm|cm|kg|kPa|psi|ml|fl\.?\s?oz|oz|pt|qt|L|in|"
    r"\u00b0|degrees?|ohms?|%"
)
QTY_RE = re.compile(
    rf"(?P<value>{_RANGE})\s*(?P<unit>{_UNIT_ALTERNATION})(?![A-Za-z])",
    re.IGNORECASE,
)

_CANON = {
    "nm": "Nm", "n.m": "Nm", "n\u00b7m": "Nm", "n-m": "Nm", "n m": "Nm",
    "lbft": "lb-ft", "lb-ft": "lb-ft", "lb ft": "lb-ft",
    "lbin": "lb-in", "lb-in": "lb-in", "lb in": "lb-in",
    "mm": "mm", "cm": "cm", "kg": "kg", "kpa": "kPa", "psi": "psi",
    "ml": "ml", "l": "L", "pt": "pt", "qt": "qt", "oz": "oz",
    "floz": "fl oz", "fl.oz": "fl oz", "fl oz": "fl oz", "fl. oz": "fl oz",
    "in": "in", "\u00b0": "\u00b0", "degree": "\u00b0", "degrees": "\u00b0",
    "ohm": "ohms", "ohms": "ohms", "%": "%",
}

TORQUE_UNITS = ("Nm", "lb-ft", "lb-in")
VOLUME_UNITS = ("L", "ml", "pt", "qt", "fl oz", "oz")


def canon_unit(unit: str) -> str:
    key = unit.strip().lower()
    return _CANON.get(key, _CANON.get(key.replace(" ", ""), unit.strip()))


def norm_number(value: str) -> str:
    """'1,420' -> '1420'; '0.254 - 0.965' -> '0.254-0.965'."""
    v = value.replace(",", "")
    v = re.sub(r"\s*(?:-|\u2013|\u2014|to)\s*", "-", v)
    return v.strip()


def parse_quantities(text: str) -> tuple[list[Quantity], str]:
    """Return ``(quantities, leftover_text)``.

    ``leftover_text`` is whatever remains after removing the quantities, e.g.
    ``"9.56 mm (0.38 in) below bottom of housing"`` -> ``"below bottom of housing"``.
    """
    quantities: list[Quantity] = []
    spans: list[tuple[int, int]] = []
    for m in QTY_RE.finditer(text):
        quantities.append(Quantity(value=norm_number(m.group("value")), unit=canon_unit(m.group("unit"))))
        spans.append(m.span())

    leftover = text
    for start, end in reversed(spans):
        leftover = leftover[:start] + leftover[end:]
    leftover = re.sub(r"\(\s*\)", "", leftover)
    leftover = re.sub(r"\s+", " ", leftover).strip(" ;,-\u2014")
    return quantities, leftover


def numbers_in(text: str) -> list[str]:
    """All numeric tokens in ``text`` with thousands separators removed."""
    return [t.replace(",", "") for t in re.findall(_NUM, text)]
