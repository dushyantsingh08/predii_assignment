"""Text cleaning and normalisation helpers.

The service manual is a browser print-out of an HTML workshop manual, so the
text layer contains a handful of recurring artefacts that hurt both retrieval
and LLM extraction:

* the Wingdings bullet glyph is extracted as a bare ``z``
* ``(R)``/``(TM)`` symbols and non-breaking spaces
* part / spec codes wrapped across lines inside table cells
  (``WSS-M6C65-`` newline ``A1``)
* ``—`` used as the "no value in this column" marker
"""
from __future__ import annotations

import re

_TRANSLATE = str.maketrans(
    {
        "\u00ae": "",    # (R)
        "\u2122": "",    # (TM)
        "\u00a0": " ",
        "\u2009": " ",
        "\u202f": " ",
        "\u200b": "",
        "\u2011": "-",   # non-breaking hyphen
    }
)
_WS = re.compile(r"[ \t]+")
_BULLET_START = re.compile(r"(^|\n)\s*z\s+(?=\S)")
_WRAPPED_HYPHEN = re.compile(r"-[ \t]*\n[ \t]*")

DASHES = {"", "\u2014", "\u2013", "-", "\u2012", "\u2015"}


def is_dash(value: str | None) -> bool:
    """True for the manual's "no value" placeholder cells."""
    return (value or "").strip() in DASHES


def clean_line(text: str) -> str:
    """Normalise one line of body text."""
    text = text.translate(_TRANSLATE)
    text = _WS.sub(" ", text).strip()
    text = re.sub(r"^z\s+(?=\S)", "- ", text)        # Wingdings bullet -> "-"
    return text


def normalize_cell(text: str | None) -> str:
    """Normalise a table cell into a single clean line."""
    if not text:
        return ""
    text = text.translate(_TRANSLATE)
    text = _BULLET_START.sub(r"\1- ", text)
    text = _WRAPPED_HYPHEN.sub("-", text)             # re-join wrapped codes
    text = text.replace("\n", " ")
    return _WS.sub(" ", text).strip()


def word_count(text: str) -> int:
    return len(text.split())


# --- light stemming used by BM25 and by the rule-based relevance filter -----

def stem(word: str) -> str:
    """Tiny plural stemmer: bolts->bolt, nuts->nut, bodies->body, axles->axle."""
    w = word.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("sses", "xes", "ches", "shes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w
