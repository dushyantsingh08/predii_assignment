"""Query understanding: what kind of spec is being asked for, and which words matter.

Intent drives two things:

* **retrieval** - chunks of the matching table type get a boost
  (a torque question should favour ``torque_table`` chunks);
* **filtering** - the rule-based extractor only keeps records of the matching
  ``spec_type``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .cleaning import stem

# first matching pattern wins
INTENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("torque", re.compile(r"\b(torque[sd]?|tighten\w*|n\.?m|lb[- ]?ft|lb[- ]?in|foot[- ]?pounds?|ft[- ]?lbs?)\b", re.I)),
    ("part_number", re.compile(r"\b(part\s*(number|no\.?|#|code)s?|p/n|pn|motorcraft (part|code))\b", re.I)),
    ("capacity", re.compile(r"\b(capacit\w*|how (much|many)|fill(ing)? (volume|amount|quantity)|volume|litres?|liters?|quarts?|refill|ml)\b", re.I)),
    ("fluid_spec", re.compile(
        r"\b(fluid|oil|lubricant|grease|sealant|coolant)\b.*\b(spec\w*|type|grade|standard|meet\w*|viscosity)\b|"
        r"\b(spec\w*|type|grade|standard|viscosity)\b.*\b(fluid|oil|lubricant|grease|sealant|coolant)\b", re.I)),
]

INTENT_SPEC_TYPES: dict[str, set[str]] = {
    "torque": {"Torque"},
    "part_number": {"Part Number"},
    "capacity": {"Fluid Capacity"},
    "fluid_spec": {"Fluid Specification", "Part Number"},
    "specification": set(),                       # no restriction
}
INTENT_CHUNK_TYPES: dict[str, set[str]] = {
    "torque": {"torque_table"},
    "part_number": {"parts_table", "material_table"},
    "capacity": {"material_table"},
    "fluid_spec": {"material_table"},
    "specification": {"general_table"},
}
INTENT_EXPANSION = {
    "torque": "torque specifications nm lb-ft lb-in tighten",
    "capacity": "fill capacity specification material",
    "part_number": "part number",
    "fluid_spec": "specification material fluid",
    "specification": "general specifications",
}

_STOP = {
    "the", "a", "an", "for", "of", "to", "in", "on", "at", "is", "are", "was", "what", "whats", "what's", "which",
    "how", "much", "many", "does", "do", "give", "me", "find", "show", "get", "need", "required", "recommended",
    "and", "or", "with", "value", "values", "spec", "specs", "specification", "specifications", "f-150", "f150",
    "ford", "truck", "vehicle", "please", "tell", "about", "should", "be", "i", "my", "number", "numbers",
}
_INTENT_WORDS = {
    "torque", "torques", "tighten", "tightening", "nm", "capacity", "capacities", "fill", "part", "no", "code",
    "codes", "type", "grade", "lb-ft", "lb-in", "litres", "liters", "volume",
}
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-/.]*")


@dataclass
class QueryInfo:
    text: str
    intent: str = "specification"
    spec_types: set[str] = field(default_factory=set)
    terms: list[str] = field(default_factory=list)      # stemmed content words

    @property
    def chunk_types(self) -> set[str]:
        return INTENT_CHUNK_TYPES[self.intent]

    @property
    def expansion(self) -> str:
        return INTENT_EXPANSION[self.intent]


def analyze_query(text: str) -> QueryInfo:
    intent = "specification"
    for name, pat in INTENT_PATTERNS:
        if pat.search(text):
            intent = name
            break
    toks = [t.lower().strip(".-/") for t in _TOKEN.findall(text)]
    terms = [stem(t) for t in toks if t and t not in _STOP and t not in _INTENT_WORDS]
    return QueryInfo(text=text, intent=intent, spec_types=set(INTENT_SPEC_TYPES[intent]), terms=terms)


def haystack_tokens(*parts: str | None) -> set[str]:
    """Stemmed token set used for term-overlap matching."""
    out: set[str] = set()
    for p in parts:
        for t in _TOKEN.findall((p or "").lower()):
            out.add(stem(t.strip(".-/")))
            out.update(stem(x) for x in re.split(r"[-/]", t) if x)
    return out


def overlap_score(info: QueryInfo, *haystack_parts: str | None) -> float:
    """Fraction of the query's content terms found in the record text (0..1)."""
    if not info.terms:
        return 1.0
    hay = haystack_tokens(*haystack_parts)
    return sum(1 for t in info.terms if t in hay) / len(info.terms)


POSITION_AXES = [{"front", "rear"}, {"upper", "lower"}, {"left", "right"}, {"inner", "outer"}]


def position_conflict(info: QueryInfo, *haystack_parts: str | None) -> bool:
    """True if the record names the *opposite* position to the one asked for.

    "lower ball joint nut" must not return the *upper* ball joint nut, and "front brake disc"
    must not return a record that only talks about the rear.  A record that mentions both
    (e.g. a product called "Rear Axle Lubricant" in the Front Drive Axle section) is kept.
    """
    hay = haystack_tokens(*haystack_parts)
    asked = set(info.terms)
    for axis in POSITION_AXES:
        want = axis & asked
        if len(want) == 1:
            (w,) = want
            other = next(iter(axis - {w}))
            if other in hay and w not in hay:
                return True
    return False
