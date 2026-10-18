"""Output data model.

The first four fields (component, spec_type, value, unit) are exactly the
format requested in the assignment.  The remaining fields add provenance and
quality signals so every number can be traced back to a manual page.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

SPEC_TYPES = (
    "Torque",
    "Fluid Capacity",
    "Fluid Specification",
    "Part Number",
    "Specification",
)


def normalize_spec_type(raw: str | None) -> str:
    """Map free-form LLM labels onto the fixed vocabulary."""
    s = (raw or "").strip().lower()
    if "torque" in s:
        return "Torque"
    if "capacity" in s or "fill" in s or "volume" in s:
        return "Fluid Capacity"
    if "part" in s and ("number" in s or "no" in s or "code" in s):
        return "Part Number"
    if "fluid" in s or "lubricant" in s or "oil" in s or "material" in s:
        return "Fluid Specification"
    return "Specification"


class Quantity(BaseModel):
    value: str
    unit: Optional[str] = None


class SpecRecord(BaseModel):
    # ---- the four fields required by the assignment ------------------------
    component: str
    spec_type: str
    value: Optional[str] = None
    unit: Optional[str] = None
    # ---- extras ------------------------------------------------------------
    alt_values: list[Quantity] = Field(default_factory=list)   # same spec, other units
    condition: Optional[str] = None                           # variant: "SVT Raptor", "4WD" ...
    notes: Optional[str] = None
    section: Optional[str] = None                             # "206-03 Front Disc Brake"
    page: Optional[int] = None                                # 1-based PDF page
    source: Literal["table", "text", "llm"] = "table"
    verified: bool = False                                    # value found verbatim in source text
    evidence: Optional[str] = None
    chunk_id: Optional[str] = None
    also_found_in: list[str] = Field(default_factory=list)   # same fact printed in other sections/pages

    @field_validator("spec_type", mode="before")
    @classmethod
    def _norm_type(cls, v):
        return normalize_spec_type(v)

    # ------------------------------------------------------------------
    def key(self) -> tuple:
        """Identity used for de-duplication (ignores brand names and market suffixes)."""
        name = self.component.split(" / ")[0].lower()
        name = re.sub(r"\bmotorcraft\b|\((?:us|canada)\)", " ", name)
        return (
            re.sub(r"\s+", " ", name).strip(),
            self.spec_type,
            (self.value or "").lower(),
            (self.unit or "").lower(),
            (self.condition or "").lower(),
        )

    def to_assignment_dict(self) -> dict:
        """The minimal shape shown in the assignment brief."""
        return {
            "component": self.component,
            "spec_type": self.spec_type,
            "value": self.value,
            "unit": self.unit,
        }

    def to_flat_dict(self) -> dict:
        """One CSV-friendly row."""
        d = self.model_dump(exclude={"alt_values", "evidence", "chunk_id", "also_found_in"})
        d["alt_values"] = "; ".join(f"{q.value} {q.unit or ''}".strip() for q in self.alt_values)
        d["also_found_in"] = "; ".join(self.also_found_in)
        return d


CSV_COLUMNS = [
    "component", "spec_type", "value", "unit", "alt_values", "condition",
    "notes", "section", "page", "also_found_in", "source", "verified",
]
