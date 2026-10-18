"""Render a PDF page to PNG with an extracted spec highlighted (used by the UI)."""
from __future__ import annotations

from typing import Iterable, Optional

try:
    import pymupdf  # type: ignore
except ImportError:  # pragma: no cover
    import fitz as pymupdf  # type: ignore


def render_page(
    pdf_path: str, page_no: int, component: Optional[str] = None, values: Iterable[str] = (), zoom: float = 1.6
) -> bytes:
    """PNG bytes of 1-based ``page_no`` with the component and its values highlighted.

    Values are only highlighted on the same text row as the component when the component
    can be located, so a number like "35" is not marked everywhere on the page.
    """
    doc = pymupdf.open(pdf_path)
    try:
        page = doc[page_no - 1]
        rows: list = []
        if component:
            for rect in page.search_for(component.strip()[:80]):
                page.add_highlight_annot(rect).update()
                rows.append(rect)
        for val in values:
            val = (val or "").strip()
            if len(val) < 1:
                continue
            for rect in page.search_for(val):
                near = not rows or any(abs((rect.y0 + rect.y1) / 2 - (r.y0 + r.y1) / 2) < 8 for r in rows)
                if near:
                    page.add_highlight_annot(rect).update()
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        return pix.tobytes("png")
    finally:
        doc.close()
