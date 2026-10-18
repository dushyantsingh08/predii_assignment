"""Runs only when the service manual PDF is available (SPECX_PDF or data/service-manual.pdf)."""
from specx.chunker import build_chunks
from specx.config import Settings
from specx.pdf_loader import load_manual
from specx.rule_parser import RuleExtractor


def test_front_disc_brake_spec_page(pdf_path):
    manual = load_manual(pdf_path, Settings(), pages=range(635, 638), progress=False)
    doc = manual.docs[0]
    assert (doc.section_id, doc.doc_type, doc.file_id) == ("206-03", "SPECIFICATIONS", "SE263001")
    assert (doc.page_start, doc.page_end) == (636, 637)              # page 2 stitched back in
    kinds = [e.table_kind for e in doc.elements if e.kind == "table"]
    assert kinds == ["material", "general", "torque"]
    # no print noise survives
    text = " ".join(e.text for e in doc.elements)
    assert "file:///" not in text and "Page 1 sur" not in text and "repair4less" not in text

    chunks = build_chunks(manual, Settings())
    torque = next(c for c in chunks if c.chunk_type == "torque_table")
    recs = {r.component: r for r in RuleExtractor(chunks).extract(torque)}
    guide = recs["Brake caliper guide pin bolts"]
    assert (guide.value, guide.unit, guide.alt_values[0].value, guide.page) == ("37", "Nm", "27", 636)
    shield = recs["Brake disc shield bolts"]                         # Nm + lb-in, lb-ft is a dash
    assert (shield.value, shield.alt_values[0].unit) == ("18", "lb-in")


def test_continuation_page_inherits_section_and_torque_header(pdf_path):
    manual = load_manual(pdf_path, Settings(), pages=range(448, 451), progress=False)   # pages 449-451
    torque = [e for d in manual.docs for e in d.elements if e.kind == "table" and e.table_kind == "torque"]
    assert len(torque) == 2 and torque[1].header == ["Description", "Nm", "lb-ft"]
    assert {d.section_id for d in manual.docs} == {"205-03"}
