from specx.chunker import build_chunks
from specx.config import Settings
from specx.pdf_loader import Element, LogicalDoc, ManualData


def doc(elements, doc_type="SPECIFICATIONS"):
    return ManualData("x.pdf", 10, docs=[LogicalDoc("SE1", "206-03", "Front Disc Brake", doc_type, "", 1, 3, elements)])


def test_spec_tables_get_own_chunks_with_breadcrumb():
    els = [
        Element("heading", 1, text="Torque Specifications", level=2),
        Element("table", 1, text="", table_kind="torque", header=["Description", "Nm", "lb-ft", "lb-in"],
                rows=[["Brake caliper guide pin bolts", "37", "27", "\u2014"]]),
        Element("text", 1, text="a Refer to the procedure in this section."),
    ]
    chunks = build_chunks(doc(els), Settings())
    assert len(chunks) == 1 and chunks[0].chunk_type == "torque_table"
    assert chunks[0].text.startswith("[206-03 Front Disc Brake | Specifications |")
    assert chunks[0].footnote.startswith("a Refer") and "Refer to the procedure" in chunks[0].text
    assert chunks[0].has_torque


def test_long_tables_split_with_header_repeated():
    rows = [[f"Bolt {i}", str(i), "1", "\u2014"] for i in range(40)]
    els = [Element("table", 1, table_kind="torque", header=["Description", "Nm", "lb-ft", "lb-in"], rows=rows)]
    chunks = build_chunks(doc(els), Settings(table_max_rows=18))
    assert len(chunks) == 3
    assert all(c.text.count("| Description | Nm |") == 1 for c in chunks)
    assert sum(len(c.rows) for c in chunks) == 40


def test_procedures_split_at_component_headings_and_keep_path():
    words = " ".join(["word"] * 60)
    els = [
        Element("heading", 1, text="Brake Disc", level=1), Element("heading", 1, text="Removal", level=2),
        Element("text", 1, text=f"1. Remove the wheel. {words}"),
        Element("heading", 2, text="Brake Pads", level=1), Element("heading", 2, text="Removal", level=2),
        Element("text", 2, text=f"1. Remove the pads. To install, tighten to 37 Nm (27 lb-ft). {words}"),
    ]
    chunks = build_chunks(doc(els, "REMOVAL AND INSTALLATION"), Settings())
    assert [c.heading_path[0] for c in chunks] == ["Brake Disc", "Brake Pads"]
    assert chunks[1].has_torque and not chunks[0].has_torque
    assert chunks[1].page_start == 2


def test_oversized_paragraph_is_split():
    big = "\n".join(f"{i}. step text {' '.join(['w'] * 30)}" for i in range(1, 40))
    chunks = build_chunks(doc([Element("text", 1, text=big)], "REMOVAL AND INSTALLATION"), Settings(max_chunk_words=200))
    assert len(chunks) > 3 and max(len(c.body.split()) for c in chunks) < 330


def test_page_attribution_for_multi_page_chunk():
    els = [Element("text", 5, text="alpha " * 20), Element("text", 6, text="To install, tighten to 99 Nm (73 lb-ft). " + "beta " * 20)]
    c = build_chunks(doc(els, "REMOVAL AND INSTALLATION"), Settings())[0]
    assert c.page_of("99 Nm") == 6 and c.page_of("alpha") == 5
