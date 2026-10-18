from specx.cleaning import normalize_cell, clean_line, stem, is_dash
from specx.pdf_loader import _norm_table, classify_table, table_to_markdown


def test_wrapped_codes_are_rejoined():
    assert normalize_cell("WSS-M6C65-\nA1") == "WSS-M6C65-A1"
    assert normalize_cell("Motorcraft\u00ae SAE 75W-\n140 Synthetic") == "Motorcraft SAE 75W-140 Synthetic"


def test_bullet_glyph_and_symbols():
    assert clean_line("z To install, tighten") == "- To install, tighten"
    assert clean_line("Motorcraft\u00ae") == "Motorcraft"


def test_dash_detection_and_stem():
    assert is_dash("\u2014") and is_dash("") and not is_dash("0")
    assert stem("bolts") == "bolt" and stem("bodies") == "body" and stem("glass") == "glass"


def test_part_table_drops_blank_row_and_column():
    raw = [["", "", "", ""], ["Item", "Part Number", "Description", ""], ["1", "W506545", "Shock absorber lower bolt", ""]]
    rows = _norm_table(raw)
    assert rows[0] == ["Item", "Part Number", "Description"]
    kind, header, body = classify_table(rows, None)
    assert kind == "parts" and len(body) == 1


def test_classification():
    assert classify_table([["Description", "Nm", "lb-ft", "lb-in"], ["Bolt", "10", "7", "-"]], None)[0] == "torque"
    assert classify_table([["Item", "Specification", "Fill Capacity"], ["Fluid", "X", "1 L"]], None)[0] == "material"
    assert classify_table([["Item", "Specification"], ["Thing", "3 mm"]], None)[0] == "general"
    assert classify_table([["Test Step", "Result / Action to Take"], ["A", "B"]], None)[0] == "other"


def test_headerless_torque_continuation_inherits_header():
    hdr = ["Description", "Nm", "lb-ft"]
    kind, header, body = classify_table([["Differential bearing cap bolts", "105", "77"], ["Fill plug", "25", "18"]], hdr)
    assert kind == "torque" and header == hdr and len(body) == 2
    # a numeric shim table must NOT be mistaken for torque
    assert classify_table([["0.152", "0.006", "0.203"]], hdr)[0] == "other"


def test_markdown_render():
    md = table_to_markdown("torque", ["Description", "Nm"], [["Bolt", "10"]])
    assert md.splitlines() == ["| Description | Nm |", "|---|---|", "| Bolt | 10 |"]
