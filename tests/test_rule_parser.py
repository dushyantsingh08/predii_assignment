from conftest import make_chunk
from specx.rule_parser import (RuleExtractor, parse_general_table, parse_material_table, parse_parts_table,
                               parse_torque_table, split_condition, split_item_and_codes)


def torque_chunk(rows, header=("Description", "Nm", "lb-ft", "lb-in"), **kw):
    return make_chunk("t1", "torque_table", list(header), rows, **kw)


def test_torque_dash_columns_map_by_header():
    recs = parse_torque_table(torque_chunk([
        ["Brake caliper guide pin bolts", "37", "27", "—"],
        ["Brake disc shield bolts", "18", "—", "159"],
    ]))
    a, b = recs
    assert (a.value, a.unit, [(x.value, x.unit) for x in a.alt_values]) == ("37", "Nm", [("27", "lb-ft")])
    assert (b.value, b.unit, [(x.value, x.unit) for x in b.alt_values]) == ("18", "Nm", [("159", "lb-in")])


def test_torque_two_unit_header():
    recs = parse_torque_table(torque_chunk([["Valve stem screw", "1.5", "13"]], header=("Description", "Nm", "lb-in")))
    assert recs[0].alt_values[0].unit == "lb-in"


def test_footnote_row_has_no_value():
    c = torque_chunk([["Wheel nuts a", "—", "—", "—"]], footnote="a Refer to the procedure in this section.")
    r = parse_torque_table(c)[0]
    assert r.component == "Wheel nuts" and r.value is None
    assert "procedure" in r.notes


def test_condition_split_but_acronyms_kept():
    assert split_condition("Shock absorber lower nut (SVT Raptor)") == ("Shock absorber lower nut", "SVT Raptor")
    assert split_condition("Valve stem-to-Tire Pressure Monitoring System (TPMS) sensor screw")[1] is None
    assert split_condition("Skid plate bolts (SVT Raptor vehicles)")[1] == "SVT Raptor vehicles"


def test_part_code_split():
    name, codes = split_item_and_codes("Motorcraft SAE 75W-140 Synthetic Rear Axle Lubricant XY-75W140-QL (US); CXY-75W140-1L (Canada)")
    assert name.endswith("Rear Axle Lubricant") and codes.startswith("XY-75W140-QL")
    assert split_item_and_codes("SAE 80W-90 lubricant")[1] == ""      # viscosity grade is not a part code


def test_material_table_three_record_types():
    c = make_chunk("m1", "material_table", ["Item", "Specification", "Fill Capacity"], [
        ["High Performance DOT 3 Motor Vehicle Brake Fluid PM-1-C (US); CPM-1-C (Canada)", "WSS-M6C62-A or WSS-M6C65-A1", "1,420 ml (3 pt)"],
        ["Metal Brake Parts Cleaner PM-4-A or PM-4-B (US); CPM-4 (Canada)", "—", "—"],
    ])
    recs = parse_material_table(c)
    kinds = [(r.spec_type, r.value) for r in recs]
    assert ("Part Number", "PM-1-C (US); CPM-1-C (Canada)") in kinds
    assert ("Fluid Specification", "WSS-M6C62-A or WSS-M6C65-A1") in kinds
    cap = next(r for r in recs if r.spec_type == "Fluid Capacity")
    assert (cap.value, cap.unit, cap.alt_values[0].unit) == ("1420", "ml", "pt")
    assert sum(r.spec_type == "Fluid Capacity" for r in recs) == 1       # dash rows give no capacity


def test_general_table_categories_and_text_values():
    c = make_chunk("g1", "general_table", ["Item", "Specification"], [
        ["Brake Pads", ""],
        ["Brake pad minimum thickness", "3.0 mm (0.118 in)"],
        ["Tires", "See Safety Certification Label on driver door jamb"],
    ])
    a, b = parse_general_table(c)
    assert (a.value, a.unit, a.notes) == ("3.0", "mm", "Brake Pads")
    assert b.value.startswith("See Safety") and b.unit is None


def test_parts_table():
    c = make_chunk("p1", "parts_table", ["Item", "Part Number", "Description"],
                   [["1", "W506545", "Shock absorber lower bolt"], ["3", "18125", "Shock absorber"]],
                   heading_path=["Shock Absorber"])
    recs = parse_parts_table(c)
    assert [(r.component, r.value) for r in recs] == [("Shock absorber lower bolt", "W506545"), ("Shock absorber", "18125")]
    assert "Shock Absorber" in recs[0].notes


def test_inline_torque_matched_to_table_by_value():
    table = torque_chunk([["Brake caliper flow bolt", "35", "26", "—"], ["Brake caliper guide pin bolts", "37", "27", "—"]])
    proc = make_chunk("x1", "text", [], [], doc_type="REMOVAL AND INSTALLATION", heading_path=["Brake Caliper", "Removal"],
                      body="1. NOTE: Use 2 new copper washers.\nTo install, tighten to 35 Nm (26 lb-ft).")
    rec = RuleExtractor([table]).extract(proc)[0]
    assert (rec.component, rec.value, rec.unit, rec.source) == ("Brake caliper flow bolt", "35", "Nm", "text")


def test_inline_torque_names_specific_fastener():
    proc = make_chunk("x2", "text", [], [], doc_type="REMOVAL AND INSTALLATION", heading_path=["Brake Caliper"],
                      body="NOTE: Tighten the lower brake caliper guide pin bolt first.\nTo install, tighten to 37 Nm (27 lb-ft).")
    rec = RuleExtractor([]).extract(proc)[0]
    assert "guide pin bolt" in rec.component.lower()
