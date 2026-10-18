from specx.units import parse_quantities, norm_number, canon_unit


def q(text):
    quantities, left = parse_quantities(text)
    return [(x.value, x.unit) for x in quantities], left


def test_metric_and_imperial():
    assert q("32 mm (1.259 in)") == ([("32", "mm"), ("1.259", "in")], "")


def test_thousands_separator_and_pints():
    assert q("1,420 ml (3.0 pt)")[0] == [("1420", "ml"), ("3.0", "pt")]


def test_range():
    assert q("0.254-0.965 mm (0.010-0.038 in)")[0] == [("0.254-0.965", "mm"), ("0.010-0.038", "in")]


def test_torque_range_and_lb_in():
    assert q("1.5-3.0 Nm (13-26 lb-in)")[0] == [("1.5-3.0", "Nm"), ("13-26", "lb-in")]


def test_leftover_text_kept():
    vals, left = q("2.84L (6.0 pt) (to the bottom of the filler hole)")
    assert vals == [("2.84", "L"), ("6.0", "pt")]
    assert left == "(to the bottom of the filler hole)"


def test_no_quantity():
    assert q("See Safety Certification Label on driver door jamb")[0] == []


def test_helpers():
    assert norm_number("1,420") == "1420"
    assert norm_number("0.254 - 0.965") == "0.254-0.965"
    assert canon_unit("N·m") == "Nm" and canon_unit("lb ft") == "lb-ft"
