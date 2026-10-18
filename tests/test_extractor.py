import json

import pytest
from conftest import make_chunk
from specx.extractor import ExtractionError, extract_with_llm, is_grounded, parse_json_object
from specx.llm import CallableLLM

TABLE = ("| Description | Nm | lb-ft | lb-in |\n|---|---|---|---|\n| Brake caliper guide pin bolts | 37 | 27 | \u2014 |\n"
         "| Brake caliper flow bolt | 35 | 26 | \u2014 |")


def chunks():
    return [make_chunk("c1", "torque_table", [], [], body=TABLE, text="[206-03 Front Disc Brake | p.636]\n" + TABLE)]


def test_parse_json_variants():
    assert parse_json_object('{"results": []}') == {"results": []}
    assert parse_json_object('```json\n{"results": [1]}\n```') == {"results": [1]}
    assert parse_json_object('Sure! {"results": []} hope that helps') == {"results": []}
    with pytest.raises(ExtractionError):
        parse_json_object("no json here")


def test_grounding_is_boundary_aware():
    t = "[x | p.599]\n| bolt | 37 | 27 |\nWSS-M6C65-\nA1 and 1,420 ml 1.5-3.0 Nm"
    assert is_grounded("37", t) and is_grounded("1420", t) and is_grounded("1.5-3.0", t)
    assert is_grounded("WSS-M6C65-A1", t)
    assert not is_grounded("99", t) and not is_grounded("599", t) and not is_grounded(None, t)


def llm_returning(payload):
    return CallableLLM(lambda s, u: payload if isinstance(payload, str) else json.dumps(payload), "mock")


def test_extraction_validates_and_flags_hallucination():
    payload = {"results": [
        {"component": "Brake caliper guide pin bolts", "spec_type": "torque", "value": "37", "unit": "Nm",
         "alt_values": [{"value": "27", "unit": "lb-ft"}], "source": "C1", "evidence": "37 | 27"},
        {"component": "Brake caliper flow bolt", "spec_type": "Torque", "value": "99", "unit": "Nm", "source": "C1"},
        {"component": "", "value": "1"},                     # malformed: ignored
    ]}
    recs = extract_with_llm(llm_returning(payload), "q", chunks())
    assert len(recs) == 2
    good, bad = recs
    assert good.spec_type == "Torque" and good.verified and good.page == 636 and good.alt_values[0].unit == "lb-ft"
    assert not bad.verified                                   # 99 is not in the excerpt


def test_extraction_normalises_numbers_and_survives_one_bad_reply():
    replies = iter(["oops not json", json.dumps({"results": [
        {"component": "Brake fluid", "spec_type": "capacity", "value": "1,420", "unit": "ml", "source": "C1"}]})])
    table = "| Brake fluid | 1,420 ml (3 pt) |"
    ch = [make_chunk("c1", "material_table", [], [], body=table, text=table)]
    recs = extract_with_llm(CallableLLM(lambda s, u: next(replies)), "q", ch)
    assert recs[0].value == "1420" and recs[0].spec_type == "Fluid Capacity" and recs[0].verified


def test_prompt_contains_breadcrumb_and_handles():
    seen = {}
    def fn(system, user):
        seen["user"] = user
        return '{"results": []}'
    extract_with_llm(CallableLLM(fn), "Torque for X", chunks())
    assert "[C1]" in seen["user"] and "206-03 Front Disc Brake" in seen["user"] and "Question: Torque for X" in seen["user"]
