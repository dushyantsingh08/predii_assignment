from conftest import make_chunk
from specx.config import Settings
from specx.query import analyze_query, overlap_score
from specx.retriever import HybridRetriever, tokenize


def test_intents():
    assert analyze_query("Torque for brake caliper bolts").intent == "torque"
    assert analyze_query("How much brake fluid does it take?").intent == "capacity"
    assert analyze_query("Part number for shock absorber lower bolt").intent == "part_number"
    assert analyze_query("brake fluid specification").intent == "fluid_spec"
    assert analyze_query("front brake disc minimum thickness").intent == "specification"


def test_terms_drop_intent_and_stop_words_and_stem():
    assert analyze_query("Torque for brake caliper bolts").terms == ["brake", "caliper", "bolt"]


def test_overlap_score():
    info = analyze_query("front brake caliper bolts")
    assert overlap_score(info, "Brake caliper guide pin bolts", "Front Disc Brake") == 1.0
    assert overlap_score(info, "Brake caliper guide pin bolts", "Rear Disc Brake") == 0.75


def test_tokenizer_keeps_codes_and_pieces():
    toks = tokenize("PM-4-A WSS-M6C62-A bolts")
    assert "pm-4-a" in toks and "wss-m6c62-a" in toks and "bolt" in toks and "wss" in toks


def _corpus():
    return [
        make_chunk("a", "torque_table", ["Description", "Nm"], [], body="Brake caliper guide pin bolts 37 Nm torque specifications"),
        make_chunk("b", "material_table", ["Item"], [], body="Brake fluid WSS-M6C62-A 1,420 ml fill capacity", page=599),
        make_chunk("c", "text", [], [], doc_type="DESCRIPTION AND OPERATION", body="The brake system consists of pads, discs and calipers"),
        make_chunk("d", "text", [], [], doc_type="REMOVAL AND INSTALLATION", body="Remove the wheel and tire then the shock absorber"),
    ]


def test_hybrid_ranks_exact_code_and_intent_first(hash_embedder):
    chunks = _corpus()
    r = HybridRetriever(chunks, hash_embedder.encode_passages([c.text for c in chunks]), hash_embedder, Settings())
    assert r.search("Torque for brake caliper bolts", k=2)[0].chunk.id == "a"
    assert r.search("WSS-M6C62-A", k=1)[0].chunk.id == "b"          # exact code: BM25 carries it
    hit = r.search("brake fluid capacity", k=1)[0]
    assert hit.chunk.id == "b" and hit.boost > 1.0                 # intent boost applied


def test_position_conflict_rejects_opposites_only():
    from specx.query import position_conflict
    lower = analyze_query("lower ball joint nut torque")
    assert position_conflict(lower, "Upper ball joint nut", "Front Suspension")
    assert not position_conflict(lower, "Lower ball joint nut", "Front Suspension")
    front = analyze_query("Front axle lubricant capacity")
    assert position_conflict(front, "Premium Rear Axle Lubricant", "Rear Drive Axle")
    assert not position_conflict(front, "Premium Rear Axle Lubricant", "Front Drive Axle")   # mentions both: keep
    assert not position_conflict(analyze_query("brake caliper bolts"), "Rear caliper bolt")  # no position asked
