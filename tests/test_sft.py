"""Tests for the SFT data tools: dialect markers, teacher-answer parsers and the intent taxonomy."""

from pathlib import Path

from danalm.config import load_config
from danalm.data.dialect import msa_markers, non_gulf_markers
from danalm.data.intents import load_intents
from danalm.data.labelled import iter_labelled, map_label, sample_by_intent, spread_sample
from danalm.data.reply_checks import claims_done_action, reply_problem
from danalm.teacher.client import parse_json_objects, parse_labels, parse_numbered

REPO = Path(__file__).resolve().parents[1]


def test_non_gulf_markers_flag_clear_words_only():
    assert non_gulf_markers("انا عايز اعرف ليه الطلب مش واصل") == [
        "عايز(egyptian)",
        "ليه(egyptian)",
        "مش(egyptian)",
    ]
    assert non_gulf_markers("واش هناك مشكلة في الحساب ديالي") == [
        "واش(maghrebi)",
        "ديالي(maghrebi)",
    ]
    # Gulf text, including عشان and the future prefix ب, is not flagged
    assert non_gulf_markers("ابغي اعرف ليش الطلب مب واصل عشان بطلب مرة ثانية") == []
    # a marker only counts as a whole word: "مشكلة" contains "مش" but is fine
    assert non_gulf_markers("عندي مشكلة في البطاقة") == []


def test_parse_json_objects_handles_lines_fences_arrays_and_junk():
    answer = (
        '```json\n{"message": "hi", "reply": "hello"}\n'
        '1. {"message": "card blocked", "reply": "sorry"}\n'
        "not json at all\n"
        '{"message": "broken", "reply": \n```'
    )
    assert parse_json_objects(answer) == [
        {"message": "hi", "reply": "hello"},
        {"message": "card blocked", "reply": "sorry"},
    ]
    assert parse_json_objects('[{"message": "a", "reply": "b"}, 3]') == [
        {"message": "a", "reply": "b"}
    ]


def test_parse_labels_reads_numbered_lines():
    answer = "1: order_status\n2. refund_request\n3) other\nsome chatter\n4: Order Status"
    assert parse_labels(answer) == {1: "order_status", 2: "refund_request", 3: "other"}


def test_intent_taxonomy_is_complete_and_unique():
    cfg = load_config(REPO / "configs" / "sft" / "generate.yaml")
    intents = load_intents(str(REPO / cfg["sft"]["intents_file"]))
    names = [i["name"] for i in intents]
    assert len(names) == 21 and len(set(names)) == 21
    assert {"other", "handoff_to_human"} <= set(names)
    assert {i["domain"] for i in intents} == {"banking", "telecom", "delivery", "general"}
    assert all(i["description"].strip() for i in intents)


def test_dialect_markers_see_through_attached_conjunctions_and_sh_negation():
    assert non_gulf_markers("الطلب مكتوب delivered ومفيش شي") == ["مفيش(egyptian)"]
    assert non_gulf_markers("شو صار؟ ما وصلتش الفلوس") == ["ما...ش(negation)"]
    # Gulf negation, "ليش" and cash ending in ش are fine
    assert non_gulf_markers("ما وصلت الفلوس ليش؟ ما عندي كاش") == []
    assert non_gulf_markers("انا محتاج مساعدة") == []  # محتاج is Gulf too


def test_msa_markers_flag_formal_words_in_messages():
    assert msa_markers("لماذا ظهر fee على الحساب") == ["لماذا(msa)"]
    assert msa_markers("ابغي اعرف ليش") == []


def test_replies_claiming_done_actions_are_detected():
    assert claims_done_action("Your card has been blocked, a new one is on the way.")
    assert claims_done_action("We have frozen your card.")
    assert claims_done_action("تم تجميد البطاقة")
    assert not claims_done_action("Please freeze the card in the app, or call us and we will help.")
    assert not claims_done_action("سيتم التحقق من الطلب بعد ما ترسل رقم الطلب")


def test_parse_numbered_keeps_valid_numbers_only():
    lines = [
        '{"n": 2, "reply": "b"}',
        '{"n": 1, "reply": "a"}',
        '{"n": 9, "reply": "out of range"}',
        '{"reply": "no number"}',
        '{"n": 3, "reply": " "}',
        "not json",
    ]
    assert parse_numbered("\n".join(lines), 3, "reply") == {1: "a", 2: "b"}


def test_reply_problem_checks_language_claims_and_dialect():
    assert reply_problem("سنحول طلبك للفريق المختص.", ["ar"], gulf_filter=True) is None
    assert reply_problem("ok, thanks", ["ar"], gulf_filter=True) == "reply_lang_en"
    assert reply_problem("Your card has been blocked.", ["en"], gulf_filter=False) == (
        "reply_claims_action"
    )
    assert reply_problem("ليه الطلب ما وصل؟", ["ar"], gulf_filter=True) == "non_gulf_dialect"
    assert reply_problem("ليه الطلب ما وصل؟", ["ar"], gulf_filter=False) is None


def test_map_label_uses_wildcard_and_skips_excluded_prefixes():
    src = {"label_map": {"oos": "other", "lost_card": None, "*": "other"},
           "exclude_label_prefixes": ["takeaway_"]}  # fmt: skip
    assert map_label("oos", src) == "other"
    assert map_label("weather_query", src) == "other"  # "*" covers unlisted labels
    assert map_label("lost_card", src) is None  # mapped to null = skipped
    assert map_label("takeaway_order", src) is None
    assert map_label("x", {"label_map": {"y": "other"}, "exclude_label_prefixes": []}) is None


def test_spread_sample_takes_labels_in_turn_and_is_seeded():
    rows = [{"source_label": "big", "i": i} for i in range(50)]
    rows += [{"source_label": "small", "i": i} for i in range(3)]
    sample = spread_sample(rows, per_label=None, total=6, seed=1)
    assert sum(r["source_label"] == "small" for r in sample) == 3  # not crowded out
    assert sample == spread_sample(rows, per_label=None, total=6, seed=1)
    capped = spread_sample(rows, per_label=2, total=None, seed=1)
    assert len(capped) == 4


def test_iter_labelled_reads_a_csv_at_a_pinned_url(tmp_path):
    (tmp_path / "abc123.csv").write_text(
        "text,category\nmy card was declined,declined_card_payment\nhello,greeting\n",
        encoding="utf-8",
    )
    src = {"url": (tmp_path / "{revision}.csv").as_uri().replace("%7B", "{").replace("%7D", "}"),
           "revision": "abc123", "text_column": "text", "label_column": "category"}  # fmt: skip
    assert list(iter_labelled(src, fs=None)) == [
        ("my card was declined", "declined_card_payment"),
        ("hello", "greeting"),
    ]


def test_sample_by_intent_caps_each_intent():
    rows = [{"intent": "a", "source_label": f"l{i % 3}"} for i in range(10)]
    rows += [{"intent": "b", "source_label": "x"} for _ in range(2)]
    sample = sample_by_intent(rows, per_intent=4, seed=0)
    assert [r["intent"] for r in sample].count("a") == 4
    assert [r["intent"] for r in sample].count("b") == 2
    assert len({r["source_label"] for r in sample if r["intent"] == "a"}) == 3  # spread
