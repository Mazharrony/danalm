"""Tests for the SFT data tools: dialect markers, teacher-answer parsers and the intent taxonomy."""

from pathlib import Path

from danalm.config import load_config
from danalm.data.dialect import msa_markers, non_gulf_markers
from danalm.data.intents import load_intents
from danalm.data.reply_checks import claims_done_action
from danalm.teacher.client import parse_json_objects, parse_labels

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
