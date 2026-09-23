"""Tests for the parquet sampler, the data ledger and the teacher-output parser."""

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from danalm.data import ledger
from danalm.data.hub import sample_parquet, text_stats
from danalm.teacher import client
from danalm.teacher.checkpoint import AnswerLog
from danalm.teacher.client import chat_with_retries, parse_lines
from danalm.teacher.server import assert_teacher_stopped


# ---------------------------------------------------------------- parquet sampling
@pytest.fixture
def parquet_files(tmp_path):
    """Two local parquet files, 5 row groups of 10 rows each."""
    paths = []
    for f in range(2):
        rows = [f"file{f} doc{i} " + "x" * 10 for i in range(50)]
        path = tmp_path / f"part{f}.parquet"
        pq.write_table(pa.table({"text": rows, "extra": list(range(50))}), path, row_group_size=10)
        paths.append(str(path))
    return paths


def sample(paths: list[str], target_chars: int, seed: int) -> tuple[list[str], list]:
    """Consume the streaming sampler into (texts, row groups read)."""
    units: list = []
    texts = list(
        sample_parquet(fsspec.filesystem("file"), paths, ["text"], target_chars, seed, units)
    )
    return texts, units


def test_sampling_stops_at_the_character_target(parquet_files):
    texts, units = sample(parquet_files, target_chars=500, seed=0)
    assert 500 <= sum(map(len, texts)) < 500 + 30
    assert len(units) == len({tuple(u) for u in units})


def test_sampling_is_seeded(parquet_files):
    first = sample(parquet_files, 300, seed=1)[0]
    assert sample(parquet_files, 300, seed=1)[0] == first
    assert sample(parquet_files, 300, seed=2)[0] != first


def test_target_zero_reads_everything(parquet_files):
    texts, units = sample(parquet_files, 0, seed=0)
    assert (len(texts), len(units)) == (100, 10)


def test_text_stats_counts_utf8_bytes():
    assert text_stats(["دانة pearl"]) == {"docs": 1, "chars": 10, "utf8_bytes": 14, "words": 2}


# ---------------------------------------------------------------- ledger
def test_ledger_merges_by_id_and_renders_totals(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger.upsert(path, [{"id": "a", "docs_raw": 10, "licence": "CC-BY-4.0"}])
    ledger.upsert(path, [{"id": "b", "tokens": {"tok": 50}}])
    entries = ledger.upsert(path, [{"id": "a", "docs_kept": 9, "tokens": {"tok": 90}}])
    assert [e["id"] for e in entries] == ["a", "b"]
    assert entries[0] == {
        "id": "a",
        "docs_raw": 10,
        "licence": "CC-BY-4.0",
        "docs_kept": 9,
        "tokens": {"tok": 90},
    }
    assert ledger.read(path) == entries
    table = ledger.render_table(entries)
    assert "10 → 9" in table
    assert "140 tokens (tok)" in table


def test_ledger_markdown_is_replaced_between_markers(tmp_path):
    md = tmp_path / "LEDGER.md"
    md.write_text(f"# Ledger\n\n{ledger.START}\nold\n{ledger.END}\n\n## Other\n", encoding="utf-8")
    ledger.write_markdown(md, [{"id": "x", "source": "a|b"}])
    text = md.read_text(encoding="utf-8")
    assert "old" not in text and "a\\|b" in text and text.endswith("## Other\n")


# ---------------------------------------------------------------- teacher
def test_parse_lines_strips_list_markers_but_keeps_arabizi_and_ranges():
    answer = '1. "ابي اعرف رصيدي"\n2) 3andi mushkila\n- 2-3 days late\n\n• كرتي blocked\n١. شكرا'
    assert parse_lines(answer) == [
        "ابي اعرف رصيدي",
        "3andi mushkila",
        "2-3 days late",
        "كرتي blocked",
        "شكرا",
    ]


def test_assert_teacher_stopped_passes_when_nothing_listens():
    assert_teacher_stopped("127.0.0.1", 1)  # port 1: nothing listens there


def test_answer_log_resumes_and_drops_a_torn_tail(tmp_path):
    path, keys = tmp_path / "answers.jsonl", ["a", "b", "c", "d"]
    log = AnswerLog(path, keys)
    log.add(0, "a", answer="x")
    log.add(1, "b", answer="ي")
    log.close()
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"i": 2, "key": "c", "ans' + "\0" * 8)  # a crash in the middle of a write
    log = AnswerLog(path, keys)
    assert sorted(log.done) == [0, 1] and log.done[1]["answer"] == "ي"
    log.add(3, "d", answer="z")
    log.close()
    assert sorted(AnswerLog(path, keys).done) == [0, 1, 3]


def test_answer_log_refuses_a_different_plan(tmp_path):
    path = tmp_path / "answers.jsonl"
    log = AnswerLog(path, ["a", "b"])
    log.add(1, "b", answer="x")
    log.close()
    with pytest.raises(ValueError, match="does not match"):
        AnswerLog(path, ["a", "other"])


def test_chat_with_retries_retries_then_gives_up(monkeypatch):
    calls = []

    def flaky(*args):
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionRefusedError("server busy")
        return "answer", {}

    monkeypatch.setattr(client, "chat", flaky)
    assert chat_with_retries("url", [], {}, 1, retries=2, backoff_s=0) == ("answer", {})
    calls.clear()
    with pytest.raises(ConnectionRefusedError):
        chat_with_retries("url", [], {}, 1, retries=1, backoff_s=0)
    assert len(calls) == 2
