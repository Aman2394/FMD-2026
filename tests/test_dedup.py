import pytest
from src.data.dedup import dedup


def _make_records(claims: list[tuple[str, int]]) -> list[dict]:
    import hashlib
    return [
        {
            "prompt_text": c,
            "claim_text":  c,
            "label":       lbl,
            "claim_hash":  hashlib.sha1(c.encode()).hexdigest(),
        }
        for c, lbl in claims
    ]


def test_no_duplicates_unchanged():
    records = _make_records([("claim A", 0), ("claim B", 1)])
    df = dedup(records)
    assert len(df) == 2


def test_exact_duplicates_removed():
    records = _make_records([("claim A", 0), ("claim A", 0), ("claim B", 1)])
    df = dedup(records)
    assert len(df) == 2
    assert list(df["claim_text"]) == ["claim A", "claim B"]


def test_keeps_first_occurrence():
    records = _make_records([("dup", 0), ("dup", 1)])
    df = dedup(records)
    assert len(df) == 1
    assert df.iloc[0]["label"] == 0   # first kept


def test_reset_index():
    records = _make_records([("a", 0), ("a", 0), ("b", 1)])
    df = dedup(records)
    assert list(df.index) == list(range(len(df)))
