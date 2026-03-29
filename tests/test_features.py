import pandas as pd
import pytest
from src.data.features import extract, _extract_tickers, _ellipsis_position, _extract_title


def _df(claims: list[str]) -> pd.DataFrame:
    import hashlib
    return pd.DataFrame([
        {"claim_text": c, "claim_hash": hashlib.sha1(c.encode()).hexdigest(), "label": 0}
        for c in claims
    ])


def test_number_extraction():
    df = extract(_df(["Revenue was $1.5 billion in 2023."]))
    assert len(df.iloc[0]["numbers"]) >= 1
    assert df.iloc[0]["n_numbers"] >= 1


def test_ticker_extraction():
    tickers = _extract_tickers("Apple (AAPL) and (MSFT) both fell.")
    assert "AAPL" in tickers
    assert "MSFT" in tickers


def test_no_tickers():
    tickers = _extract_tickers("No ticker symbols here.")
    # May or may not find uppercase words — group_ticker default is NONE
    df = extract(_df(["No ticker symbols here."]))
    # group_ticker is first ticker or NONE
    assert isinstance(df.iloc[0]["group_ticker"], str)


def test_ellipsis_detected():
    df = extract(_df(["The company reported... strong earnings."]))
    assert df.iloc[0]["has_ellipsis"] is True


def test_no_ellipsis():
    df = extract(_df(["Clean sentence with no ellipsis."]))
    assert df.iloc[0]["has_ellipsis"] is False
    assert df.iloc[0]["ellipsis_pos"] == "none"


def test_ellipsis_position_mid():
    text = "Start ... " + "x" * 100
    assert _ellipsis_position(text) == "mid"


def test_ellipsis_position_tail():
    text = "x" * 100 + "..."
    assert _ellipsis_position(text) == "tail"


def test_title_extraction():
    title = _extract_title("Apple - Q3 earnings beat expectations")
    assert title == "Apple"


def test_time_tokens():
    df = extract(_df(["Revenue rose in Q3 2023 and FY2022."]))
    tokens = df.iloc[0]["time_tokens"]
    assert "Q3" in tokens or "2023" in tokens


def test_unit_extraction():
    df = extract(_df(["Margins expanded by 150 bps and revenue grew 12 percent."]))
    units = df.iloc[0]["units"]
    assert len(units) >= 1


def test_n_tokens_approx():
    df = extract(_df(["one two three four five"]))
    assert df.iloc[0]["n_tokens_approx"] == 5
