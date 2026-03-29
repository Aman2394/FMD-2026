import re
import pandas as pd

NUMBER_RE   = re.compile(r"[+-]?\$?[\d,]+\.?\d*(?:\s*(?:million|billion|bn|mn|[KkMmBb]))?")
UNIT_RE     = re.compile(r"\b(\d+\.?\d*)\s*(%|percent|bps|basis points)\b", re.I)
TICKER_RE   = re.compile(
    r"\b([A-Z]{1,5})\b(?=\s*[\(\[]?\b(?:NASDAQ|NYSE|NYSE MKT)\b)?|"
    r"\(([A-Z]{1,5})\)"
)
TIME_RE     = re.compile(
    r"\b(20\d{2}|Q[1-4]|FY\d{2,4}|H[12]\s*20\d{2}|"
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\b"
)
def extract(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["numbers"]         = df["claim_text"].apply(lambda t: NUMBER_RE.findall(t))
    df["n_numbers"]       = df["numbers"].apply(len)
    df["units"]           = df["claim_text"].apply(lambda t: UNIT_RE.findall(t))
    df["tickers"]         = df["claim_text"].apply(_extract_tickers)
    df["n_tickers"]       = df["tickers"].apply(len)
    df["time_tokens"]     = df["claim_text"].apply(lambda t: TIME_RE.findall(t))
    df["group_ticker"]    = df["tickers"].apply(lambda t: t[0] if t else "NONE")
    df["n_tokens_approx"] = df["claim_text"].apply(lambda t: len(t.split()))
    return df


def _extract_tickers(text: str) -> list[str]:
    return list(dict.fromkeys(
        t for match in TICKER_RE.finditer(text)
        for t in match.groups() if t
    ))
