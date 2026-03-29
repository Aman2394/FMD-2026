import pandas as pd


def dedup(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    before = len(df)
    df = df.drop_duplicates(subset="claim_hash", keep="first")
    print(f"Dedup: {before} → {len(df)} rows ({before - len(df)} removed)")
    return df.reset_index(drop=True)
