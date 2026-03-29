import hashlib
import os
import pandas as pd
import pytest
from src.data.folds import make_split, load_split


def _make_df(n: int = 100) -> pd.DataFrame:
    rows = []
    for i in range(n):
        claim = f"Claim number {i} about some financial topic"
        rows.append({
            "claim_text": claim,
            "claim_hash": hashlib.sha1(claim.encode()).hexdigest(),
            "label":      i % 2,
        })
    return pd.DataFrame(rows)


def test_split_sizes(tmp_path):
    df = _make_df(100)
    folds_path = str(tmp_path / "folds.json")
    df = make_split(df, val_size=0.15, test_size=0.15, seed=42, output_path=folds_path)

    n_train = (df["split"] == "train").sum()
    n_val   = (df["split"] == "val").sum()
    n_test  = (df["split"] == "test").sum()

    assert n_train + n_val + n_test == len(df)
    assert n_val > 0 and n_test > 0 and n_train > 0


def test_no_overlap_between_splits(tmp_path):
    df = _make_df(100)
    folds_path = str(tmp_path / "folds.json")
    df = make_split(df, val_size=0.15, test_size=0.15, seed=42, output_path=folds_path)

    train_hashes = set(df.loc[df["split"] == "train", "claim_hash"])
    val_hashes   = set(df.loc[df["split"] == "val",   "claim_hash"])
    test_hashes  = set(df.loc[df["split"] == "test",  "claim_hash"])

    assert not (train_hashes & val_hashes),  "Train/val overlap"
    assert not (train_hashes & test_hashes), "Train/test overlap"
    assert not (val_hashes   & test_hashes), "Val/test overlap"


def test_stratified_label_balance(tmp_path):
    df = _make_df(100)
    folds_path = str(tmp_path / "folds.json")
    df = make_split(df, val_size=0.15, test_size=0.15, seed=42, output_path=folds_path)

    for split in ["train", "val", "test"]:
        rate = df.loc[df["split"] == split, "label"].mean()
        # Expect label=1 rate close to 0.5 (synthetic data is perfectly balanced)
        assert 0.3 <= rate <= 0.7, f"{split} label rate={rate:.2f} is unbalanced"


def test_folds_json_written(tmp_path):
    df = _make_df(100)
    folds_path = str(tmp_path / "folds.json")
    make_split(df, output_path=folds_path)
    assert os.path.isfile(folds_path)


def test_load_split_matches(tmp_path):
    df = _make_df(100)
    folds_path = str(tmp_path / "folds.json")
    df_split = make_split(df, output_path=folds_path)
    df_loaded = load_split(df, folds_path=folds_path)
    assert (df_split["split"] == df_loaded["split"]).all()
