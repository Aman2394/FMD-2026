"""
Out-of-Distribution (OOD) evaluation splits.

Three OOD dimensions:
  1. ticker-out  — held-out ticker symbols not seen during training
  2. length      — long-tail claims (top/bottom quartile by token count)
  3. numeracy    — high-numeracy claims (≥ median number count)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.evaluation.metrics import compute_all


def ticker_out_split(df: pd.DataFrame,
                     held_out_fraction: float = 0.2,
                     seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold out claims whose primary ticker was not seen in the training portion.

    Returns (train_df, ood_df).
    """
    rng     = np.random.default_rng(seed)
    tickers = df["group_ticker"].unique()
    n_held  = max(1, int(len(tickers) * held_out_fraction))
    ood_tickers = set(rng.choice(tickers, size=n_held, replace=False))

    ood_df   = df[df["group_ticker"].isin(ood_tickers)].copy()
    train_df = df[~df["group_ticker"].isin(ood_tickers)].copy()
    return train_df, ood_df


def length_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (short_df, long_df) — bottom and top quartile by token count.
    """
    q25 = df["n_tokens_approx"].quantile(0.25)
    q75 = df["n_tokens_approx"].quantile(0.75)
    return (
        df[df["n_tokens_approx"] <= q25].copy(),
        df[df["n_tokens_approx"] >= q75].copy(),
    )


def numeracy_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (low_num_df, high_num_df) split at median n_numbers.
    """
    median = df["n_numbers"].median()
    return (
        df[df["n_numbers"] <= median].copy(),
        df[df["n_numbers"] > median].copy(),
    )


def evaluate_ood(clf, df_train: pd.DataFrame, df_ood: pd.DataFrame,
                 text_col: str = "claim_text",
                 label_col: str = "label") -> dict:
    """
    Fit clf on df_train and evaluate on df_ood.

    Returns dict of metrics on the OOD slice.
    """
    clf.fit(df_train[text_col], df_train[label_col])
    y_true = df_ood[label_col].values
    y_pred = clf.predict(df_ood[text_col])
    y_prob = clf.predict_proba(df_ood[text_col])[:, 1]
    return compute_all(y_true, y_pred, y_prob)


def run_ood_battery(clf_fn, df: pd.DataFrame,
                    text_col: str = "claim_text",
                    label_col: str = "label") -> dict:
    """
    Run all three OOD evaluations on a freshly instantiated classifier.

    Args:
        clf_fn: Zero-argument callable returning a fresh unfitted classifier.
        df:     Full deduplicated training dataframe with feature columns.
    Returns:
        Dict: split_name → metrics dict.
    """
    results = {}

    # 1. Ticker-out
    train, ood = ticker_out_split(df)
    if len(ood) > 0:
        m = evaluate_ood(clf_fn(), train, ood, text_col, label_col)
        results["ticker_out"] = m
        print(f"  ticker_out   macro_f1={m['macro_f1']:.4f}  f1_false={m['f1_false']:.4f}")

    # 2. Length OOD
    short, long_ = length_splits(df)
    if len(short) > 0 and len(long_) > 0:
        m_short = evaluate_ood(clf_fn(), long_, short, text_col, label_col)
        m_long  = evaluate_ood(clf_fn(), short, long_, text_col, label_col)
        results["length_short"] = m_short
        results["length_long"]  = m_long
        print(f"  length_short macro_f1={m_short['macro_f1']:.4f}")
        print(f"  length_long  macro_f1={m_long['macro_f1']:.4f}")

    # 3. Numeracy OOD
    low_num, high_num = numeracy_split(df)
    if len(low_num) > 0 and len(high_num) > 0:
        m = evaluate_ood(clf_fn(), low_num, high_num, text_col, label_col)
        results["high_numeracy"] = m
        print(f"  high_numeracy macro_f1={m['macro_f1']:.4f}")

    return results
