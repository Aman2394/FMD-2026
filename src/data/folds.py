import json
import pandas as pd
from sklearn.model_selection import train_test_split


def make_split(df: pd.DataFrame,
               val_size: float = 0.15,
               test_size: float = 0.15,
               seed: int = 42,
               output_path: str = "data/processed/folds.json") -> pd.DataFrame:
    """
    Stratified train / val / test split.

    Assigns each row a 'split' column: 'train', 'val', or 'test'.
    Written once to folds.json — never regenerate after first run.
    """
    df = df.copy()

    # First carve out test set
    idx_trainval, idx_test = train_test_split(
        df.index,
        test_size=test_size,
        stratify=df["label"],
        random_state=seed,
    )
    # Then split remaining into train / val
    val_ratio = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_ratio,
        stratify=df.loc[idx_trainval, "label"],
        random_state=seed,
    )

    df["split"] = "train"
    df.loc[idx_val,  "split"] = "val"
    df.loc[idx_test, "split"] = "test"

    # Persist
    split_map = df[["claim_hash", "split"]].set_index("claim_hash")["split"].to_dict()
    with open(output_path, "w") as f:
        json.dump(split_map, f)

    counts = df["split"].value_counts()
    label_dist = df.groupby("split")["label"].mean().round(3)
    print(f"Split saved to {output_path}")
    print(f"  train={counts['train']}  val={counts['val']}  test={counts['test']}")
    print(f"  label=1 rate — train={label_dist['train']}  val={label_dist['val']}  test={label_dist['test']}")
    return df


def load_split(df: pd.DataFrame,
               folds_path: str = "data/processed/folds.json") -> pd.DataFrame:
    """Attach pre-computed split assignments to df (read-only)."""
    with open(folds_path) as f:
        split_map = json.load(f)
    df = df.copy()
    df["split"] = df["claim_hash"].map(split_map).fillna("train")
    return df
