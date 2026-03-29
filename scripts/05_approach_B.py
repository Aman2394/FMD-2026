"""
Script 05 — Approach B: Contrastive Counterfactual Learning (R005).
Uses FinBERT + SupCon loss.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.guard import check_path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.data.loader import sha256_file
from src.data.folds import load_split
from src.models.approach_B import ContrastiveClassifier
from src.training.trainer import CheckpointManager, run_training, eval_epoch
from src.evaluation.metrics import compute_all
from src.reporting.logger import log_run, save_manifest
from src.utils import get_device

SFT_PATH   = "data/raw/misinfo_SFT_train_for_cot.json"
RL_PATH    = "data/raw/misinfo_RL_train_for_cot.json"
FOLDS_PATH = "data/processed/folds.json"
PARQUET    = "data/processed/train_dedup.parquet"
RUN_LOG    = "runs/run_log.csv"
RUN_ID     = "R005"
MODEL_NAME = "ProsusAI/finbert"
MAX_LEN    = 256
BATCH_SIZE = 32
N_EPOCHS   = 20
PATIENCE   = 5
LR         = 2e-5
ALPHA      = 0.5
SEEDS      = [0, 1, 2]
DEVICE     = get_device()


class ClaimDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.enc = tokenizer(texts, truncation=True, padding="max_length",
                             max_length=max_length, return_tensors="pt")

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {"input_ids":      self.enc["input_ids"][idx],
                "attention_mask": self.enc["attention_mask"][idx],
                "labels":         self.labels[idx]}


def main():
    hashes = {"SFT": sha256_file(SFT_PATH), "RL": sha256_file(RL_PATH)}
    df = pd.read_parquet(PARQUET)
    df = load_split(df, folds_path=FOLDS_PATH)

    train = df[df["split"] == "train"].reset_index(drop=True)
    val   = df[df["split"] == "val"].reset_index(drop=True)
    test  = df[df["split"] == "test"].reset_index(drop=True)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    seed_val_f1s = []
    for seed in SEEDS:
        print(f"\n=== Approach B (FinBERT) | Seed {seed} ===")
        torch.manual_seed(seed)

        train_dl = DataLoader(ClaimDataset(train["claim_text"].tolist(),
                                           train["label"].tolist(), tokenizer, MAX_LEN),
                              batch_size=BATCH_SIZE, shuffle=True)
        val_dl   = DataLoader(ClaimDataset(val["claim_text"].tolist(),
                                           val["label"].tolist(), tokenizer, MAX_LEN),
                              batch_size=BATCH_SIZE)
        test_dl  = DataLoader(ClaimDataset(test["claim_text"].tolist(),
                                           test["label"].tolist(), tokenizer, MAX_LEN),
                              batch_size=BATCH_SIZE)

        encoder = AutoModel.from_pretrained(MODEL_NAME)
        model   = ContrastiveClassifier(encoder, alpha=ALPHA).to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(train_dl) * N_EPOCHS
        get_linear_schedule_with_warmup(optimizer, int(0.1*total_steps), total_steps)

        ckpt = CheckpointManager(f"{RUN_ID}_seed{seed}")
        run_training(model, train_dl, val_dl, optimizer, ckpt,
                     DEVICE, n_epochs=N_EPOCHS, patience=PATIENCE)

        y_true_v, y_pred_v, y_prob_v = eval_epoch(model, val_dl, DEVICE)
        y_true_t, y_pred_t, y_prob_t = eval_epoch(model, test_dl, DEVICE)
        m_val  = compute_all(y_true_v, y_pred_v, y_prob_v)
        m_test = compute_all(y_true_t, y_pred_t, y_prob_t)
        seed_val_f1s.append(m_val["macro_f1"])

        print(f"  val  macro_f1={m_val['macro_f1']:.4f}")
        print(f"  test macro_f1={m_test['macro_f1']:.4f}")

        log_run(RUN_LOG, {
            "run_id": f"{RUN_ID}_seed{seed}", "model": "Approach B (FinBERT Contrastive)",
            "dedup": "Y", "split": "stratified", "seed": seed,
            **{k: f"{v:.4f}" for k, v in m_val.items()},
            "notes": f"FinBERT SupCon alpha={ALPHA} test_f1={m_test['macro_f1']:.4f}",
        })

    save_manifest(RUN_ID, config={"model": MODEL_NAME, "alpha": ALPHA, "seeds": SEEDS}, hashes=hashes)
    print(f"\nR005 avg val macro_f1={np.mean(seed_val_f1s):.4f}")


if __name__ == "__main__":
    main()
