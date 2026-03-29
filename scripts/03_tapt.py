"""
Script 03 — TAPT + scratch Transformer baseline (R003).

Trains a masked LM on the training split only, then fine-tunes
a sequence classifier. Evaluates on val and test splits.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.guard import check_path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.data.loader import sha256_file
from src.data.folds import load_split
from src.training.tapt import run_tapt
from src.training.trainer import CheckpointManager, run_training
from src.evaluation.metrics import compute_all
from src.reporting.logger import log_run, save_manifest
from src.utils import get_device

SFT_PATH   = "data/raw/misinfo_SFT_train_for_cot.json"
RL_PATH    = "data/raw/misinfo_RL_train_for_cot.json"
FOLDS_PATH = "data/processed/folds.json"
PARQUET    = "data/processed/train_dedup.parquet"
RUN_LOG    = "runs/run_log.csv"
RUN_ID     = "R003"
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 256
BATCH_SIZE = 8
N_EPOCHS   = 20
PATIENCE   = 5
LR         = 2e-5
SEEDS      = [0, 1, 2]
DEVICE     = get_device()


class ClaimDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.enc    = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
        )

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


def main():
    hashes = {"SFT": sha256_file(SFT_PATH), "RL": sha256_file(RL_PATH)}
    df = pd.read_parquet(PARQUET)
    df = load_split(df, folds_path=FOLDS_PATH)

    train = df[df["split"] == "train"]
    val   = df[df["split"] == "val"]
    test  = df[df["split"] == "test"]
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    seed_val_f1s = []
    for seed in SEEDS:
        print(f"\n=== R003 | Seed {seed} ===")
        torch.manual_seed(seed)

        # TAPT on training split only
        tapt_path = run_tapt(
            texts=train["claim_text"].tolist(),
            fold_idx=0, run_id=f"{RUN_ID}_seed{seed}",
            model_name=MODEL_NAME, n_epochs=3,
            batch_size=BATCH_SIZE, device=DEVICE,
        )

        tokenizer = AutoTokenizer.from_pretrained(tapt_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            tapt_path, num_labels=2
        ).to(DEVICE)

        train_dl = DataLoader(ClaimDataset(train["claim_text"].tolist(),
                                           train["label"].tolist(), tokenizer, MAX_LEN),
                              batch_size=BATCH_SIZE, shuffle=True)
        val_dl   = DataLoader(ClaimDataset(val["claim_text"].tolist(),
                                           val["label"].tolist(),   tokenizer, MAX_LEN),
                              batch_size=BATCH_SIZE)
        test_dl  = DataLoader(ClaimDataset(test["claim_text"].tolist(),
                                           test["label"].tolist(),  tokenizer, MAX_LEN),
                              batch_size=BATCH_SIZE)

        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(train_dl) * N_EPOCHS
        get_linear_schedule_with_warmup(optimizer, int(0.1*total_steps), total_steps)

        ckpt = CheckpointManager(f"{RUN_ID}_seed{seed}")
        run_training(model, train_dl, val_dl, optimizer, ckpt,
                     DEVICE, n_epochs=N_EPOCHS, patience=PATIENCE)

        # Final test evaluation
        from src.training.trainer import eval_epoch
        y_true, y_pred, y_prob = eval_epoch(model, test_dl, DEVICE)
        m_test = compute_all(y_true, y_pred, y_prob)

        y_true_v, y_pred_v, y_prob_v = eval_epoch(model, val_dl, DEVICE)
        m_val = compute_all(y_true_v, y_pred_v, y_prob_v)
        seed_val_f1s.append(m_val["macro_f1"])

        print(f"  val  macro_f1={m_val['macro_f1']:.4f}")
        print(f"  test macro_f1={m_test['macro_f1']:.4f}")

        log_run(RUN_LOG, {
            "run_id": f"{RUN_ID}_seed{seed}",
            "model":  f"Transformer+TAPT ({MODEL_NAME})",
            "dedup":  "Y", "split": "stratified", "seed": seed,
            **{k: f"{v:.4f}" for k, v in m_val.items()},
            "notes":  f"TAPT distilbert test_macro_f1={m_test['macro_f1']:.4f}",
        })

    save_manifest(RUN_ID, config={"model": MODEL_NAME, "seeds": SEEDS}, hashes=hashes)
    print(f"\nR003 avg val macro_f1={np.mean(seed_val_f1s):.4f}")


if __name__ == "__main__":
    main()
