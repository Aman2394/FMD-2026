"""
Script 11 — Knowledge Distillation: Qwen2.5 teacher → FinBERT student (R011).

Phase 1 (runs once):
    Zero-shot Qwen2.5 inference on training split → soft labels
    saved to data/processed/teacher_probs.npy

Phase 2 (runs per seed):
    FinBERT fine-tuned with:
        L = α · CE(student, hard_label) + (1-α) · T² · KL(teacher ‖ student)

GPU requirements:
    Teacher inference:  ~8 GB VRAM (4-bit Qwen2.5-7B) or ~4 GB (Qwen2.5-3B)
    Student training:   ~4 GB VRAM (FinBERT, batch 32)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.guard import check_path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

from src.data.loader import sha256_file
from src.data.folds import load_split
from src.training.distill import kd_loss, get_teacher_probs
from src.training.trainer import CheckpointManager, eval_epoch
from src.evaluation.metrics import compute_all
from src.reporting.logger import log_run, save_manifest
from src.utils import get_device

# ── paths ────────────────────────────────────────────────────────────────────
SFT_PATH      = "data/raw/misinfo_SFT_train_for_cot.json"
RL_PATH       = "data/raw/misinfo_RL_train_for_cot.json"
FOLDS_PATH    = "data/processed/folds.json"
PARQUET       = "data/processed/train_dedup.parquet"
TEACHER_PROBS = "data/processed/teacher_probs.npy"
RUN_LOG       = "runs/run_log.csv"
RUN_ID        = "R011"

# ── hyper-params ─────────────────────────────────────────────────────────────
TEACHER_MODEL = "Qwen/Qwen2.5-7B-Instruct"   # swap to "Qwen/Qwen2.5-3B-Instruct" for T4
STUDENT_MODEL = "ProsusAI/finbert"
MAX_LEN       = 256
BATCH_SIZE    = 32
N_EPOCHS      = 20
PATIENCE      = 5
LR            = 2e-5
KD_ALPHA      = 0.5     # 0 = pure distillation, 1 = pure hard-label CE
KD_TEMP       = 2.0     # softening temperature
SEEDS         = [0, 1, 2]
DEVICE        = get_device()


# ── dataset ───────────────────────────────────────────────────────────────────
class DistillDataset(Dataset):
    def __init__(self, df, teacher_probs_false, tokenizer, max_length=256):
        self.labels = torch.tensor(df["label"].values, dtype=torch.long)
        p = torch.tensor(teacher_probs_false, dtype=torch.float32)
        self.teacher_probs = torch.stack([1.0 - p, p], dim=1)  # (N, 2): [P(true), P(false)]
        self.enc = tokenizer(
            df["claim_text"].tolist(), truncation=True,
            padding="max_length", max_length=max_length, return_tensors="pt",
        )

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "labels":         self.labels[idx],
            "teacher_probs":  self.teacher_probs[idx],
        }


# ── student model ─────────────────────────────────────────────────────────────
class StudentClassifier(nn.Module):
    def __init__(self, model_name, num_labels=2, kd_alpha=0.5, kd_temp=2.0):
        super().__init__()
        self.backbone  = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, ignore_mismatched_sizes=True,
        )
        self.kd_alpha  = kd_alpha
        self.kd_temp   = kd_temp

    def forward(self, input_ids, attention_mask, labels=None,
                teacher_probs=None, **kwargs):
        out    = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits
        result = {"logits": logits}
        if labels is not None:
            if teacher_probs is not None:
                result["loss"] = kd_loss(
                    logits, teacher_probs, labels,
                    alpha=self.kd_alpha, temperature=self.kd_temp,
                )
            else:
                result["loss"] = F.cross_entropy(logits, labels)
        return result


# ── training loop ─────────────────────────────────────────────────────────────
def run_training(model, train_loader, val_loader, optimizer, scheduler,
                 ckpt, device, n_epochs, patience):
    best_val, no_improve = -1.0, 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(**batch)
            out["loss"].backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += out["loss"].item()

        y_true, y_pred, y_prob = eval_epoch(model, val_loader, device)
        m = compute_all(y_true, y_pred, y_prob)
        print(f"Epoch {epoch:03d} | loss={total_loss/len(train_loader):.4f} | "
              f"macro_f1={m['macro_f1']:.4f} | f1_false={m['f1_false']:.4f}")
        ckpt.update(model, m, epoch)

        if m["macro_f1"] > best_val:
            best_val, no_improve = m["macro_f1"], 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    return best_val


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    hashes = {"SFT": sha256_file(SFT_PATH), "RL": sha256_file(RL_PATH)}
    df = pd.read_parquet(PARQUET)
    df = load_split(df, folds_path=FOLDS_PATH)

    train = df[df["split"] == "train"].reset_index(drop=True)
    val   = df[df["split"] == "val"].reset_index(drop=True)
    test  = df[df["split"] == "test"].reset_index(drop=True)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    # ── Phase 1: teacher inference (once) ────────────────────────────────────
    if not os.path.exists(TEACHER_PROBS):
        print(f"\nPhase 1 — {TEACHER_MODEL} zero-shot inference on {len(train)} examples...")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        t_tok = AutoTokenizer.from_pretrained(TEACHER_MODEL)
        t_mod = AutoModelForCausalLM.from_pretrained(
            TEACHER_MODEL,
            quantization_config=bnb_cfg,
            device_map="auto",
        )
        all_probs = get_teacher_probs(
            train["claim_text"].tolist(), t_mod, t_tok,
            device=DEVICE, batch_size=4,
        )
        np.save(TEACHER_PROBS, all_probs)
        print(f"Saved → {TEACHER_PROBS}  (mean={all_probs.mean():.3f})")
        del t_mod, t_tok
        torch.cuda.empty_cache()
    else:
        print(f"Loading cached teacher probs from {TEACHER_PROBS}")
        all_probs = np.load(TEACHER_PROBS)

    # Sanity-check: teacher accuracy on train hard labels
    teacher_preds = (all_probs > 0.5).astype(int)
    teacher_acc   = (teacher_preds == train["label"].values).mean()
    print(f"Teacher train accuracy (zero-shot): {teacher_acc:.3f}  "
          f"mean_p_false={all_probs.mean():.3f}")

    # ── Phase 2: student fine-tuning ─────────────────────────────────────────
    s_tok = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    UNINFORM = np.full(len(val), 0.5, dtype=np.float32)   # no teacher signal on val/test

    seed_val_f1s = []
    for seed in SEEDS:
        print(f"\n=== R011 | KD FinBERT | Seed {seed} ===")
        torch.manual_seed(seed)

        train_dl = DataLoader(
            DistillDataset(train, all_probs, s_tok, MAX_LEN),
            batch_size=BATCH_SIZE, shuffle=True,
        )
        val_dl = DataLoader(
            DistillDataset(val, UNINFORM, s_tok, MAX_LEN),
            batch_size=BATCH_SIZE,
        )
        test_dl = DataLoader(
            DistillDataset(test, np.full(len(test), 0.5, dtype=np.float32), s_tok, MAX_LEN),
            batch_size=BATCH_SIZE,
        )

        model     = StudentClassifier(STUDENT_MODEL, kd_alpha=KD_ALPHA, kd_temp=KD_TEMP).to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(train_dl) * N_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(0.1 * total_steps), total_steps,
        )

        ckpt     = CheckpointManager(f"{RUN_ID}_seed{seed}")
        best_val = run_training(model, train_dl, val_dl, optimizer, scheduler,
                                ckpt, DEVICE, N_EPOCHS, PATIENCE)
        seed_val_f1s.append(best_val)

        y_true_v, y_pred_v, y_prob_v = eval_epoch(model, val_dl,  DEVICE)
        y_true_t, y_pred_t, y_prob_t = eval_epoch(model, test_dl, DEVICE)
        m_val  = compute_all(y_true_v, y_pred_v, y_prob_v)
        m_test = compute_all(y_true_t, y_pred_t, y_prob_t)

        os.makedirs("runs/oof", exist_ok=True)
        np.save(f"runs/oof/{RUN_ID}_seed{seed}_prob.npy", y_prob_v)
        np.save(f"runs/oof/{RUN_ID}_seed{seed}_true.npy", y_true_v)

        print(f"  val  macro_f1={m_val['macro_f1']:.4f}")
        print(f"  test macro_f1={m_test['macro_f1']:.4f}")

        log_run(RUN_LOG, {
            "run_id": f"{RUN_ID}_seed{seed}",
            "model":  f"KD FinBERT (teacher={TEACHER_MODEL.split('/')[-1]})",
            "dedup": "Y", "split": "stratified", "seed": seed,
            **{k: f"{v:.4f}" for k, v in m_val.items()},
            "notes": (f"alpha={KD_ALPHA} T={KD_TEMP} "
                      f"teacher_acc={teacher_acc:.3f} "
                      f"test_f1={m_test['macro_f1']:.4f}"),
        })

    save_manifest(RUN_ID, config={
        "teacher": TEACHER_MODEL, "student": STUDENT_MODEL,
        "kd_alpha": KD_ALPHA, "kd_temp": KD_TEMP, "seeds": SEEDS,
    }, hashes=hashes)
    print(f"\nR011 avg val macro_f1={np.mean(seed_val_f1s):.4f}")


if __name__ == "__main__":
    main()
