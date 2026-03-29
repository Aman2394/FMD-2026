# CLAUDE.md — Financial Misinformation Classifier (Training-Only Programme)

## Project overview

Binary classifier for financial-news snippets: **True** (no misinformation) vs **False** (misinformation).  
All development — tokeniser fitting, TAPT, augmentation, model selection, calibration, hyperparameter tuning — uses **only the labelled training datasets**.  
The blind set is sealed until the single final submission run.

---

## Repository layout

```
project/
├── CLAUDE.md                        # This file
├── data/
│   ├── raw/
│   │   ├── misinfo_SFT_train_for_cot.json
│   │   └── misinfo_RL_train_for_cot.json
│   │   # BLIND FILE — do NOT place here until final run
│   └── processed/
│       ├── train_dedup.parquet       # After exact-dedup
│       ├── features.parquet          # Cached regex extractions (keyed by claim hash)
│       └── folds.json                # GroupKFold assignments — NEVER regenerate
├── src/
│   ├── data/
│   │   ├── guard.py                  # Blind-access guard (raises if blind.json opened without flag)
│   │   ├── loader.py                 # Load + hash JSONs; extract prompt_text / claim_text / label
│   │   ├── dedup.py                  # Exact-dedup by claim_text SHA-1; group key computation
│   │   ├── features.py               # Deterministic regex extractor (numbers, tickers, time, units)
│   │   └── folds.py                  # GroupKFold(title) + ticker OOD split; exports folds.json
│   ├── models/
│   │   ├── baselines.py              # TF-IDF+LR, SVM+cal, kNN vote
│   │   ├── approach_A.py             # Numeric dual-stream Transformer + aux losses
│   │   ├── approach_B.py             # Contrastive counterfactual (SupCon + BCE)
│   │   ├── approach_C.py             # Fold-safe kNN memory + calibrator
│   │   ├── approach_D.py             # Heterogeneous GNN (entity–event graph)
│   │   ├── approach_E.py             # LoRA / from-scratch instruction classifier
│   │   └── approach_F.py             # Uncertainty-aware calibrated ensemble
│   ├── training/
│   │   ├── tapt.py                   # Fold-wise MLM/denoising TAPT on training corpus only
│   │   ├── trainer.py                # Training loop, checkpointing, loss dispatch
│   │   └── augment.py                # Training-distribution perturbations (numeric/ticker/time/ellipsis)
│   ├── evaluation/
│   │   ├── metrics.py                # macro-F1, PR-AUC(False), ROC-AUC, Brier, ECE
│   │   ├── calibration.py            # Temperature scaling + isotonic regression (CV-only)
│   │   ├── robustness.py             # Perturbation battery; delta reporting
│   │   ├── ood.py                    # OOD split evaluations (ticker-out, length, numeracy)
│   │   └── stats.py                  # Paired bootstrap, McNemar, DeLong, Holm-Bonferroni / BH
│   └── reporting/
│       ├── logger.py                 # CSV run-log writer (see template below)
│       └── plots.py                  # ROC/PR curves, calibration plots, robustness curves
├── scripts/
│   ├── 01_eda.py                     # Mandatory training-only stats + minimal-pair inventory
│   ├── 02_baselines.py
│   ├── 03_tapt.py
│   ├── 04_approach_A.py
│   ├── 05_approach_B.py
│   ├── 06_approach_C.py
│   ├── 07_approach_D.py
│   ├── 08_approach_E.py
│   ├── 09_ensemble_F.py
│   ├── 10_ablations.py
│   └── 99_final_blind_eval.py        # ONE-SHOT — only run with ALLOW_BLIND_EVAL=1
├── runs/
│   ├── run_log.csv                   # Append-only; see CSV template
│   └── manifests/                    # Per-run JSON: hashes, seeds, config, commit
├── checkpoints/                      # best_macro_f1/, best_f1_false/, best_pr_auc/, best_calib/
├── results/
│   ├── cv_summary.md                 # Markdown table (see template)
│   └── submission/
│       └── predictions.csv           # Final blind predictions (generated once)
├── tests/
│   ├── test_guard.py                 # Blind-access guard unit tests
│   ├── test_dedup.py
│   ├── test_folds_no_leakage.py      # Assert no title overlap across folds
│   └── test_features.py
└── requirements.txt
```

---

## Environment setup

```bash
python -m venv .venv && source .venv/bin/activate

pip install \
  torch torchvision \
  transformers datasets tokenizers \
  scikit-learn scipy numpy pandas pyarrow \
  torch-geometric \          # Approach D (GNN)
  peft \                     # Approach E (LoRA)
  matplotlib seaborn \
  pytest

# Pin versions for reproducibility
pip freeze > requirements.txt
```

**Python:** 3.10+  
**Recommended:** CUDA-enabled GPU (single RTX 3090/4090 sufficient; QLoRA for Approach E if VRAM < 24 GB)

---

## Critical compliance rules

### Blind-access guard

Every script imports `guard.py` as its **first** action:

```python
# src/data/guard.py
import os, sys

BLIND_FILENAMES = {"blind.json", "blind_test.json"}   # extend as needed

def check_path(path: str):
    if any(b in str(path) for b in BLIND_FILENAMES):
        if not os.environ.get("ALLOW_BLIND_EVAL"):
            raise RuntimeError(
                f"BLIND ACCESS BLOCKED: {path}\n"
                "Set ALLOW_BLIND_EVAL=1 only for the final submission run."
            )
        else:
            _audit_log(path)

def _audit_log(path: str):
    import datetime
    with open("blind_access_audit.log", "a") as f:
        f.write(f"{datetime.datetime.utcnow().isoformat()} BLIND_READ {path}\n")
```

Usage in every loader:

```python
from src.data.guard import check_path
check_path(filepath)          # raises before any file is opened
```

### Dataset hashing (run every script)

```python
import hashlib, json

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

HASHES = {
    "SFT": sha256_file("data/raw/misinfo_SFT_train_for_cot.json"),
    "RL":  sha256_file("data/raw/misinfo_RL_train_for_cot.json"),
}
# Log HASHES to run manifest at start of every experiment
```

### Fold file lock

`data/processed/folds.json` is generated **once** by `src/data/folds.py` and committed.  
Never regenerate it; load it read-only in all experiments.

---

## Data pipeline

### Step 1 — Load and extract

```python
# src/data/loader.py
import json, hashlib, re

INSTRUCTION_PREFIX_RE = re.compile(
    r"^.*?(true|false).*?\n.*?\n\s*", re.IGNORECASE | re.DOTALL
)

def load_training(path: str) -> list[dict]:
    with open(path) as f:
        raw = json.load(f)
    records = []
    for item in raw:
        prompt = item["prompt"]          # full prompt_text
        label  = 1 if str(item["label"]).strip().lower() == "false" else 0
        claim  = _extract_claim(prompt)
        records.append({
            "prompt_text": prompt,
            "claim_text":  claim,
            "label":       label,        # 1 = False (misinformation), 0 = True
            "claim_hash":  hashlib.sha1(claim.encode()).hexdigest(),
        })
    return records

def _extract_claim(prompt: str) -> str:
    # Strip fixed instruction prefix (first 2 lines + blank lines)
    lines = prompt.splitlines()
    for i, line in enumerate(lines):
        if line.strip() and i >= 2:
            return "\n".join(lines[i:]).strip()
    return prompt.strip()
```

### Step 2 — Deduplication

```python
# src/data/dedup.py  (PRIMARY policy: exact-dedup by claim_text hash)
import pandas as pd

def dedup(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    before = len(df)
    df = df.drop_duplicates(subset="claim_hash", keep="first")
    print(f"Dedup: {before} → {len(df)} rows ({before - len(df)} removed)")
    return df.reset_index(drop=True)
```

### Step 3 — Feature extraction

```python
# src/data/features.py
import re, pandas as pd

NUMBER_RE   = re.compile(r"[+-]?\$?[\d,]+\.?\d*(?:\s*(?:million|billion|bn|mn|[KkMmBb]))?")
UNIT_RE     = re.compile(r"\b(\d+\.?\d*)\s*(%|percent|bps|basis points)\b", re.I)
TICKER_RE   = re.compile(r"\b([A-Z]{1,5})\b(?=\s*[\(\[]?\b(?:NASDAQ|NYSE|NYSE MKT)\b)?|"
                         r"\(([A-Z]{1,5})\)")
TIME_RE     = re.compile(r"\b(20\d{2}|Q[1-4]|FY\d{2,4}|H[12]\s*20\d{2}|"
                         r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
                         r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
                         r"Dec(?:ember)?)\b")
ELLIPSIS_RE = re.compile(r"\.{2,}|…|\[\.\.\.\]|\(continued\)", re.I)
TITLE_SEP   = re.compile(r"\s[-—–:]\s")

def extract(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["numbers"]        = df["claim_text"].apply(lambda t: NUMBER_RE.findall(t))
    df["n_numbers"]      = df["numbers"].apply(len)
    df["units"]          = df["claim_text"].apply(lambda t: UNIT_RE.findall(t))
    df["tickers"]        = df["claim_text"].apply(_extract_tickers)
    df["n_tickers"]      = df["tickers"].apply(len)
    df["time_tokens"]    = df["claim_text"].apply(lambda t: TIME_RE.findall(t))
    df["has_ellipsis"]   = df["claim_text"].apply(lambda t: bool(ELLIPSIS_RE.search(t)))
    df["ellipsis_pos"]   = df["claim_text"].apply(_ellipsis_position)
    df["group_title"]    = df["claim_text"].apply(_extract_title)
    df["group_ticker"]   = df["tickers"].apply(lambda t: t[0] if t else "NONE")
    df["n_tokens_approx"]= df["claim_text"].apply(lambda t: len(t.split()))
    return df

def _extract_tickers(text: str) -> list[str]:
    return list(dict.fromkeys(
        t for match in TICKER_RE.finditer(text)
        for t in match.groups() if t
    ))

def _ellipsis_position(text: str) -> str:
    m = ELLIPSIS_RE.search(text)
    if not m: return "none"
    pos = m.start() / max(len(text), 1)
    return "mid" if pos < 0.8 else "tail"

def _extract_title(text: str) -> str:
    first_line = text.splitlines()[0] if text else ""
    parts = TITLE_SEP.split(first_line, maxsplit=1)
    return parts[0].strip()[:120]
```

### Step 4 — Fold assignment

```python
# src/data/folds.py
import json
from sklearn.model_selection import GroupKFold
import pandas as pd

def make_folds(df: pd.DataFrame, n_splits: int = 5,
               output_path: str = "data/processed/folds.json") -> pd.DataFrame:
    gkf = GroupKFold(n_splits=n_splits)
    df["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(
        gkf.split(df, df["label"], groups=df["group_title"])
    ):
        df.loc[val_idx, "fold"] = fold_idx

    # Persist fold assignments
    fold_map = df[["claim_hash", "fold"]].set_index("claim_hash")["fold"].to_dict()
    with open(output_path, "w") as f:
        json.dump(fold_map, f)
    print(f"Folds saved to {output_path}")

    # Leakage check: no title in both train and val within any fold
    for fold_idx in range(n_splits):
        val_titles  = set(df.loc[df["fold"] == fold_idx,  "group_title"])
        train_titles= set(df.loc[df["fold"] != fold_idx,  "group_title"])
        overlap = val_titles & train_titles
        assert not overlap, f"LEAKAGE in fold {fold_idx}: {overlap}"
    print("Leakage check passed.")
    return df
```

---

## Baseline models

```python
# src/models/baselines.py
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import numpy as np

def tfidf_lr(max_features: int = 50_000) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            max_features=max_features, sublinear_tf=True
        )),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0))
    ])

def tfidf_svm() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            max_features=30_000, sublinear_tf=True
        )),
        ("clf", CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3))
    ])

def knn_vote(k: int = 10) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("knn",   KNeighborsClassifier(n_neighbors=k, metric="cosine"))
    ])
```

---

## Novel approaches (skeletons)

### Approach A — Numeric dual-stream Transformer

```python
# src/models/approach_A.py
import torch, torch.nn as nn

class NumericEncoder(nn.Module):
    """Encodes a fixed-size numeric feature vector per claim."""
    def __init__(self, input_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
    def forward(self, x):           # x: (B, input_dim)
        return self.net(x)          # (B, hidden)

class DualStreamClassifier(nn.Module):
    def __init__(self, text_encoder, numeric_dim=16, numeric_hidden=64,
                 num_labels=2, dropout=0.1):
        super().__init__()
        self.text_enc   = text_encoder           # HuggingFace encoder or scratch Transformer
        self.num_enc    = NumericEncoder(numeric_dim, numeric_hidden)
        text_hidden     = text_encoder.config.hidden_size
        fused_dim       = text_hidden + numeric_hidden

        self.gate       = nn.Linear(fused_dim, fused_dim)
        self.classifier = nn.Linear(fused_dim, num_labels)
        self.aux_masknum= nn.Linear(fused_dim, 10)   # magnitude bucket prediction
        self.aux_unit   = nn.Linear(fused_dim, 5)    # unit type prediction
        self.aux_cons   = nn.Linear(fused_dim, 2)    # internal consistency pseudo-label
        self.dropout    = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, numeric_feats,
                labels=None, masknum_labels=None, unit_labels=None, cons_labels=None):
        text_out  = self.text_enc(input_ids=input_ids,
                                  attention_mask=attention_mask).last_hidden_state[:, 0]
        num_out   = self.num_enc(numeric_feats)
        fused     = torch.cat([text_out, num_out], dim=-1)
        fused     = fused * torch.sigmoid(self.gate(fused))   # gating
        fused     = self.dropout(fused)

        logits    = self.classifier(fused)
        output    = {"logits": logits}

        if labels is not None:
            weight    = torch.tensor([1.0, 992/825]).to(labels.device)  # class weights
            loss_cls  = nn.CrossEntropyLoss(weight=weight)(logits, labels)
            loss_mn   = nn.CrossEntropyLoss()(self.aux_masknum(fused), masknum_labels) \
                        if masknum_labels is not None else 0.0
            loss_unit = nn.CrossEntropyLoss()(self.aux_unit(fused), unit_labels) \
                        if unit_labels is not None else 0.0
            loss_cons = nn.CrossEntropyLoss()(self.aux_cons(fused), cons_labels) \
                        if cons_labels is not None else 0.0
            output["loss"] = loss_cls + 0.3*loss_mn + 0.2*loss_unit + 0.1*loss_cons
        return output
```

### Approach B — Contrastive minimal-pair learning

```python
# src/models/approach_B.py
import torch, torch.nn as nn, torch.nn.functional as F

class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: (B, D) L2-normalised; labels: (B,)
        features = F.normalize(features, dim=-1)
        sim      = torch.matmul(features, features.T) / self.temperature
        mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask_self= torch.eye(len(labels), device=labels.device)
        mask_pos = mask_pos - mask_self
        exp_sim  = torch.exp(sim) * (1 - mask_self)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        loss     = -(mask_pos * log_prob).sum(dim=1) / (mask_pos.sum(dim=1) + 1e-8)
        return loss.mean()

class ContrastiveClassifier(nn.Module):
    def __init__(self, encoder, num_labels=2, alpha=0.5, dropout=0.1):
        super().__init__()
        self.encoder    = encoder
        self.classifier = nn.Linear(encoder.config.hidden_size, num_labels)
        self.supcon     = SupConLoss()
        self.alpha      = alpha
        self.dropout    = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, labels=None):
        h       = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask).last_hidden_state[:, 0]
        h       = self.dropout(h)
        logits  = self.classifier(h)
        output  = {"logits": logits, "embeddings": h}
        if labels is not None:
            weight        = torch.tensor([1.0, 992/825]).to(labels.device)
            loss_cls      = nn.CrossEntropyLoss(weight=weight)(logits, labels)
            loss_con      = self.supcon(h, labels)
            output["loss"]= loss_cls + self.alpha * loss_con
        return output
```

### Approach C — Fold-safe kNN memory

```python
# src/models/approach_C.py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class FoldSafeMemory:
    """kNN retrieval restricted to the training fold only."""
    def __init__(self, k: int = 15):
        self.k    = k
        self.index= None
        self.y    = None

    def build(self, X_train: np.ndarray, y_train: np.ndarray):
        self.index = NearestNeighbors(n_neighbors=self.k, metric="cosine").fit(X_train)
        self.y     = y_train

    def neighbour_features(self, X_query: np.ndarray) -> np.ndarray:
        distances, indices = self.index.kneighbors(X_query)
        feats = []
        for dist, idx in zip(distances, indices):
            sim     = 1 - dist
            nbr_y   = self.y[idx]
            vote    = nbr_y.mean()                          # weighted vote
            entropy = -(vote * np.log(vote+1e-9) + (1-vote)*np.log(1-vote+1e-9))
            margin  = sim[nbr_y==1].mean() - sim[nbr_y==0].mean() \
                      if len(np.unique(nbr_y)) > 1 else 0.0
            feats.append([vote, entropy, margin, sim.max(), sim.mean()])
        return np.array(feats)

class KNNCalibrator:
    def __init__(self, k: int = 15):
        self.memory = FoldSafeMemory(k)
        self.scaler = StandardScaler()
        self.meta   = LogisticRegression(class_weight="balanced")

    def fit(self, X_train, y_train, X_val, y_val):
        self.memory.build(X_train, y_train)
        feats_val = self.memory.neighbour_features(X_val)
        feats_val = self.scaler.fit_transform(feats_val)
        self.meta.fit(feats_val, y_val)

    def predict_proba(self, X_query, X_train=None, y_train=None):
        if X_train is not None:
            self.memory.build(X_train, y_train)
        feats = self.memory.neighbour_features(X_query)
        feats = self.scaler.transform(feats)
        return self.meta.predict_proba(feats)
```

### Approach D — Heterogeneous GNN (skeleton)

```python
# src/models/approach_D.py
# Requires: torch-geometric
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv

class FinancialClaimGNN(nn.Module):
    def __init__(self, metadata, hidden: int = 64, num_labels: int = 2):
        super().__init__()
        self.conv1 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden)
            for edge_type in metadata[1]
        })
        self.conv2 = HeteroConv({
            edge_type: SAGEConv(hidden, hidden)
            for edge_type in metadata[1]
        })
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, x_dict, edge_index_dict, claim_batch_idx):
        x = self.conv1(x_dict, edge_index_dict)
        x = {k: v.relu() for k, v in x.items()}
        x = self.conv2(x, edge_index_dict)
        claim_repr = x["claim"][claim_batch_idx]
        return self.classifier(claim_repr)
```

### Approach E — LoRA instruction classifier

```python
# src/models/approach_E.py
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification

def make_lora_classifier(model_name: str, r: int = 16,
                         lora_alpha: int = 32, lora_dropout: float = 0.1):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    config = LoraConfig(
        task_type     = TaskType.SEQ_CLS,
        r             = r,
        lora_alpha    = lora_alpha,
        lora_dropout  = lora_dropout,
        target_modules= ["query", "value"],
        bias          = "none",
    )
    return get_peft_model(model, config)
```

### Approach F — Calibrated ensemble

```python
# src/models/approach_F.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

class CalibratedEnsemble:
    """Stacks OOF predictions from components A–D; calibrates via isotonic regression."""
    def __init__(self, temperature_init: float = 1.0, method: str = "isotonic"):
        self.method = method
        self.stacker= LogisticRegression(class_weight="balanced")
        self.cal    = IsotonicRegression(out_of_bounds="clip") if method == "isotonic" \
                      else None
        self.tau    = 0.5      # abstention threshold (tune on CV)

    def fit(self, oof_probs: np.ndarray, y: np.ndarray):
        # oof_probs: (N, n_models)  — out-of-fold probabilities from each model
        self.stacker.fit(oof_probs, y)
        stacked = self.stacker.predict_proba(oof_probs)[:, 1]
        if self.cal:
            self.cal.fit(stacked, y)

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        stacked = self.stacker.predict_proba(probs)[:, 1]
        if self.cal:
            stacked = self.cal.transform(stacked)
        return stacked

    def predict(self, probs: np.ndarray):
        p = self.predict_proba(probs)
        pred    = np.where(p >= 0.5, 1, 0)
        abstain = np.abs(p - 0.5) < (0.5 - self.tau)
        return pred, abstain, p
```

---

## Augmentation (training-distribution only)

```python
# src/training/augment.py
import random, re
from src.data.features import NUMBER_RE, TICKER_RE, TIME_RE

def numeric_swap(text: str, num_inventory: list[str], rate: float = 0.5) -> str:
    """Replace a random numeric span with a same-unit token from training inventory."""
    spans = [(m.start(), m.end(), m.group()) for m in NUMBER_RE.finditer(text)]
    for start, end, val in random.sample(spans, k=max(1, int(len(spans)*rate))):
        replacement = random.choice(num_inventory)
        text = text[:start] + replacement + text[end:]
    return text

def ticker_swap(text: str, ticker_inventory: list[str]) -> str:
    spans = [(m.start(), m.end()) for m in TICKER_RE.finditer(text)]
    if not spans: return text
    start, end = random.choice(spans)
    return text[:start] + random.choice(ticker_inventory) + text[end:]

def temporal_swap(text: str, time_inventory: list[str]) -> str:
    spans = [(m.start(), m.end()) for m in TIME_RE.finditer(text)]
    if not spans: return text
    start, end = random.choice(spans)
    return text[:start] + random.choice(time_inventory) + text[end:]

def ellipsis_augment(text: str, action: str = "insert") -> str:
    if action == "insert":
        sentences = text.split(". ")
        if len(sentences) > 1:
            cut = random.randint(1, len(sentences)-1)
            return ". ".join(sentences[:cut]) + "..."
    elif action == "remove":
        return re.sub(r"\.{2,}|…|\[\.\.\.\]", "", text).strip()
    return text

def prefix_dropout(prompt: str, rate: float = 0.3) -> str:
    """Randomly drop the instruction prefix to reduce template overfitting."""
    if random.random() < rate:
        lines = prompt.splitlines()
        for i, line in enumerate(lines):
            if line.strip() and i >= 2:
                return "\n".join(lines[i:]).strip()
    return prompt
```

---

## Evaluation harness

```python
# src/evaluation/metrics.py
import numpy as np
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_recall_curve, auc,
    brier_score_loss
)

def compute_all(y_true, y_pred, y_prob, n_bins: int = 10) -> dict:
    macro_f1   = f1_score(y_true, y_pred, average="macro")
    f1_false   = f1_score(y_true, y_pred, pos_label=1)   # 1 = False (misinformation)
    roc_auc    = roc_auc_score(y_true, y_prob)
    prec, rec, _= precision_recall_curve(y_true, y_prob)
    pr_auc     = auc(rec, prec)
    brier      = brier_score_loss(y_true, y_prob)
    ece        = _ece(y_true, y_prob, n_bins)
    return dict(macro_f1=macro_f1, f1_false=f1_false,
                roc_auc=roc_auc, pr_auc_false=pr_auc,
                brier=brier, ece=ece)

def _ece(y_true, y_prob, n_bins: int) -> float:
    bins  = np.linspace(0, 1, n_bins + 1)
    ece   = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0: continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return ece
```

```python
# src/evaluation/stats.py
import numpy as np
from scipy.stats import binom_test

def paired_bootstrap_ci(y_true, y_prob_a, y_prob_b,
                         metric_fn, n_resamples: int = 2000,
                         ci: float = 0.95) -> dict:
    n = len(y_true)
    deltas = []
    for _ in range(n_resamples):
        idx     = np.random.choice(n, n, replace=True)
        da = metric_fn(y_true[idx], y_prob_a[idx])
        db = metric_fn(y_true[idx], y_prob_b[idx])
        deltas.append(da - db)
    deltas  = np.array(deltas)
    alpha   = (1 - ci) / 2
    return {
        "mean_delta": float(np.mean(deltas)),
        "ci_low":     float(np.percentile(deltas, 100*alpha)),
        "ci_high":    float(np.percentile(deltas, 100*(1-alpha))),
        "p_value":    float(np.mean(deltas <= 0)),
    }
```

---

## Run logging

```python
# src/reporting/logger.py
import csv, os, json, datetime

FIELDS = ["run_id","model","dedup","split","seed",
          "macro_f1","f1_false","pr_auc_false","roc_auc",
          "brier","ece","threshold","notes"]

def log_run(path: str, row: dict):
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists: writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDS})

def save_manifest(run_id: str, config: dict, hashes: dict, outdir: str = "runs/manifests"):
    os.makedirs(outdir, exist_ok=True)
    manifest = {
        "run_id":    run_id,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "hashes":    hashes,
        "config":    config,
    }
    with open(f"{outdir}/{run_id}.json", "w") as f:
        json.dump(manifest, f, indent=2)
```

---

## Checkpoint policy

```python
# src/training/trainer.py  (checkpoint helper)
import torch, os

CKPT_KEYS = ["best_macro_f1", "best_f1_false", "best_pr_auc", "best_calib"]

class CheckpointManager:
    def __init__(self, run_id: str, base_dir: str = "checkpoints"):
        self.base   = base_dir
        self.run_id = run_id
        self.best   = {k: -1e9 for k in CKPT_KEYS}
        self.best["best_calib"] = 1e9   # lower is better

    def update(self, model, metrics: dict, epoch: int):
        mapping = {
            "best_macro_f1": ("macro_f1",    True),
            "best_f1_false": ("f1_false",    True),
            "best_pr_auc":   ("pr_auc_false",True),
            "best_calib":    ("brier",       False),  # lower is better
        }
        for key, (metric, higher_is_better) in mapping.items():
            val = metrics.get(metric, None)
            if val is None: continue
            improved = (val > self.best[key]) if higher_is_better \
                       else (val < self.best[key])
            if improved:
                self.best[key] = val
                path = os.path.join(self.base, key, self.run_id)
                os.makedirs(path, exist_ok=True)
                torch.save(model.state_dict(), f"{path}/epoch{epoch}.pt")
```

---

## Hyperparameter search ranges

| Component | Parameter | Range |
|---|---|---|
| Optimiser | learning rate | `1e-6` to `5e-4` (log-uniform) |
| Optimiser | batch size | 16 – 256 (grad accumulation) |
| Optimiser | epochs | 5 – 50 (early stopping, patience 5) |
| Regularisation | dropout | 0.0 – 0.4 |
| Regularisation | weight decay | 0.0 – 0.2 |
| Sequence | max tokens | 128 – 512 (default 256) |
| Approach A | λ1, λ2, λ3 (aux weights) | 0.05 – 1.0 |
| Approach B | α (contrastive weight) | 0.1 – 3.0 |
| Approach B | τ (temperature) | 0.03 – 0.2 |
| Approach C | k neighbours | 1 – 50 |
| Approach E | LoRA rank r | 4, 8, 16, 32, 64 |
| Approach F | ensemble size | 3 – 15 |
| Approach F | abstention threshold τ | 0.3 – 0.5 |

---

## CV summary table template (fill per run)

| Run ID | Model | Dedup | Split | Seeds | Macro-F1 | F1(False) | PR-AUC(False) | ROC-AUC | Brier ↓ | ECE ↓ | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| R001 | TF-IDF+LR | Y | GroupKFold(title) | 0,1,2 | | | | | | | |
| R002 | kNN vote | Y | GroupKFold(title) | 0,1,2 | | | | | | | |
| R003 | Transformer (scratch+TAPT) | Y | GroupKFold(title) | 0,1,2 | | | | | | | |
| R004 | Approach A | Y | GroupKFold(title) | 0,1,2 | | | | | | | |
| R005 | Approach B | Y | GroupKFold(title) | 0,1,2 | | | | | | | |
| R006 | Approach C | Y | GroupKFold(title) | 0,1,2 | | | | | | | |
| R007 | Approach D | Y | GroupKFold(title) | 0,1,2 | | | | | | | |
| R008 | Approach E | Y | GroupKFold(title) | 0,1,2 | | | | | | | |
| R009 | Ensemble F | Y | GroupKFold(title) | 0,1,2 | | | | | | | |

---

## Ablation checklist

- [ ] A1: Random split vs GroupKFold(title) — quantify leakage inflation
- [ ] A2: Dedup vs no-dedup — duplicate-driven overestimation
- [ ] A3: Keep instruction prefix vs strip it — template artefacts
- [ ] A4: Mask all numbers → measure F1 drop (numeracy dependence)
- [ ] A5: Mask all tickers → measure F1 drop (entity dependence)
- [ ] A6: Approach B — contrastive off (α=0)
- [ ] A7: Approach A — aux losses off (λ=0)
- [ ] A8: Calibration off (raw logit threshold vs calibrated)

---

## Final blind submission — one-shot protocol

```bash
# ONLY run after all modelling decisions are frozen
ALLOW_BLIND_EVAL=1 python scripts/99_final_blind_eval.py \
    --blind_path data/raw/blind_test.json \
    --model_dir  checkpoints/best_macro_f1/R009 \
    --output     results/submission/predictions.csv
```

The script must:
1. Load the frozen pipeline (dedup config, feature extractors, fold-independent).
2. Train on **full training set** (both JSONs, deduped, same preprocessing).
3. Open the blind file **exactly once** — guard logs the access.
4. Write `predictions.csv` with columns: `id, label, p_false`.
5. Write `results/submission/manifest_final.json` proving no prior blind access.

---

## Compliance statement

> **"All modelling, augmentation, pretraining (including TAPT), tuning, calibration, and analysis were performed using only the labelled training dataset (`misinfo_SFT_train_for_cot.json`, `misinfo_RL_train_for_cot.json`); the blind set was accessed exactly once at final submission time to generate predictions."**

---

## 8-week timeline

| Week | Deliverable | Acceptance criteria |
|---|---|---|
| 1 | Blind guard + dataset hashing + EDA stats + minimal-pair inventory | Guard raises on blind access; hashes recorded; stats table populated |
| 2 | Fold file exported + leakage unit tests + baseline leaderboard | Zero title overlap across folds; R001–R002 rows in run_log.csv |
| 3 | Fold-wise TAPT pipeline + scratch Transformer baseline | R003 logged; TAPT is fold-restricted |
| 4 | Approach A (numeric dual-stream) + robustness battery | R004 logged; perturbation delta curves produced |
| 5 | Approach B (contrastive) + minimal-pair benchmark | R005 logged; minimal-pair accuracy reported |
| 6 | Approach C (kNN memory) + Approach D (GNN prototype) | R006–R007 logged; leakage tests on kNN index pass |
| 7 | Approach E + Ensemble F + full ablation matrix + stats | All runs R001–R009 with CIs + Holm-Bonferroni adjusted p-values |
| 8 | Freeze decisions → final blind one-shot run + submission manifest | `predictions.csv` + `manifest_final.json` produced; audit log clean |