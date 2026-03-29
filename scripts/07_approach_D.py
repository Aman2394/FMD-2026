"""
Script 07 — Approach D: Heterogeneous GNN (R007).
Requires: torch-geometric, sentence-transformers (optional).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.guard import check_path

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

from src.data.loader import sha256_file
from src.data.folds import load_split
from src.models.approach_D import FinancialClaimGNN, build_hetero_graph
from src.evaluation.metrics import compute_all
from src.reporting.logger import log_run, save_manifest
from src.utils import get_device

SFT_PATH   = "data/raw/misinfo_SFT_train_for_cot.json"
RL_PATH    = "data/raw/misinfo_RL_train_for_cot.json"
FOLDS_PATH = "data/processed/folds.json"
PARQUET    = "data/processed/train_dedup.parquet"
RUN_LOG    = "runs/run_log.csv"
RUN_ID     = "R007"
HIDDEN     = 64
N_EPOCHS   = 50
LR         = 1e-3
SEEDS      = [0, 1, 2]
DEVICE     = get_device()


def get_embeddings(texts):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2").encode(texts, show_progress_bar=False)
    except ImportError:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        X = TfidfVectorizer(max_features=10_000, sublinear_tf=True).fit_transform(texts)
        return TruncatedSVD(n_components=64).fit_transform(X)


def train_gnn(data, train_mask, val_mask, test_mask, seed):
    torch.manual_seed(seed)
    model = FinancialClaimGNN(data.metadata(), hidden=HIDDEN).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)
    data = data.to(DEVICE)

    best_val_f1, best_test_metrics = 0.0, {}
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out  = model(data.x_dict, data.edge_index_dict, torch.where(train_mask)[0])
        loss = torch.nn.CrossEntropyLoss()(out, data["claim"].y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                for mask, name in [(val_mask, "val"), (test_mask, "test")]:
                    out_s = model(data.x_dict, data.edge_index_dict, torch.where(mask)[0])
                    probs = torch.softmax(out_s, dim=-1)[:, 1].cpu().numpy()
                    preds = out_s.argmax(dim=-1).cpu().numpy()
                    y     = data["claim"].y[mask].cpu().numpy()
                    m     = compute_all(y, preds, probs)
                    if name == "val" and m["macro_f1"] > best_val_f1:
                        best_val_f1 = m["macro_f1"]
                        best_test_metrics = compute_all(
                            data["claim"].y[test_mask].cpu().numpy(),
                            model(data.x_dict, data.edge_index_dict,
                                  torch.where(test_mask)[0]).argmax(-1).cpu().numpy(),
                            torch.softmax(model(data.x_dict, data.edge_index_dict,
                                  torch.where(test_mask)[0]), -1)[:, 1].cpu().numpy(),
                        )
    return best_val_f1, best_test_metrics


def main():
    hashes = {"SFT": sha256_file(SFT_PATH), "RL": sha256_file(RL_PATH)}
    df = pd.read_parquet(PARQUET)
    df = load_split(df, folds_path=FOLDS_PATH)
    print(f"Getting embeddings for {len(df)} claims...")
    embeddings = get_embeddings(df["claim_text"].tolist())

    train_mask = torch.tensor(df["split"].values == "train")
    val_mask   = torch.tensor(df["split"].values == "val")
    test_mask  = torch.tensor(df["split"].values == "test")
    data = build_hetero_graph(df, embeddings)

    seed_val_f1s = []
    for seed in SEEDS:
        print(f"\n=== Approach D | Seed {seed} ===")
        val_f1, m_test = train_gnn(data, train_mask, val_mask, test_mask, seed)
        seed_val_f1s.append(val_f1)
        print(f"  best val macro_f1={val_f1:.4f}  test macro_f1={m_test.get('macro_f1', 0):.4f}")
        log_run(RUN_LOG, {
            "run_id": f"{RUN_ID}_seed{seed}", "model": "Approach D (GNN)",
            "dedup": "Y", "split": "stratified", "seed": seed,
            "macro_f1": f"{val_f1:.4f}",
            "notes": f"HeteroConv SAGEConv test_f1={m_test.get('macro_f1', 0):.4f}",
        })

    save_manifest(RUN_ID, config={"model": "Approach D", "hidden": HIDDEN, "seeds": SEEDS}, hashes=hashes)
    print(f"\nR007 avg val macro_f1={np.mean(seed_val_f1s):.4f}")


if __name__ == "__main__":
    main()
