"""
Approach D — Heterogeneous Graph Neural Network.

Builds an entity–event graph over claims, tickers, time tokens, and
numeric values. Two-layer HeteroConv (SAGEConv) produces claim representations
for classification.

Requires: torch-geometric
"""
import torch
import torch.nn as nn

try:
    from torch_geometric.nn import HeteroConv, SAGEConv
    from torch_geometric.data import HeteroData
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


class FinancialClaimGNN(nn.Module):
    def __init__(self, metadata, hidden: int = 64, num_labels: int = 2):
        super().__init__()
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch-geometric is required for Approach D. "
                              "Install with: pip install torch-geometric")
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


def build_hetero_graph(df, text_embeddings: "np.ndarray") -> "HeteroData":
    """
    Construct a HeteroData graph from a dataframe of claims.

    Node types: claim, ticker, time_token, number_bucket
    Edge types:
      (claim, mentions_ticker, ticker)
      (claim, mentions_time,   time_token)
      (claim, mentions_number, number_bucket)
      (claim, co_occurs,       claim)   — same group_title

    Args:
        df: DataFrame with columns claim_text, tickers, time_tokens,
            numbers, group_title, label.
        text_embeddings: (N, D) numpy array of sentence embeddings.
    """
    import numpy as np
    import torch

    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("torch-geometric is required for Approach D.")

    data = HeteroData()

    # Claim nodes
    data["claim"].x     = torch.tensor(text_embeddings, dtype=torch.float)
    data["claim"].y     = torch.tensor(df["label"].values, dtype=torch.long)

    # Build ticker vocabulary
    all_tickers = sorted({t for tickers in df["tickers"] for t in tickers})
    ticker_idx  = {t: i for i, t in enumerate(all_tickers)}
    n_tickers   = max(len(all_tickers), 1)
    data["ticker"].x = torch.zeros(n_tickers, 8)   # small embedding dim

    # Build time-token vocabulary
    all_times = sorted({t for times in df["time_tokens"] for t in times})
    time_idx  = {t: i for i, t in enumerate(all_times)}
    n_times   = max(len(all_times), 1)
    data["time_token"].x = torch.zeros(n_times, 8)

    # Claim → ticker edges
    c_ticker_src, c_ticker_dst = [], []
    for ci, row in enumerate(df.itertuples(index=False)):
        for t in row.tickers:
            if t in ticker_idx:
                c_ticker_src.append(ci)
                c_ticker_dst.append(ticker_idx[t])
    if c_ticker_src:
        data["claim", "mentions_ticker", "ticker"].edge_index = torch.tensor(
            [c_ticker_src, c_ticker_dst], dtype=torch.long
        )

    # Claim → time_token edges
    c_time_src, c_time_dst = [], []
    for ci, row in enumerate(df.itertuples(index=False)):
        for t in row.time_tokens:
            if t in time_idx:
                c_time_src.append(ci)
                c_time_dst.append(time_idx[t])
    if c_time_src:
        data["claim", "mentions_time", "time_token"].edge_index = torch.tensor(
            [c_time_src, c_time_dst], dtype=torch.long
        )

    # Claim co-occurrence edges (same group_title)
    title_to_claims: dict[str, list] = {}
    for ci, row in enumerate(df.itertuples(index=False)):
        title_to_claims.setdefault(row.group_title, []).append(ci)

    co_src, co_dst = [], []
    for claims in title_to_claims.values():
        for i in claims:
            for j in claims:
                if i != j:
                    co_src.append(i)
                    co_dst.append(j)
    if co_src:
        data["claim", "co_occurs", "claim"].edge_index = torch.tensor(
            [co_src, co_dst], dtype=torch.long
        )

    return data
