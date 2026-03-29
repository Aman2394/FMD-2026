"""
Approach B — Contrastive Counterfactual Learning.

Combines standard cross-entropy with supervised contrastive loss (SupCon).
Minimal pairs (same title, different numeric values, opposite labels)
are prioritised during batch sampling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: (B, D) — will be L2-normalised; labels: (B,)
        features = F.normalize(features, dim=-1)
        sim      = torch.matmul(features, features.T) / self.temperature
        mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask_self = torch.eye(len(labels), device=labels.device)
        mask_pos  = mask_pos - mask_self
        exp_sim   = torch.exp(sim) * (1 - mask_self)
        log_prob  = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        loss      = -(mask_pos * log_prob).sum(dim=1) / (mask_pos.sum(dim=1) + 1e-8)
        return loss.mean()


class ContrastiveClassifier(nn.Module):
    def __init__(self, encoder, num_labels: int = 2,
                 alpha: float = 0.5, dropout: float = 0.1):
        super().__init__()
        self.encoder    = encoder
        self.classifier = nn.Linear(encoder.config.hidden_size, num_labels)
        self.supcon     = SupConLoss()
        self.alpha      = alpha
        self.dropout    = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, labels=None):
        h      = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0]
        h      = self.dropout(h)
        logits = self.classifier(h)
        output = {"logits": logits, "embeddings": h}

        if labels is not None:
            weight        = torch.tensor([1.0, 992 / 825]).to(labels.device)
            loss_cls      = nn.CrossEntropyLoss(weight=weight)(logits, labels)
            loss_con      = self.supcon(h, labels)
            output["loss"] = loss_cls + self.alpha * loss_con
        return output


def minimal_pair_sampler(df, batch_size: int = 32, pair_ratio: float = 0.5):
    """
    Yields batches that mix random samples with minimal pairs.

    Args:
        df: DataFrame with columns claim_text, label, group_title.
            Must include minimal pairs (same group_title, different labels).
        batch_size: Total batch size.
        pair_ratio: Fraction of the batch drawn from minimal pairs.
    """
    import random
    import numpy as np

    # Build minimal-pair index: group_title → {0: [indices], 1: [indices]}
    pair_groups = {}
    for title, grp in df.groupby("group_title"):
        classes = grp["label"].unique()
        if len(classes) == 2:
            pair_groups[title] = {
                cls: grp[grp["label"] == cls].index.tolist()
                for cls in classes
            }

    n_pairs   = int(batch_size * pair_ratio) // 2 * 2   # even number
    n_random  = batch_size - n_pairs
    all_idx   = df.index.tolist()

    while True:
        batch_idx = []

        # Sample from minimal pairs
        titles = list(pair_groups.keys())
        random.shuffle(titles)
        for title in titles:
            if len(batch_idx) >= n_pairs:
                break
            g = pair_groups[title]
            pos = random.choice(g[1])
            neg = random.choice(g[0])
            batch_idx.extend([pos, neg])

        # Fill remainder randomly
        batch_idx += random.choices(all_idx, k=n_random)
        random.shuffle(batch_idx)
        yield df.loc[batch_idx]
