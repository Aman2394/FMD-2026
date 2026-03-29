"""
Approach A — Numeric Dual-Stream Transformer.

Fuses a text CLS embedding with a structured numeric feature vector
via a learned gating mechanism. Three auxiliary heads regularise
the numeric stream: magnitude bucket, unit type, and internal consistency.
"""
import torch
import torch.nn as nn


class NumericEncoder(nn.Module):
    """Encodes a fixed-size numeric feature vector per claim."""
    def __init__(self, input_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x):       # x: (B, input_dim)
        return self.net(x)      # (B, hidden)


class DualStreamClassifier(nn.Module):
    def __init__(self, text_encoder, numeric_dim: int = 16,
                 numeric_hidden: int = 64, num_labels: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.text_enc   = text_encoder
        self.num_enc    = NumericEncoder(numeric_dim, numeric_hidden)
        text_hidden     = text_encoder.config.hidden_size
        fused_dim       = text_hidden + numeric_hidden

        self.gate       = nn.Linear(fused_dim, fused_dim)
        self.classifier = nn.Linear(fused_dim, num_labels)
        self.aux_masknum= nn.Linear(fused_dim, 10)   # magnitude bucket
        self.aux_unit   = nn.Linear(fused_dim, 5)    # unit type
        self.aux_cons   = nn.Linear(fused_dim, 2)    # internal consistency pseudo-label
        self.dropout    = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, numeric_feats,
                labels=None, masknum_labels=None,
                unit_labels=None, cons_labels=None):
        text_out = self.text_enc(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0]
        num_out  = self.num_enc(numeric_feats)
        fused    = torch.cat([text_out, num_out], dim=-1)
        fused    = fused * torch.sigmoid(self.gate(fused))
        fused    = self.dropout(fused)

        logits   = self.classifier(fused)
        output   = {"logits": logits}

        if labels is not None:
            weight    = torch.tensor([1.0, 992 / 825]).to(labels.device)
            loss_cls  = nn.CrossEntropyLoss(weight=weight)(logits, labels)
            loss_mn   = nn.CrossEntropyLoss()(self.aux_masknum(fused), masknum_labels) \
                        if masknum_labels is not None else 0.0
            loss_unit = nn.CrossEntropyLoss()(self.aux_unit(fused), unit_labels) \
                        if unit_labels is not None else 0.0
            loss_cons = nn.CrossEntropyLoss()(self.aux_cons(fused), cons_labels) \
                        if cons_labels is not None else 0.0
            output["loss"] = loss_cls + 0.3 * loss_mn + 0.2 * loss_unit + 0.1 * loss_cons
        return output


def build_numeric_features(df_row) -> list[float]:
    """
    Build a 16-dim numeric feature vector from a dataframe row.

    Dimensions:
      0: n_numbers (clipped to 15)
      1: n_tickers
      2: n_tokens_approx (log-scaled)
      3: n_units (len of units list)
      4: n_time_tokens
      5–15: reserved / zero-padded
    """
    import math
    feats = [0.0] * 16
    feats[0]  = min(float(df_row.get("n_numbers", 0)), 15.0)
    feats[1]  = min(float(df_row.get("n_tickers", 0)), 10.0)
    feats[2]  = math.log1p(float(df_row.get("n_tokens_approx", 0)))
    feats[3]  = float(len(df_row.get("units", [])))
    feats[4]  = float(len(df_row.get("time_tokens", [])))
    return feats
