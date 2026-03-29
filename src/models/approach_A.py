"""
Approach A — Numeric Dual-Stream Transformer.

Fine-tunes FinBERT (AutoModelForSequenceClassification) and adds a small
numeric-feature residual projection on top of the text logits.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


class DualStreamClassifier(nn.Module):
    def __init__(self, model_name: str = "ProsusAI/finbert",
                 numeric_dim: int = 16, num_labels: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.backbone  = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.num_proj  = nn.Linear(numeric_dim, num_labels)
        self.dropout   = nn.Dropout(dropout)
        # Zero-init so numeric stream starts as a no-op
        nn.init.zeros_(self.num_proj.weight)
        nn.init.zeros_(self.num_proj.bias)

    def forward(self, input_ids, attention_mask, numeric_feats, labels=None):
        out    = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits + self.num_proj(self.dropout(numeric_feats))
        output = {"logits": logits}

        if labels is not None:
            weight         = torch.tensor([1.0, 992 / 825]).to(labels.device)
            output["loss"] = nn.CrossEntropyLoss(weight=weight)(logits, labels)
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
