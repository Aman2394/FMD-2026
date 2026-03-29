"""
Fold-wise Task-Adaptive Pre-Training (TAPT) via masked language modeling.

Each fold's TAPT model is trained ONLY on that fold's training split,
ensuring no data leakage from validation into the pretrained representations.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils import get_device
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)


class TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int = 256):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def run_tapt(
    texts: list[str],
    fold_idx: int,
    run_id: str,
    model_name: str = "distilbert-base-uncased",
    max_length: int = 256,
    mlm_probability: float = 0.15,
    n_epochs: int = 3,
    batch_size: int = 32,
    lr: float = 5e-5,
    warmup_ratio: float = 0.1,
    output_dir: str = "checkpoints",
    device: str = None,
) -> str:
    """
    Train a masked LM on `texts` (training fold only).
    Returns the path to the saved TAPT checkpoint.
    """
    if device is None:
        device = get_device()
    print(f"TAPT fold {fold_idx}: {len(texts)} texts on {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForMaskedLM.from_pretrained(model_name).to(device)

    dataset   = TextDataset(texts, tokenizer, max_length)
    collator  = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
    )
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps   = len(loader) * n_epochs
    warmup_steps  = int(total_steps * warmup_ratio)
    scheduler     = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    model.train()
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(**batch)
            loss  = out.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        print(f"  Epoch {epoch}: mlm_loss={epoch_loss / len(loader):.4f}")

    save_path = os.path.join(output_dir, "tapt", run_id, f"fold{fold_idx}")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"  TAPT checkpoint saved to {save_path}")
    return save_path
