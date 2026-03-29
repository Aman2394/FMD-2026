"""
Script 12 — Direct LLM Fine-Tuning: Qwen2.5-7B + QLoRA (R012).

Fine-tunes Qwen2.5-7B-Instruct with QLoRA on the full article prompts.
Loss is computed only on the assistant response ("true" / "false").

GPU: ~9-11 GB VRAM on a 15 GB T4 (4-bit, batch=2, grad_accum=8).

Outputs:
  checkpoints/qwen_lora/R012/          LoRA adapter weights
  data/processed/qwen_val_probs.npy    val soft probabilities (for ensemble)
  runs/run_log.csv                     R012 row
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.guard import check_path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

from src.data.loader import sha256_file
from src.data.folds import load_split
from src.evaluation.metrics import compute_all
from src.reporting.logger import log_run, save_manifest
from src.utils import get_device

# ── paths ─────────────────────────────────────────────────────────────────────
SFT_PATH   = "data/raw/misinfo_SFT_train_for_cot.json"
RL_PATH    = "data/raw/misinfo_RL_train_for_cot.json"
FOLDS_PATH = "data/processed/folds.json"
PARQUET    = "data/processed/train_dedup.parquet"
RUN_LOG    = "runs/run_log.csv"
RUN_ID     = "R012"

# ── hyper-params ──────────────────────────────────────────────────────────────
MODEL_NAME      = "Qwen/Qwen2.5-7B-Instruct"
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
TARGET_MODULES  = ["q_proj", "k_proj", "v_proj", "o_proj"]
MAX_SEQ_LEN     = 512
BATCH_SIZE      = 2          # per-device; T4 15 GB safe with 4-bit
GRAD_ACCUM      = 8          # effective batch = 16
N_EPOCHS        = 3
LR              = 2e-4
WARMUP_RATIO    = 0.03
SEED            = 42

CKPT_DIR        = f"checkpoints/qwen_lora/{RUN_ID}"
DEVICE          = get_device()

SYSTEM_PROMPT = (
    "You are a financial misinformation detector.\n"
    "Please check whether the following information is true or false "
    "and output the answer [true/false]."
)
ASSISTANT_PREFIX = "The provided information is"


# ── helpers ───────────────────────────────────────────────────────────────────
def build_chat(row: dict, tokenizer, include_response: bool = True) -> str:
    label_str = "false." if row["label"] == 1 else "true."
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": row["claim_text"]},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if include_response:
        prompt += f"{ASSISTANT_PREFIX} {label_str}"
    else:
        prompt += ASSISTANT_PREFIX
    return prompt


def get_true_false_ids(tokenizer):
    """Return single-token IDs for 'true' and 'false' variants."""
    true_ids, false_ids = [], []
    for w in ["true", " true", "True", " True"]:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            true_ids.append(ids[0])
    for w in ["false", " false", "False", " False"]:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            false_ids.append(ids[0])
    if not true_ids or not false_ids:
        raise RuntimeError("Could not find single-token IDs for true/false.")
    return true_ids, false_ids


@torch.no_grad()
def evaluate(model, tokenizer, df: pd.DataFrame,
             true_ids, false_ids, batch_size: int = 4) -> dict:
    """Logit-based eval — no generation needed, reads first token distribution."""
    model.eval()
    probs_false, hard_labels = [], df["label"].values.tolist()

    rows = df.to_dict("records")
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        prompts = [build_chat(r, tokenizer, include_response=False) for r in batch]

        enc = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_SEQ_LEN,
        ).to(model.device)

        out         = model(**enc)
        next_logits = out.logits[:, -1, :]          # (B, vocab)

        for j in range(len(batch)):
            lg_true  = next_logits[j, true_ids].max().item()
            lg_false = next_logits[j, false_ids].max().item()
            p_false  = torch.softmax(
                torch.tensor([lg_true, lg_false]), dim=0
            )[1].item()
            probs_false.append(p_false)

    y_prob = np.array(probs_false)
    y_pred = (y_prob > 0.5).astype(int)
    y_true = np.array(hard_labels)
    return compute_all(y_true, y_pred, y_prob), y_prob


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    hashes = {"SFT": sha256_file(SFT_PATH), "RL": sha256_file(RL_PATH)}
    df = pd.read_parquet(PARQUET)
    df = load_split(df, folds_path=FOLDS_PATH)

    train = df[df["split"] == "train"].reset_index(drop=True)
    val   = df[df["split"] == "val"].reset_index(drop=True)
    test  = df[df["split"] == "test"].reset_index(drop=True)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    # ── tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "right"   # needed for SFT loss masking

    # ── format training data ─────────────────────────────────────────────────
    train_texts = [build_chat(r, tokenizer, include_response=True)
                   for r in train.to_dict("records")]
    hf_train = Dataset.from_dict({"text": train_texts})

    # ── model (4-bit QLoRA) ───────────────────────────────────────────────────
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_cfg,
        device_map="auto",
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        task_type     = TaskType.CAUSAL_LM,
        r             = LORA_R,
        lora_alpha    = LORA_ALPHA,
        lora_dropout  = LORA_DROPOUT,
        target_modules= TARGET_MODULES,
        bias          = "none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── SFT training ──────────────────────────────────────────────────────────
    sft_cfg = SFTConfig(
        output_dir                  = CKPT_DIR,
        num_train_epochs            = N_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = LR,
        warmup_ratio                = WARMUP_RATIO,
        lr_scheduler_type           = "cosine",
        fp16                        = True,
        logging_steps               = 10,
        save_strategy               = "epoch",
        seed                        = SEED,
        max_seq_length              = MAX_SEQ_LEN,
        dataset_text_field          = "text",
        report_to                   = "none",
    )

    trainer = SFTTrainer(
        model         = model,
        args          = sft_cfg,
        train_dataset = hf_train,
    )

    print("\nStarting QLoRA fine-tuning...")
    trainer.train()

    # ── save LoRA adapters ────────────────────────────────────────────────────
    os.makedirs(CKPT_DIR, exist_ok=True)
    model.save_pretrained(CKPT_DIR)
    tokenizer.save_pretrained(CKPT_DIR)
    print(f"LoRA adapters saved → {CKPT_DIR}")

    # ── evaluate ─────────────────────────────────────────────────────────────
    true_ids, false_ids = get_true_false_ids(tokenizer)

    print("\nEvaluating on val...")
    m_val,  y_prob_v = evaluate(model, tokenizer, val,  true_ids, false_ids)
    print("\nEvaluating on test...")
    m_test, y_prob_t = evaluate(model, tokenizer, test, true_ids, false_ids)

    os.makedirs("runs/oof", exist_ok=True)
    np.save(f"runs/oof/{RUN_ID}_seed{SEED}_prob.npy", y_prob_v)
    np.save(f"runs/oof/{RUN_ID}_seed{SEED}_true.npy", val["label"].values)
    np.save("data/processed/qwen_val_probs.npy", y_prob_v)

    print(f"\nval  macro_f1={m_val['macro_f1']:.4f}  f1_false={m_val['f1_false']:.4f}")
    print(f"test macro_f1={m_test['macro_f1']:.4f}  f1_false={m_test['f1_false']:.4f}")

    log_run(RUN_LOG, {
        "run_id": f"{RUN_ID}_seed{SEED}",
        "model":  f"Qwen2.5-7B QLoRA r={LORA_R}",
        "dedup": "Y", "split": "stratified", "seed": SEED,
        **{k: f"{v:.4f}" for k, v in m_val.items()},
        "notes": (f"QLoRA r={LORA_R} epochs={N_EPOCHS} "
                  f"test_f1={m_test['macro_f1']:.4f}"),
    })
    save_manifest(RUN_ID, config={
        "model": MODEL_NAME, "lora_r": LORA_R, "epochs": N_EPOCHS,
        "max_seq_len": MAX_SEQ_LEN,
    }, hashes=hashes)


if __name__ == "__main__":
    main()
