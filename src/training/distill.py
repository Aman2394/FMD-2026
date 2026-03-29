"""
Knowledge distillation utilities.

Teacher (Qwen2.5 zero-shot) → soft labels → Student (FinBERT) fine-tuning.
"""
import torch
import torch.nn.functional as F
import numpy as np


def kd_loss(student_logits: torch.Tensor,
            teacher_probs: torch.Tensor,
            hard_labels: torch.Tensor,
            alpha: float = 0.5,
            temperature: float = 2.0) -> torch.Tensor:
    """
    α · CE(student, hard_label) + (1-α) · T² · KL(teacher ‖ student)

    Args:
        student_logits: (B, 2) raw logits from student
        teacher_probs:  (B, 2) soft probabilities from teacher [P(true), P(false)]
        hard_labels:    (B,)   ground-truth integer labels
        alpha:          weight for hard-label CE (0 = pure distillation)
        temperature:    softening temperature for KL term
    """
    ce = F.cross_entropy(student_logits, hard_labels)

    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    log_teacher  = torch.log(teacher_probs.clamp(min=1e-8))
    soft_teacher = F.softmax(log_teacher / temperature, dim=-1)
    kl = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)

    return alpha * ce + (1.0 - alpha) * kl


def get_teacher_probs(texts: list[str],
                      model,
                      tokenizer,
                      device: str,
                      batch_size: int = 4,
                      max_length: int = 1024) -> np.ndarray:
    """
    Zero-shot inference with a causal LM. Returns P(false/misinfo) for each text.

    Strategy: prefix the assistant turn with "The provided information is" and read
    the logits of the very next token. The token with the highest logit among the
    {"true", "false"} vocabulary entries determines the soft label.
    """
    model.eval()

    # Collect single-token IDs for "true" / "false" variants
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
        raise RuntimeError(
            "Could not find single-token IDs for 'true'/'false'. "
            "Check tokenizer vocabulary."
        )

    SYSTEM = (
        "You are a financial misinformation detector.\n"
        "Please check whether the following information is true or false "
        "and output the answer [true/false]."
    )
    ASSISTANT_PREFIX = "The provided information is"

    probs_false = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        prompts = []
        for text in batch:
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": text},
            ]
            p = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ) + ASSISTANT_PREFIX
            prompts.append(p)

        enc = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)
        next_logits = out.logits[:, -1, :]  # (B, vocab_size)

        for j in range(len(batch)):
            lg_true  = next_logits[j, true_ids].max().item()
            lg_false = next_logits[j, false_ids].max().item()
            p_false  = torch.softmax(
                torch.tensor([lg_true, lg_false]), dim=0
            )[1].item()
            probs_false.append(p_false)

        done = i + len(batch)
        if done % (batch_size * 20) == 0 or done == len(texts):
            print(f"  Teacher inference: {done}/{len(texts)}")

    return np.array(probs_false, dtype=np.float32)
