"""
Approach E — LoRA Instruction Classifier.

Fine-tunes a pre-trained sequence classification model using
Low-Rank Adaptation (LoRA) via the peft library.
Falls back to QLoRA (4-bit) when VRAM is limited.
"""
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

try:
    from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def make_lora_classifier(
    model_name: str,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: list[str] = None,
    use_qlora: bool = False,
):
    """
    Build a LoRA-adapted sequence classifier.

    Args:
        model_name:     HuggingFace model identifier.
        r:              LoRA rank (4, 8, 16, 32, 64).
        lora_alpha:     LoRA scaling factor.
        lora_dropout:   Dropout applied to LoRA layers.
        target_modules: Modules to apply LoRA to. Defaults to ["query", "value"].
        use_qlora:      Load in 4-bit (QLoRA) for reduced VRAM. Requires bitsandbytes.
    """
    if not PEFT_AVAILABLE:
        raise ImportError("peft is required for Approach E. Install with: pip install peft")

    if target_modules is None:
        target_modules = ["query", "value"]

    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )

    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    return get_peft_model(model, config)


def print_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
