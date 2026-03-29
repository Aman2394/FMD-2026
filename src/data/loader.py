import json
import hashlib
import re

from src.data.guard import check_path


def load_training(path: str, source: str = None) -> list[dict]:
    """
    Load a training JSON file and return a list of records.

    Args:
        path:   Path to the JSON file.
        source: Human-readable source tag, e.g. "SFT" or "RL".
                Inferred from the filename if not provided.
    """
    check_path(path)
    if source is None:
        source = "SFT" if "SFT" in path else "RL"
    with open(path) as f:
        raw = json.load(f)
    records = []
    for item in raw:
        prompt = item["Open-ended Verifiable Question"]
        label  = _parse_label(item["Ground-True Answer"])
        claim  = _extract_claim(prompt)
        records.append({
            "index":       item["index"],
            "source":      source,
            "prompt_text": prompt,
            "claim_text":  claim,
            "label":       label,   # 1 = False (misinformation), 0 = True
            "claim_hash":  hashlib.sha1(claim.encode()).hexdigest(),
        })
    return records


def _parse_label(answer: str) -> int:
    """Parse label from 'The provided information is false/true.' format."""
    answer_lower = answer.strip().lower()
    if "false" in answer_lower:
        return 1
    return 0


_TRUNCATION_RE = re.compile(r"\s*(\[…\]|\[\.\.\.\]|…|\[continued\]|\(continued\))\s*$", re.I)


def _extract_claim(prompt: str) -> str:
    lines = prompt.splitlines()
    for i, line in enumerate(lines):
        if line.strip() and i >= 2:
            text = "\n".join(lines[i:]).strip()
            return _TRUNCATION_RE.sub("", text).strip()
    return prompt.strip()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()
