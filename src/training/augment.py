import random

from src.data.features import NUMBER_RE, TICKER_RE, TIME_RE


def numeric_swap(text: str, num_inventory: list[str], rate: float = 0.5) -> str:
    """Replace a random numeric span with a same-unit token from training inventory."""
    spans = [(m.start(), m.end(), m.group()) for m in NUMBER_RE.finditer(text)]
    if not spans:
        return text
    for start, end, _ in random.sample(spans, k=max(1, int(len(spans) * rate))):
        replacement = random.choice(num_inventory)
        text = text[:start] + replacement + text[end:]
    return text


def ticker_swap(text: str, ticker_inventory: list[str]) -> str:
    spans = [(m.start(), m.end()) for m in TICKER_RE.finditer(text)]
    if not spans:
        return text
    start, end = random.choice(spans)
    return text[:start] + random.choice(ticker_inventory) + text[end:]


def temporal_swap(text: str, time_inventory: list[str]) -> str:
    spans = [(m.start(), m.end()) for m in TIME_RE.finditer(text)]
    if not spans:
        return text
    start, end = random.choice(spans)
    return text[:start] + random.choice(time_inventory) + text[end:]


def prefix_dropout(prompt: str, rate: float = 0.3) -> str:
    """Randomly drop the instruction prefix to reduce template overfitting."""
    if random.random() < rate:
        lines = prompt.splitlines()
        for i, line in enumerate(lines):
            if line.strip() and i >= 2:
                return "\n".join(lines[i:]).strip()
    return prompt
