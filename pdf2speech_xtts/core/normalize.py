import regex as re
from typing import Dict

def normalize_text(text: str, opts: Dict[str, bool]) -> str:
    out = text

    if opts.get("fix_hyphen_linebreaks", True):
        out = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", out)

    if opts.get("collapse_whitespace", True):
        out = re.sub(r"[ \t]+", " ", out)
        out = re.sub(r"\n{3,}", "\n\n", out)

    if opts.get("normalize_punct", True):
        out = out.replace("…", "...")
        out = re.sub(r"\s+([,.;:!?])", r"\1", out)
        out = re.sub(r"([,.;:!?])([A-Za-z])", r"\1 \2", out)

    return out.strip()
