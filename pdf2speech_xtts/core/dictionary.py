import regex as re
from typing import List, Dict, Any

def default_rules():
    return [
        {"find": "BIS", "replace": "B I S", "match_type": "word", "case_sensitive": False, "enabled": True},
        {"find": "IEC", "replace": "I E C", "match_type": "word", "case_sensitive": False, "enabled": True},
    ]

def apply_dictionary(text: str, rules: List[Dict[str, Any]]) -> str:
    out = text
    for r in rules:
        if not r.get("enabled", True):
            continue
        find = r.get("find", "")
        repl = r.get("replace", "")
        mt = r.get("match_type", "exact")
        cs = bool(r.get("case_sensitive", False))
        flags = 0 if cs else re.IGNORECASE

        if mt == "word":
            out = re.sub(rf"\b{re.escape(find)}\b", repl, out, flags=flags)
        elif mt == "regex":
            out = re.sub(find, repl, out, flags=flags)
        else:
            out = re.sub(re.escape(find), repl, out, flags=flags)
    return out
