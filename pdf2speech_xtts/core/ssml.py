import html
import regex as re

def to_ssml_preview(text: str, break_ms: int = 250) -> str:
    safe = html.escape(text)
    safe = re.sub(r"(\n\s*\n)+", rf'</s><break time="{break_ms}ms"/><s>', safe)
    safe = safe.replace("\n", " ")
    return f"<speak><s>{safe}</s></speak>"
