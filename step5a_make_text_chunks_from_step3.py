from pathlib import Path
import re
from step3_normalize import build_normalized_blocks

OUT_DIR = Path("text_chunks")
OUT_DIR.mkdir(exist_ok=True)

MAX_CHARS = 1800  # safe for CPU; increase later if needed

def clean_for_tts(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)      # remove tags if any
    text = re.sub(r"\s+", " ", text).strip()  # collapse spaces
    return text

def chunk_blocks(blocks):
    chunks = []
    cur = ""

    for b in blocks:
        t = clean_for_tts(b["text"])
        if not t:
            continue

        # headings get extra spacing to simulate pause (no SSML)
        if b["type"] == "heading":
            t = f"\n\n{t}\n\n"
        else:
            t = t + "\n\n"

        if len(cur) + len(t) > MAX_CHARS and cur.strip():
            chunks.append(cur.strip())
            cur = t
        else:
            cur += t

    if cur.strip():
        chunks.append(cur.strip())

    return chunks

def main():
    blocks = build_normalized_blocks(max_pages=None)  # FULL PDF
    chunks = chunk_blocks(blocks)

    for i, c in enumerate(chunks, start=1):
        (OUT_DIR / f"chunk_{i:04d}.txt").write_text(c, encoding="utf-8")

    print(f"✅ Normalized blocks: {len(blocks)}")
    print(f"✅ Text chunks created: {len(chunks)} in {OUT_DIR}/")

if __name__ == "__main__":
    main()
