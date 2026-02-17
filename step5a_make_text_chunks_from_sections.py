from pathlib import Path
import argparse
import re
import json

MAX_CHARS_DEFAULT = 1800

def clean_for_tts(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)      # remove tags if any
    text = re.sub(r"\s+", " ", text).strip()  # collapse spaces
    return text

def chunk_text(text: str, max_chars: int):
    """
    Simple chunker: splits by paragraphs, packs into max_chars blocks.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    cur = ""

    for p in paras:
        p = clean_for_tts(p)
        if not p:
            continue
        p = p + "\n\n"
        if len(cur) + len(p) > max_chars and cur.strip():
            chunks.append(cur.strip())
            cur = p
        else:
            cur += p

    if cur.strip():
        chunks.append(cur.strip())
    return chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="JSON list of section txt file paths")
    ap.add_argument("--out_dir", required=True, help="Output folder for chunk_*.txt")
    ap.add_argument("--max_chars", type=int, default=MAX_CHARS_DEFAULT)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_paths = json.loads(args.inputs)
    all_text = []

    for p in input_paths:
        fp = Path(p)
        if not fp.exists():
            raise FileNotFoundError(f"Section file not found: {fp}")
        txt = fp.read_text(encoding="utf-8").strip()
        if txt:
            # add a clear boundary between sections
            all_text.append(txt)

    combined = "\n\n".join(all_text).strip()
    if not combined:
        raise ValueError("All selected sections are empty. Nothing to synthesize.")

    chunks = chunk_text(combined, max_chars=args.max_chars)

    for i, c in enumerate(chunks, start=1):
        (out_dir / f"chunk_{i:04d}.txt").write_text(c, encoding="utf-8")

    print(f"✅ Selected sections: {len(input_paths)}")
    print(f"✅ Text chunks created: {len(chunks)} in {out_dir}")

if __name__ == "__main__":
    main()
