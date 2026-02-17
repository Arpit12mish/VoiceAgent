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
from pathlib import Path
import re
import json

# ----------------------------
# Config
# ----------------------------

OUT_DIR = Path("text_chunks")
OUT_DIR.mkdir(exist_ok=True)

MAX_CHARS = 1800  # tune later (XTTS can take more, keep safe for CPU)

# Choose source:
#   SECTIONS_MODE=original  -> sections_original/
#   SECTIONS_MODE=user      -> sections_user/
SECTIONS_MODE = (Path(".") / ".env").exists()  # noop, just to avoid lint noise

SECTIONS_ORIGINAL_DIR = Path("sections_original")
SECTIONS_USER_DIR = Path("sections_user")

# If you want quick switching without env vars, set this:
# SOURCE_DIR = SECTIONS_ORIGINAL_DIR
# or
# SOURCE_DIR = SECTIONS_USER_DIR


# ----------------------------
# Helpers
# ----------------------------

def clean_for_tts(text: str) -> str:
    """Remove tags, collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)      # remove tags if any
    text = re.sub(r"\s+", " ", text).strip()
    return text

def choose_source_dir() -> Path:
    """
    Priority:
      1) sections_user/ if it exists and has .txt files
      2) else sections_original/
    """
    if SECTIONS_USER_DIR.exists() and any(SECTIONS_USER_DIR.glob("*.txt")):
        return SECTIONS_USER_DIR
    if SECTIONS_ORIGINAL_DIR.exists() and any(SECTIONS_ORIGINAL_DIR.glob("*.txt")):
        return SECTIONS_ORIGINAL_DIR
    raise FileNotFoundError(
        "No sections found.\n"
        "Expected one of:\n"
        " - sections_original/*.txt  (run: python step2c_sections.py)\n"
        " - sections_user/*.txt      (generated after Streamlit edits)\n"
    )

def get_ordered_section_files(source_dir: Path):
    """
    Your section files are like: 00_00_front_matter.txt, 01_1_scope.txt, ...
    We sort by numeric prefix to keep stable order.
    """
    files = sorted(source_dir.glob("*.txt"))

    def key(p: Path):
        m = re.match(r"^(\d+)_", p.name)
        return int(m.group(1)) if m else 999999

    return sorted(files, key=key)

def chunk_text(text: str, max_chars: int):
    """
    Chunk long text into <= max_chars, trying to split on paragraph boundaries first.
    """
    text = text.strip()
    if not text:
        return []

    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks = []
    cur = ""

    for p in paras:
        p = clean_for_tts(p)
        if not p:
            continue
        piece = p + "\n\n"

        if len(cur) + len(piece) > max_chars and cur.strip():
            chunks.append(cur.strip())
            cur = piece
        else:
            cur += piece

    if cur.strip():
        chunks.append(cur.strip())

    return chunks

def write_chunk_file(idx: int, text: str):
    (OUT_DIR / f"chunk_{idx:04d}.txt").write_text(text, encoding="utf-8")

def write_manifest(items):
    """
    Optional: Manifest to show mapping: chunk -> section file & title.
    Helpful for UI/debug.
    """
    Path("text_chunks_manifest.json").write_text(
        json.dumps(items, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


# ----------------------------
# Main
# ----------------------------

def main():
    source_dir = choose_source_dir()
    section_files = get_ordered_section_files(source_dir)

    # Fresh output (optional): uncomment if you want to clear old chunks
    # for f in OUT_DIR.glob("chunk_*.txt"):
    #     f.unlink()

    chunk_index = 1
    manifest = []

    print(f"✅ Using sections from: {source_dir}/")
    print(f"✅ Sections found: {len(section_files)}")

    for sec_path in section_files:
        raw = sec_path.read_text(encoding="utf-8", errors="ignore")
        raw = raw.strip()
        if not raw:
            continue

        # Add a "section header" at the top of each section for better narration separation
        # (keeps plain text; XTTS/VITS can pause naturally with blank lines)
        section_title = sec_path.stem  # e.g. "01_1_scope"
        header = f"\n\n=== {section_title} ===\n\n"
        text = header + raw

        pieces = chunk_text(text, MAX_CHARS)

        for local_i, piece in enumerate(pieces, start=1):
            write_chunk_file(chunk_index, piece)

            manifest.append({
                "chunk": f"chunk_{chunk_index:04d}.txt",
                "source_section_file": sec_path.as_posix(),
                "source_dir": source_dir.as_posix(),
                "section_piece_index": local_i,
                "section_piece_total": len(pieces),
                "chars": len(piece),
                "words": len(piece.split())
            })

            chunk_index += 1

    write_manifest(manifest)

    print(f"\n✅ Text chunks created: {chunk_index - 1} in {OUT_DIR}/")
    print("✅ Chunk manifest: text_chunks_manifest.json")


if __name__ == "__main__":
    main()
