import json
from pathlib import Path
import re
import unicodedata

OUT_BLOCKS = Path("out_blocks.json")

# ✅ New: optional table explanations (user-written)
# Format:
# {
#   "p8_t0": "In this table, the standard defines permissible values for density, viscosity...",
#   "p9_t0": "..."
# }
TABLE_EXPLANATIONS = Path("table_explanations.json")

# ---------------------------
# Basic normalization (reuse)
# ---------------------------

LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
}

ABBREV = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.",
    "e.g.", "i.e.", "etc.", "vs.", "no.", "fig.", "eq.",
    "inc.", "ltd.", "pvt.", "co.", "dept.", "gov."
}

# ✅ Table token expected from step2_blocks.py placeholder:
# [[TABLE|p8_t0|Table 1 – Specifications for ...]]
TABLE_TOKEN_RE = re.compile(r"\[\[TABLE\|([^|]+)\|([^\]]+)\]\]")

def norm_unicode(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    for k, v in LIGATURES.items():
        s = s.replace(k, v)
    s = s.replace("\u00ad", "")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = s.replace("–", "-").replace("—", "-")
    return s

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def fix_decimal_spacing(s: str) -> str:
    return re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", s)

def remove_dot_leaders(s: str) -> str:
    s = re.sub(r"\.{5,}", " ", s)
    return normalize_ws(s)

def looks_like_page_marker(s: str) -> bool:
    t = s.strip().lower()
    if t in {"page"}:
        return True
    if re.fullmatch(r"[ivxlcdm]+", t) and len(t) <= 6:
        return True
    if re.fullmatch(r"\d{1,3}", t):
        return True
    return False

def is_boilerplate_line(s: str) -> bool:
    t = s.strip().lower()
    boiler = [
        "for bis use only",
        "copyright protected document",
        "all rights reserved",
        "© iso",
        "published in switzerland",
    ]
    return any(b in t for b in boiler)

# ---------------------------
# ✅ Table token helpers
# ---------------------------

def extract_table_tokens(text: str):
    """
    Returns list of (table_id, caption) found in text.
    """
    out = []
    for m in TABLE_TOKEN_RE.finditer(text or ""):
        out.append((m.group(1).strip(), m.group(2).strip()))
    return out

def replace_table_tokens(text: str, table_map: dict):
    """
    Replace each [[TABLE|id|caption]] with:
      - user explanation if present
      - otherwise a spoken placeholder prompt
    """
    def repl(m):
        table_id = m.group(1).strip()
        caption = m.group(2).strip()

        expl = (table_map.get(table_id) or "").strip()
        if expl:
            return f"{caption}. {expl}"

        # Default: prompt user to add explanation later
        return (
            f"{caption}. "
            f"This is a table. Please provide a short explanation for table id {table_id}."
        )

    return TABLE_TOKEN_RE.sub(repl, text or "")

def load_table_explanations():
    if TABLE_EXPLANATIONS.exists():
        try:
            data = json.loads(TABLE_EXPLANATIONS.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                # ensure str keys/values
                return {str(k): str(v) for k, v in data.items()}
        except Exception:
            pass
    return {}

# ---------------------------
# Normalize blocks
# ---------------------------

def normalize_blocks(blocks, table_map: dict):
    """
    Input: blocks from out_blocks.json
    Output: cleaned blocks (with table tokens replaced by explanation or prompt)
    """
    out = []
    for b in blocks:
        t = (b.get("text") or "")

        # ✅ 1) Replace table tokens BEFORE collapsing whitespace.
        # This ensures we don't destroy the token format accidentally.
        if "[[TABLE|" in t:
            t = replace_table_tokens(t, table_map)

        # Normal normalization
        t = norm_unicode(t)
        t = remove_dot_leaders(t)
        t = fix_decimal_spacing(t)
        t = normalize_ws(t)

        if not t:
            continue
        if looks_like_page_marker(t):
            continue
        if is_boilerplate_line(t):
            continue
        if len(t) <= 2:
            continue

        out.append({**b, "text": t})
    return out

# ---------------------------
# Step 4.1: TOC detection
# ---------------------------

TOC_START_TITLES = {"contents", "table of contents"}

def is_toc_index_line(text: str) -> bool:
    return bool(re.fullmatch(r"\d+(\.\d+)+", text.strip()))

def looks_like_toc_entry(text: str) -> bool:
    t = text.strip()
    return bool(re.search(r"\s(\d{1,3}|[ivxlcdm]{1,6})$", t.lower()))

def drop_toc(blocks):
    out = []
    in_toc = False
    toc_start_page = None

    for b in blocks:
        txt = b["text"].strip()
        low = txt.lower()

        if b["type"] == "heading" and low in TOC_START_TITLES:
            in_toc = True
            toc_start_page = b.get("page")
            continue

        if in_toc:
            if toc_start_page is not None and b.get("page") is not None and b["page"] > toc_start_page + 2:
                in_toc = False
            else:
                if is_toc_index_line(txt) or looks_like_toc_entry(txt):
                    continue
                if len(txt.split()) >= 12:
                    in_toc = False
                else:
                    continue

        out.append(b)

    return out

# ---------------------------
# Step 4.2: Sentence splitting
# ---------------------------

def split_sentences(text: str):
    """
    Sentence splitter tuned for TTS:
    - protects abbreviations
    - avoids splitting decimals
    """
    t = text

    t = re.sub(r"(\d)\.(\d)", r"\1<DEC>\2", t)

    for ab in ABBREV:
        ab_esc = re.escape(ab[:-1])
        t = re.sub(
            rf"(?i)\b{ab_esc}\.",
            lambda m: m.group(0).replace(".", "<ABBR>"),
            t,
        )

    parts = re.split(r"(?<=[\.\?\!])\s+", t)

    out = []
    for p in parts:
        p = p.replace("<DEC>", ".").replace("<ABBR>", ".")
        p = normalize_ws(p)
        if p:
            out.append(p)
    return out

# ---------------------------
# Step 4.3: SSML generation
# ---------------------------

def escape_ssml(text: str) -> str:
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))

def block_to_ssml(block):
    txt = block["text"]
    sentences = split_sentences(txt)

    if block["type"] == "heading":
        inner = " ".join(escape_ssml(s) for s in sentences)
        return f'<break time="600ms"/><emphasis level="moderate">{inner}</emphasis><break time="400ms"/>'

    inner = " ".join(escape_ssml(s) for s in sentences)
    return f'{inner}<break time="350ms"/>'

def chunk_ssml_blocks(blocks, max_chars=4000):
    chunks = []
    current = []
    current_len = 0

    for b in blocks:
        ss = block_to_ssml(b)

        if current and (current_len + len(ss)) > max_chars:
            chunks.append("<speak>\n" + "\n".join(current) + "\n</speak>")
            current = []
            current_len = 0

        current.append(ss)
        current_len += len(ss)

    if current:
        chunks.append("<speak>\n" + "\n".join(current) + "\n</speak>")

    return chunks

# ---------------------------
# ✅ Extra: print table IDs that need explanations
# ---------------------------

def collect_missing_tables(raw_blocks, table_map: dict):
    missing = []
    for b in raw_blocks:
        t = b.get("text") or ""
        for table_id, caption in extract_table_tokens(t):
            if not (table_map.get(table_id) or "").strip():
                missing.append((table_id, caption))
    # unique preserving order
    seen = set()
    uniq = []
    for tid, cap in missing:
        if tid in seen:
            continue
        seen.add(tid)
        uniq.append((tid, cap))
    return uniq

# ---------------------------
# Main
# ---------------------------

def main():
    if not OUT_BLOCKS.exists():
        raise FileNotFoundError("out_blocks.json not found. Run: python step2_blocks.py first.")

    raw_blocks = json.loads(OUT_BLOCKS.read_text(encoding="utf-8"))

    # ✅ Load user explanations (if exists)
    table_map = load_table_explanations()

    # ✅ Print missing table ids (so you can ask user in UI)
    missing = collect_missing_tables(raw_blocks, table_map)
    if missing:
        print("\n⚠️ Tables found that need user explanations:")
        for tid, cap in missing:
            print(f"  - {tid}: {cap}")
        print("Create table_explanations.json with these ids to replace table speech.\n")

    # normalize + drop toc
    blocks = normalize_blocks(raw_blocks, table_map)
    blocks = drop_toc(blocks)

    print(f"Blocks after normalize: {len(blocks)}")
    ssml_chunks = chunk_ssml_blocks(blocks, max_chars=3800)
    print(f"SSML chunks: {len(ssml_chunks)}\n")

    preview = ssml_chunks[0] if ssml_chunks else ""
    print(preview[:1200] + "\n...\n")

    out_dir = Path("ssml_chunks")
    out_dir.mkdir(exist_ok=True)
    for i, ssml in enumerate(ssml_chunks, start=1):
        (out_dir / f"ssml_{i:04d}.xml").write_text(ssml, encoding="utf-8")

    print(f"✅ Saved SSML chunk files in: {out_dir}/")

if __name__ == "__main__":
    main()
