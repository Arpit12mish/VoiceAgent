import fitz
from pathlib import Path
import re
import unicodedata
from collections import Counter

PDF_PATH = Path("ISO9735.pdf")

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

def norm_unicode(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    for k, v in LIGATURES.items():
        s = s.replace(k, v)
    s = s.replace("\u00ad", "")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = s.replace("–", "-").replace("—", "-")
    return s

def normalize_ws(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    return s.strip()

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
# PDF -> blocks (simple)
# ---------------------------

def is_bold(font: str) -> bool:
    return "bold" in (font or "").lower()

def is_italic(font: str) -> bool:
    f = (font or "").lower()
    return ("italic" in f) or ("oblique" in f)

def extract_lines(doc, max_pages=None):
    lines_out = []
    pages = len(doc) if max_pages is None else min(len(doc), max_pages)

    for pno in range(pages):
        page = doc[pno]
        data = page.get_text("dict")
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue

                texts, sizes, bolds, italics, xs, ys = [], [], [], [], [], []
                for sp in spans:
                    text = norm_unicode(sp.get("text", ""))
                    if not text.strip():
                        continue
                    texts.append(text)
                    sizes.append(float(sp.get("size", 0)))
                    f = sp.get("font", "")
                    bolds.append(is_bold(f))
                    italics.append(is_italic(f))
                    bbox = sp.get("bbox")
                    if bbox:
                        xs.append(bbox[0]); ys.append(bbox[1])

                if not texts:
                    continue

                text = normalize_ws("".join(texts))
                if not text:
                    continue

                lines_out.append({
                    "page": pno + 1,
                    "x": min(xs) if xs else 0.0,
                    "y": min(ys) if ys else 0.0,
                    "text": text,
                    "avg_size": sum(sizes)/len(sizes) if sizes else 0.0,
                    "bold_ratio": sum(1 for b in bolds if b)/len(bolds) if bolds else 0.0,
                    "italic_ratio": sum(1 for i in italics if i)/len(italics) if italics else 0.0,
                })

    lines_out.sort(key=lambda r: (r["page"], round(r["y"], 1), r["x"]))
    return lines_out

def detect_repeated_lines_anywhere(lines, min_pages_ratio=0.5, max_len=80):
    pages_seen = len(set(l["page"] for l in lines))
    cnt = Counter([l["text"] for l in lines if len(l["text"]) <= max_len])
    repeated = set()
    for text, c in cnt.items():
        if pages_seen > 0 and (c / pages_seen) >= min_pages_ratio:
            repeated.add(text)
    return repeated

def classify_line(line, body_guess=11.0):
    size = line["avg_size"]
    bold = line["bold_ratio"] >= 0.6
    if size >= body_guess * 1.5:
        return "heading"
    if bold and size >= body_guess * 1.25:
        return "heading"
    return "text"

def merge_into_blocks(lines):
    sizes = [ln["avg_size"] for ln in lines if 9 <= ln["avg_size"] <= 13]
    body_size = (sum(sizes)/len(sizes)) if sizes else 11.0

    blocks = []
    cur = None
    prev = None

    def flush():
        nonlocal cur
        if cur:
            cur["text"] = normalize_ws(cur["text"])
            if cur["text"]:
                blocks.append(cur)
        cur = None

    for ln in lines:
        ltype = classify_line(ln, body_guess=body_size)
        btype = "heading" if ltype == "heading" else "paragraph"

        if cur is None:
            cur = {"type": btype, "page": ln["page"], "text": ln["text"]}
        else:
            y_gap = (ln["y"] - prev["y"]) if (prev and ln["page"] == prev["page"]) else 999
            x_gap = abs(ln["x"] - prev["x"]) if prev else 0

            if cur["type"] != btype:
                flush()
                cur = {"type": btype, "page": ln["page"], "text": ln["text"]}
            else:
                if ln["page"] == prev["page"] and y_gap <= 18 and x_gap <= 25:
                    cur["text"] += " " + ln["text"]
                else:
                    flush()
                    cur = {"type": btype, "page": ln["page"], "text": ln["text"]}
        prev = ln

    flush()
    return blocks

def normalize_blocks(blocks):
    out = []
    for b in blocks:
        t = b["text"]
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
    # lines like "5.1.1" or "2" (but plain numbers already removed)
    return bool(re.fullmatch(r"\d+(\.\d+)+", text.strip()))

def looks_like_toc_entry(text: str) -> bool:
    # "Scope 1", "Foreword iv", "Terms and definitions 2"
    # Usually ends with a page marker (digits or roman)
    t = text.strip()
    return bool(re.search(r"\s(\d{1,3}|[ivxlcdm]{1,6})$", t.lower()))

def drop_toc(blocks):
    """
    Simple default strategy:
    - If we see a heading 'Contents', drop that heading and subsequent TOC-like lines
      until a non-TOC-like paragraph pattern appears (or page changes beyond a small range).
    """
    out = []
    in_toc = False
    toc_start_page = None

    for b in blocks:
        txt = b["text"].strip()
        low = txt.lower()

        if b["type"] == "heading" and low in TOC_START_TITLES:
            in_toc = True
            toc_start_page = b["page"]
            continue  # drop the heading

        if in_toc:
            # Stop TOC after a few pages or when entries stop looking like TOC
            if toc_start_page is not None and b["page"] > toc_start_page + 2:
                in_toc = False
            else:
                # drop typical TOC lines
                if is_toc_index_line(txt) or looks_like_toc_entry(txt):
                    continue
                # if a paragraph looks like actual prose, end toc
                if len(txt.split()) >= 12:
                    in_toc = False
                else:
                    # keep dropping short lines while in TOC
                    continue

        out.append(b)

    return out

# ---------------------------
# Step 4.2: Sentence splitting (dot logic v1)
# ---------------------------

def split_sentences(text: str):
    """
    Sentence splitter tuned for TTS:
    - protects abbreviations
    - avoids splitting decimals
    - splits on .?! when likely sentence end
    """
    t = text

    # protect decimals 3.14 -> 3<DEC>14
    t = re.sub(r"(\d)\.(\d)", r"\1<DEC>\2", t)

    # protect common abbreviations (case-insensitive)
    def protect_abbrev(m):
        return m.group(0).replace(".", "<ABBR>")
    # Build regex like r'\b(dr|mr|e\.g|i\.e)\.'
    # We'll just brute protect tokens ending with '.'
    tokens = t.split()
    for ab in ABBREV:
        ab_esc = re.escape(ab[:-1])  # without last dot
        t = re.sub(rf"(?i)\b{ab_esc}\.", lambda m: m.group(0).replace(".", "<ABBR>"), t)

    # Split on sentence punctuation
    parts = re.split(r"(?<=[\.\?\!])\s+", t)

    # restore
    out = []
    for p in parts:
        p = p.replace("<DEC>", ".").replace("<ABBR>", ".")
        p = normalize_ws(p)
        if p:
            out.append(p)
    return out

# ---------------------------
# Step 4.3: SSML generation for Google
# ---------------------------

def escape_ssml(text: str) -> str:
    # minimal escaping
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))

def block_to_ssml(block):
    txt = block["text"]
    sentences = split_sentences(txt)

    if block["type"] == "heading":
        # stronger pauses around headings
        inner = " ".join(escape_ssml(s) for s in sentences)
        return f'<break time="600ms"/><emphasis level="moderate">{inner}</emphasis><break time="400ms"/>'

    # paragraph
    inner = " ".join(escape_ssml(s) for s in sentences)
    return f'{inner}<break time="350ms"/>'

def render_ssml(blocks):
    body = "\n".join(block_to_ssml(b) for b in blocks)
    return f"<speak>\n{body}\n</speak>"

# ---------------------------
# Step 4.4: Chunk SSML (provider limits)
# ---------------------------

def chunk_ssml_blocks(blocks, max_chars=4000):
    """
    Google TTS can accept fairly large SSML, but keep it safe for long form.
    We'll chunk by blocks while staying under max_chars.
    """
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

def main():
    doc = fitz.open(PDF_PATH)

    lines = extract_lines(doc, max_pages=12)  # expand as needed
    repeated = detect_repeated_lines_anywhere(lines, min_pages_ratio=0.5)
    lines = [ln for ln in lines if ln["text"] not in repeated and not is_boilerplate_line(ln["text"])]

    blocks = merge_into_blocks(lines)
    blocks = normalize_blocks(blocks)
    blocks = drop_toc(blocks)

    print(f"Blocks after normalize: {len(blocks)}")
    ssml_chunks = chunk_ssml_blocks(blocks, max_chars=3800)
    print(f"SSML chunks: {len(ssml_chunks)}\n")

    # Print first chunk preview (truncated)
    preview = ssml_chunks[0]
    print(preview[:1200] + "\n...\n")

if __name__ == "__main__":
    main()
