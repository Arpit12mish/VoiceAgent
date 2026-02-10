import fitz
from pathlib import Path
from collections import Counter
import re
import unicodedata

PDF_PATH = Path("ISO9735.pdf")

# -------------------------------
# Normalization helpers
# -------------------------------

LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}

ABBREV = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.",
    "e.g.", "i.e.", "etc.", "vs.", "no.", "fig.", "eq.",
    "inc.", "ltd.", "pvt.", "co.", "dept.", "gov."
}

def norm_unicode(s: str) -> str:
    # Normalize unicode + replace common typography chars
    s = unicodedata.normalize("NFKC", s)
    for k, v in LIGATURES.items():
        s = s.replace(k, v)
    s = s.replace("\u00ad", "")  # soft hyphen
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = s.replace("–", "-").replace("—", "-")
    return s

def normalize_ws(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def fix_decimal_spacing(s: str) -> str:
    # 3 . 14 -> 3.14
    s = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", s)
    return s

def remove_dot_leaders(s: str) -> str:
    # TOC style: "Foreword........iv" -> "Foreword iv"
    # if there are long dot runs, replace with single space
    s = re.sub(r"\.{5,}", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def looks_like_page_marker(s: str) -> bool:
    # "ii", "iv", "1", "Page", etc.
    t = s.strip().lower()
    if t in {"page"}:
        return True
    if re.fullmatch(r"[ivxlcdm]+", t) and len(t) <= 6:  # roman numerals
        return True
    if re.fullmatch(r"\d{1,3}", t):  # plain page number
        return True
    return False

def is_boilerplate_line(s: str) -> bool:
    t = s.strip().lower()
    # Expand this list over time with what you see in PDFs.
    boiler = [
        "for bis use only",
        "copyright protected document",
        "all rights reserved",
        "© iso",
        "published in switzerland",
    ]
    return any(b in t for b in boiler)

# -------------------------------
# Extraction to blocks (reuse simplified Step 2)
# -------------------------------

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
                    text = sp.get("text", "")
                    text = norm_unicode(text)
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

                avg_size = sum(sizes)/len(sizes) if sizes else 0.0
                bold_ratio = sum(1 for b in bolds if b)/len(bolds) if bolds else 0.0
                italic_ratio = sum(1 for i in italics if i)/len(italics) if italics else 0.0

                lines_out.append({
                    "page": pno + 1,
                    "x": min(xs) if xs else 0.0,
                    "y": min(ys) if ys else 0.0,
                    "text": text,
                    "avg_size": avg_size,
                    "bold_ratio": bold_ratio,
                    "italic_ratio": italic_ratio,
                })

    lines_out.sort(key=lambda r: (r["page"], round(r["y"], 1), r["x"]))
    return lines_out

def detect_repeated_lines_anywhere(lines, min_pages_ratio=0.5, max_len=80):
    # Not only top band: some PDFs stamp text anywhere (like your BIS stamp).
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

# -------------------------------
# Block-level normalization
# -------------------------------

def normalize_blocks(blocks):
    cleaned = []
    for b in blocks:
        t = b["text"]
        t = norm_unicode(t)
        t = remove_dot_leaders(t)
        t = fix_decimal_spacing(t)
        t = normalize_ws(t)

        # Drop obvious noise
        if not t:
            continue
        if looks_like_page_marker(t):
            continue
        if is_boilerplate_line(t):
            continue

        # Drop very short junk lines (tune later)
        if len(t) <= 2:
            continue

        # Optional: skip “Contents” headings entirely (TOC)
        if b["type"] == "heading" and t.lower() == "contents":
            continue

        cleaned.append({**b, "text": t})
    return cleaned

def main():
    doc = fitz.open(PDF_PATH)
    lines = extract_lines(doc, max_pages=8)  # iterate on first 8 pages  # keep for debugging only
    repeated = detect_repeated_lines_anywhere(lines, min_pages_ratio=0.5)

    # Remove repeated + known boilerplate lines
    lines = [ln for ln in lines if ln["text"] not in repeated and not is_boilerplate_line(ln["text"])]

    blocks = merge_into_blocks(lines)
    blocks2 = normalize_blocks(blocks)

    print(f"Repeated lines removed (anywhere): {list(repeated)[:10]}")
    print(f"Blocks before normalize: {len(blocks)}")
    print(f"Blocks after normalize : {len(blocks2)}\n")

    for b in blocks2[:40]:
        print(f'[p{b["page"]}] {b["type"].upper()}: {b["text"]}')

def build_normalized_blocks(max_pages=None):
    doc = fitz.open(PDF_PATH)

    lines = extract_lines(doc, max_pages=max_pages)
    repeated = detect_repeated_lines_anywhere(lines, min_pages_ratio=0.5)

    # Remove repeated + known boilerplate lines
    lines = [ln for ln in lines if ln["text"] not in repeated and not is_boilerplate_line(ln["text"])]

    blocks = merge_into_blocks(lines)
    blocks2 = normalize_blocks(blocks)
    return blocks2


if __name__ == "__main__":
    main()
