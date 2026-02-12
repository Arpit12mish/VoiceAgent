import fitz
from pathlib import Path
from collections import Counter
import re

PDF_PATH = Path("IS60867.pdf")

# ---------- helpers ----------
def is_bold(font: str) -> bool:
    return "bold" in (font or "").lower()

def is_italic(font: str) -> bool:
    f = (font or "").lower()
    return ("italic" in f) or ("oblique" in f)

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

# --- Watermark keyword check (strong filter) ---
WATERMARK_KEYWORDS = [
    "supplied by",
    "under the license",
    "license from bis",
    "valid upto",
    "book supply bureau",
    "csir-national physical laboratory",
]

def is_watermark_text(t: str) -> bool:
    tl = t.strip().lower()
    return any(k in tl for k in WATERMARK_KEYWORDS)

def in_header_or_footer(y0: float, y1: float, page_h: float) -> bool:
    # tighter: top 4% + bottom 4%
    top = page_h * 0.04
    bottom = page_h * 0.96
    return (y1 <= top) or (y0 >= bottom)

def in_right_stamp_strip(x0: float, x1: float, page_w: float) -> bool:
    # right strip: if most of span bbox is in last 5% width
    return x0 >= page_w * 0.95 or x1 >= page_w * 0.98

def extract_lines(doc, max_pages=None):
    lines_out = []
    pages = len(doc) if max_pages is None else min(len(doc), max_pages)

    for pno in range(pages):
        page = doc[pno]
        w = float(page.rect.width)
        h = float(page.rect.height)

        data = page.get_text("dict")
        for block in data.get("blocks", []):
            if block.get("type", 0) != 0:
                continue

            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue

                texts, sizes, bold_flags, italic_flags = [], [], [], []
                xs, ys, x1s, y1s = [], [], [], []

                for sp in spans:
                    t = sp.get("text", "")
                    t = t.replace("\u00ad", "")
                    if not t.strip():
                        continue

                    bbox = sp.get("bbox", None)
                    if bbox:
                        x0, y0, x1, y1 = bbox

                        # 1) Remove explicit watermark by keywords (best)
                        if is_watermark_text(t):
                            continue

                        # 2) Remove header/footer bands
                        if in_header_or_footer(y0, y1, h):
                            continue

                        # 3) Remove right stamp strip (vertical license stamp)
                        if in_right_stamp_strip(x0, x1, w):
                            continue

                        xs.append(x0); ys.append(y0); x1s.append(x1); y1s.append(y1)

                    texts.append(t)
                    sizes.append(float(sp.get("size", 0)))
                    f = sp.get("font", "")
                    bold_flags.append(is_bold(f))
                    italic_flags.append(is_italic(f))

                if not texts:
                    continue

                text = normalize_ws("".join(texts))
                if not text:
                    continue

                # if line itself is watermark (merged)
                if is_watermark_text(text):
                    continue

                avg_size = sum(sizes) / len(sizes) if sizes else 0.0
                bold_ratio = sum(1 for b in bold_flags if b) / len(bold_flags) if bold_flags else 0.0
                italic_ratio = sum(1 for i in italic_flags if i) / len(italic_flags) if italic_flags else 0.0

                x = min(xs) if xs else 0.0
                y = min(ys) if ys else 0.0
                bbox_line = [min(xs), min(ys), max(x1s), max(y1s)] if xs and ys and x1s and y1s else None

                lines_out.append({
                    "page": pno + 1,
                    "x": x,
                    "y": y,
                    "text": text,
                    "avg_size": avg_size,
                    "bold_ratio": bold_ratio,
                    "italic_ratio": italic_ratio,
                    "bbox": bbox_line,
                    "page_w": w,
                    "page_h": h,
                })

    lines_out.sort(key=lambda r: (r["page"], round(r["y"], 1), r["x"]))
    return lines_out

def detect_repeated_lines_anywhere(lines, min_pages_ratio=0.5, max_len=200):
    """
    Only remove repeated lines if they look like watermark/footer-type lines.
    Avoid removing common words like 'and', 'of', etc.
    """
    pages_seen = len(set(l["page"] for l in lines))
    texts = [l["text"] for l in lines if len(l["text"]) <= max_len]

    cnt = Counter(texts)
    repeated = set()

    for text, c in cnt.items():
        if pages_seen == 0:
            continue
        if (c / pages_seen) < min_pages_ratio:
            continue

        # ✅ must be long OR must contain watermark keywords
        if len(text) >= 25 or is_watermark_text(text):
            repeated.add(text)

    return repeated

def classify_line(line, body_font_size_guess=11.0):
    size = line["avg_size"]
    bold = line["bold_ratio"] >= 0.6

    if size >= body_font_size_guess * 1.5:
        return "heading"
    if bold and size >= body_font_size_guess * 1.25:
        return "heading"
    return "text"

def merge_into_blocks(lines):
    sizes = [ln["avg_size"] for ln in lines if 9 <= ln["avg_size"] <= 13]
    body_size = (sum(sizes)/len(sizes)) if sizes else 11.0

    blocks = []
    current = None
    prev = None

    def flush():
        nonlocal current
        if current:
            current["text"] = normalize_ws(current["text"])
            if current["text"]:
                blocks.append(current)
        current = None

    for ln in lines:
        ltype = classify_line(ln, body_font_size_guess=body_size)
        btype = "heading" if ltype == "heading" else "paragraph"

        if current is None:
            current = {"page": ln["page"], "type": btype, "text": ln["text"]}
        else:
            y_gap = (ln["y"] - prev["y"]) if (prev and ln["page"] == prev["page"]) else 999
            x_gap = abs(ln["x"] - prev["x"]) if prev else 0

            if current["type"] != btype:
                flush()
                current = {"page": ln["page"], "type": btype, "text": ln["text"]}
            else:
                if ln["page"] == prev["page"] and y_gap <= 18 and x_gap <= 35:
                    current["text"] += " " + ln["text"]
                else:
                    flush()
                    current = {"page": ln["page"], "type": btype, "text": ln["text"]}

        prev = ln

    flush()
    return blocks, body_size

def main():
    doc = fitz.open(PDF_PATH)
    lines = extract_lines(doc, max_pages=15)

    repeated = detect_repeated_lines_anywhere(lines, min_pages_ratio=0.5)
    if repeated:
        lines = [ln for ln in lines if ln["text"] not in repeated]

    blocks, body_size = merge_into_blocks(lines)

    print(f"Body font size guess: {body_size:.2f}")
    print(f"Repeated lines removed: {list(repeated)[:10]}")
    print(f"Blocks produced (first 15 pages): {len(blocks)}\n")

    for b in blocks[:35]:
        print(f'[p{b["page"]}] {b["type"].upper()}: {b["text"]}')

if __name__ == "__main__":
    main()
