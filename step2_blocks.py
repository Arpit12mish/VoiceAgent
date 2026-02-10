import fitz
from pathlib import Path
from collections import Counter
import re

PDF_PATH = Path("ISO9735.pdf")

def is_bold(font: str) -> bool:
    return "bold" in (font or "").lower()

def is_italic(font: str) -> bool:
    f = (font or "").lower()
    return ("italic" in f) or ("oblique" in f)

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def extract_lines(doc, max_pages=None):
    """
    Returns a list of lines with merged spans per line.
    Each line contains: page, y (approx), x, text, avg_size, bold_ratio, italic_ratio
    """
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

                # Merge spans in that line
                texts = []
                sizes = []
                bold_flags = []
                italic_flags = []
                xs = []
                ys = []

                for sp in spans:
                    t = sp.get("text", "")
                    t = t.replace("\u00ad", "")  # soft hyphen
                    if not t.strip():
                        continue

                    texts.append(t)
                    sizes.append(float(sp.get("size", 0)))
                    f = sp.get("font", "")
                    bold_flags.append(is_bold(f))
                    italic_flags.append(is_italic(f))
                    bbox = sp.get("bbox", None)
                    if bbox:
                        xs.append(bbox[0])
                        ys.append(bbox[1])

                if not texts:
                    continue

                text = normalize_ws("".join(texts))
                if not text:
                    continue

                avg_size = sum(sizes)/len(sizes) if sizes else 0.0
                bold_ratio = sum(1 for b in bold_flags if b)/len(bold_flags) if bold_flags else 0.0
                italic_ratio = sum(1 for i in italic_flags if i)/len(italic_flags) if italic_flags else 0.0

                x = min(xs) if xs else 0.0
                y = min(ys) if ys else 0.0

                lines_out.append({
                    "page": pno + 1,
                    "x": x,
                    "y": y,
                    "text": text,
                    "avg_size": avg_size,
                    "bold_ratio": bold_ratio,
                    "italic_ratio": italic_ratio
                })

    # Sort reading order: page -> y -> x
    lines_out.sort(key=lambda r: (r["page"], round(r["y"], 1), r["x"]))
    return lines_out

def detect_repeated_headers(lines, top_band_y=120, min_pages_ratio=0.4):
    """
    Find lines that repeat on many pages near the top (headers).
    Works well for 'FOR BIS USE ONLY' etc.
    """
    # Only consider lines in top band
    candidates = [ln["text"] for ln in lines if ln["y"] <= top_band_y and len(ln["text"]) <= 60]
    cnt = Counter(candidates)

    # Approx page count from data
    pages_seen = len(set(ln["page"] for ln in lines))
    repeated = set()
    for text, c in cnt.items():
        if pages_seen > 0 and (c / pages_seen) >= min_pages_ratio:
            repeated.add(text)
    return repeated

def classify_line(line, body_font_size_guess=11.0):
    """
    Heuristic:
    - heading if size much larger than body OR bold large
    - paragraph otherwise
    """
    size = line["avg_size"]
    bold = line["bold_ratio"] >= 0.6

    if size >= body_font_size_guess * 1.5:
        return "heading"
    if bold and size >= body_font_size_guess * 1.25:
        return "heading"
    return "text"

def merge_into_blocks(lines):
    """
    Merge consecutive lines into blocks based on:
    - same type (heading/text)
    - proximity in y (line spacing)
    - indentation similarity
    """
    # Estimate typical body size from lines that look like text
    sizes = [ln["avg_size"] for ln in lines if 9 <= ln["avg_size"] <= 13]
    body_size = (sum(sizes)/len(sizes)) if sizes else 11.0

    blocks = []
    current = None

    def flush():
        nonlocal current
        if current:
            current["text"] = normalize_ws(current["text"])
            if current["text"]:
                blocks.append(current)
        current = None

    prev = None
    for ln in lines:
        ltype = classify_line(ln, body_font_size_guess=body_size)

        if current is None:
            current = {
                "page_start": ln["page"],
                "page_end": ln["page"],
                "type": "heading" if ltype == "heading" else "paragraph",
                "text": ln["text"],
                "meta": {
                    "avg_size": ln["avg_size"],
                    "bold_ratio": ln["bold_ratio"],
                    "italic_ratio": ln["italic_ratio"],
                }
            }
        else:
            # Decide whether to merge
            same_page_or_next = (ln["page"] == prev["page"]) if prev else True
            y_gap = (ln["y"] - prev["y"]) if (prev and ln["page"] == prev["page"]) else 999
            x_gap = abs(ln["x"] - prev["x"]) if prev else 0

            current_is_heading = (current["type"] == "heading")
            next_is_heading = (ltype == "heading")

            # Rule: don't mix heading with paragraph
            if current_is_heading != next_is_heading:
                flush()
                current = {
                    "page_start": ln["page"],
                    "page_end": ln["page"],
                    "type": "heading" if next_is_heading else "paragraph",
                    "text": ln["text"],
                    "meta": {
                        "avg_size": ln["avg_size"],
                        "bold_ratio": ln["bold_ratio"],
                        "italic_ratio": ln["italic_ratio"],
                    }
                }
            else:
                # Merge if close enough (same paragraph)
                # Typical line gap ~ 8-16 in many PDFs; we use a tolerant threshold.
                if same_page_or_next and y_gap <= 18 and x_gap <= 25:
                    current["text"] += " " + ln["text"]
                    current["page_end"] = ln["page"]
                else:
                    flush()
                    current = {
                        "page_start": ln["page"],
                        "page_end": ln["page"],
                        "type": "heading" if next_is_heading else "paragraph",
                        "text": ln["text"],
                        "meta": {
                            "avg_size": ln["avg_size"],
                            "bold_ratio": ln["bold_ratio"],
                            "italic_ratio": ln["italic_ratio"],
                        }
                    }

        prev = ln

    flush()
    return blocks, body_size

def main():
    doc = fitz.open(PDF_PATH)
    lines = extract_lines(doc, max_pages=5)  # use first 5 pages for quick iteration

    headers = detect_repeated_headers(lines, top_band_y=140, min_pages_ratio=0.5)
    if headers:
        lines = [ln for ln in lines if ln["text"] not in headers]

    blocks, body_size = merge_into_blocks(lines)

    print(f"Body font size guess: {body_size:.2f}")
    print(f"Detected repeated headers removed: {list(headers)[:5]}")
    print(f"Blocks produced (first 5 pages): {len(blocks)}\n")

    for b in blocks[:30]:
        p = f'p{b["page_start"]}' if b["page_start"] == b["page_end"] else f'p{b["page_start"]}-p{b["page_end"]}'
        print(f'[{p}] {b["type"].upper()}: {b["text"]}')

if __name__ == "__main__":
    main()
