import fitz
from pathlib import Path
import re
import json

PDF_PATH = Path("IS60867.pdf")
OUT_DIR = Path("sections")
OUT_DIR.mkdir(exist_ok=True)

# -----------------------
# Filters: watermark/header/footer/right strip
# -----------------------
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
    top = page_h * 0.04
    bottom = page_h * 0.96
    return (y1 <= top) or (y0 >= bottom)

def in_right_stamp_strip(x0: float, x1: float, page_w: float) -> bool:
    return x0 >= page_w * 0.95 or x1 >= page_w * 0.98

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

# -----------------------
# Extract clean lines
# -----------------------
def extract_clean_lines(doc, max_pages=None):
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

                texts = []
                xs, ys, x1s, y1s = [], [], [], []

                for sp in spans:
                    t = sp.get("text", "")
                    t = t.replace("\u00ad", "")
                    if not t.strip():
                        continue

                    bbox = sp.get("bbox", None)
                    if bbox:
                        x0, y0, x1, y1 = bbox

                        # filter
                        if is_watermark_text(t):
                            continue
                        if in_header_or_footer(y0, y1, h):
                            continue
                        if in_right_stamp_strip(x0, x1, w):
                            continue

                        xs.append(x0); ys.append(y0); x1s.append(x1); y1s.append(y1)

                    texts.append(t)

                if not texts:
                    continue

                text = normalize_ws("".join(texts))
                if not text:
                    continue
                if is_watermark_text(text):
                    continue

                lines_out.append({
                    "page": pno + 1,
                    "text": text,
                    "x": min(xs) if xs else 0.0,
                    "y": min(ys) if ys else 0.0,
                })

    lines_out.sort(key=lambda r: (r["page"], round(r["y"], 1), r["x"]))
    return lines_out

# -----------------------
# Heading detection (STRICT)
# -----------------------
RE_NUM_HEADING = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.+)$")   # 1 Scope, 2.1 Title
RE_ANNEX = re.compile(r"^\s*Annex\s+([A-Z])\b(.*)$", re.I)

KNOWN_HEADINGS = {
    "introduction",
    "scope",
    "normative references",
    "terms and definitions",
    "terms, definitions",
    "bibliography",
    "foreword",
    "national foreword",
}

UNIT_WORDS = {
    "cm", "mm", "m", "hz", "khz", "mhz", "ghz",
    "cm2", "mm2", "m2",
    "cm3", "mm3", "m3",
    "kv", "v", "a", "ma",
    "kg", "g", "mg",
    "°c", "c",
    "%", "percent",
}

def looks_like_measurement_title(title: str) -> bool:
    t = title.lower()
    # contains obvious units or patterns like "40 Hz", "380 cm", "cm3"
    if any(u in t for u in UNIT_WORDS):
        return True
    if re.search(r"\b\d+\s*(cm|mm|m|hz|kv|v|a|kg|g|%)\b", t):
        return True
    if re.search(r"\b(cm\d+|mm\d+|m\d+)\b", t):
        return True
    return False

def heading_quality_ok(title: str) -> bool:
    # must contain letters
    if not re.search(r"[A-Za-z]", title):
        return False
    # too long => likely a sentence, not a heading
    if len(title) > 90:
        return False
    # avoid "… and … shows a double peak …" style sentences
    if title.count(" ") > 14:
        return False
    if looks_like_measurement_title(title):
        return False
    return True

def is_section_heading(line: str):
    t = line.strip()
    tl = t.lower()

    # direct known headings
    if tl in KNOWN_HEADINGS:
        return ("named", tl.replace(" ", "_"), t)

    # Annex A / B / C...
    m = RE_ANNEX.match(t)
    if m:
        letter = m.group(1).upper()
        rest = (m.group(2) or "").strip(" -—:()")
        title = f"Annex {letter}" + (f" {rest}" if rest else "")
        return ("annex", f"annex_{letter}", title)

    # Numbered heading like 1 Scope / 2 Normative references
    m = RE_NUM_HEADING.match(t)
    if m:
        num = m.group(1)         # "6" or "6.2"
        title = m.group(2).strip(" -—:")  # clean trailing dash

        # IMPORTANT: reject big top-level numbers like 380
        top = int(num.split(".")[0])
        if top < 1 or top > 30:
            return None

        if not heading_quality_ok(title):
            return None

        return ("num", num, title)

    return None

def safe_filename(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return name[:70] if name else "section"

# -----------------------
# Build sections
# -----------------------
def build_sections(lines):
    sections = []
    current = None

    def flush():
        nonlocal current
        if current and current["lines"]:
            sections.append(current)
        current = None

    for ln in lines:
        h = is_section_heading(ln["text"])
        if h:
            flush()
            kind, key, title = h
            current = {
                "key": key,
                "title": title,
                "kind": kind,
                "start_page": ln["page"],
                "lines": [],
            }
            continue

        if current is None:
            current = {
                "key": "front_matter",
                "title": "Front matter",
                "kind": "meta",
                "start_page": ln["page"],
                "lines": [],
            }

        current["lines"].append(ln)

    flush()
    return sections

def write_sections(sections):
    outline = []
    for idx, sec in enumerate(sections, start=1):
        if sec["key"] == "front_matter":
            file_name = "00_front_matter.txt"
        elif sec["kind"] == "num":
            top = int(sec["key"].split(".")[0])
            file_name = f"{top:02d}_{safe_filename(sec['title'])}.txt"
        else:
            file_name = f"{idx:02d}_{safe_filename(sec['title'])}.txt"

        txt = "\n".join(l["text"] for l in sec["lines"]).strip() + "\n"
        (OUT_DIR / file_name).write_text(txt, encoding="utf-8")

        outline.append({
            "file": file_name,
            "key": sec["key"],
            "title": sec["title"],
            "kind": sec["kind"],
            "start_page": sec["start_page"],
            "line_count": len(sec["lines"]),
        })

    (OUT_DIR / "outline.json").write_text(json.dumps(outline, indent=2), encoding="utf-8")
    return outline

def main():
    # clean output folder first (prevents old random files staying)
    for f in OUT_DIR.glob("*.txt"):
        f.unlink()
    if (OUT_DIR / "outline.json").exists():
        (OUT_DIR / "outline.json").unlink()

    doc = fitz.open(PDF_PATH)
    lines = extract_clean_lines(doc)

    sections = build_sections(lines)
    outline = write_sections(sections)

    print(f"✅ Total lines: {len(lines)}")
    print(f"✅ Sections created: {len(outline)}")
    print(f"✅ Saved to: {OUT_DIR}/")
    print("\nTop sections:")
    for o in outline[:15]:
        print(f"- {o['file']}  |  {o['key']}  |  {o['title']}  (p{o['start_page']})")

if __name__ == "__main__":
    main()
