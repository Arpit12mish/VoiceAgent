import json
import re
from pathlib import Path
import fitz

PDF_PATH = Path("ISO9735.pdf")

OUT_DIR = Path("sections_original")
OUT_DIR.mkdir(exist_ok=True)

MANIFEST_PATH = Path("sections_manifest.json")

# ---- your helpers (same idea as step2_blocks.py) ----
def is_bold(font: str) -> bool:
    return "bold" in (font or "").lower()

def is_italic(font: str) -> bool:
    f = (font or "").lower()
    return ("italic" in f) or ("oblique" in f)

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

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
                    t = sp.get("text", "")
                    t = t.replace("\u00ad", "")
                    if not t.strip():
                        continue
                    texts.append(t)
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

                lines_out.append({
                    "page": pno + 1,
                    "x": min(xs) if xs else 0.0,
                    "y": min(ys) if ys else 0.0,
                    "text": text,
                    "avg_size": avg_size,
                    "bold_ratio": bold_ratio,
                })

    lines_out.sort(key=lambda r: (r["page"], round(r["y"], 1), r["x"]))
    return lines_out

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

# ---- section logic ----

SEC_NUM_HEADING = re.compile(r"^\s*(\d+(\.\d+)*)\s+(.+)$", re.IGNORECASE)
ANNEX_HEADING = re.compile(r"^\s*annex\s+([A-Z])\b", re.IGNORECASE)
BIBLIO = re.compile(r"^\s*bibliography\s*$", re.IGNORECASE)

def slugify(s: str, max_len=60) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s_]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s[:max_len] if s else "section"

def is_section_heading(text: str) -> bool:
    t = text.strip()
    return bool(SEC_NUM_HEADING.match(t) or ANNEX_HEADING.match(t) or BIBLIO.match(t))

def build_sections(blocks):
    sections = []
    current = None

    def start_section(sec_id, title, page_start):
        return {
            "id": sec_id,
            "title": title,
            "page_start": page_start,
            "page_end": page_start,
            "text": ""
        }

    def flush():
        nonlocal current
        if current and current["text"].strip():
            current["text"] = current["text"].strip()
            sections.append(current)
        current = None

    # If the PDF begins without a "1 ..." heading => front matter
    front = start_section("00_front_matter", "Front matter", 1)
    front_started = False

    for b in blocks:
        t = b["text"].strip()
        p = b["page"]

        if b["type"] == "heading" and is_section_heading(t):
            # flush existing section / front
            if current:
                flush()
            elif front_started:
                # front matter already collected
                front["page_end"] = p
            else:
                # first time switching from front matter
                if front["text"].strip():
                    sections.append(front)
                front_started = True

            # make new section id from heading
            m = SEC_NUM_HEADING.match(t)
            if m:
                num = m.group(1)
                title = f"{num} {m.group(3).strip()}"
                sid = f"{num}_{slugify(m.group(3))}"
            else:
                m2 = ANNEX_HEADING.match(t)
                if m2:
                    letter = m2.group(1)
                    title = f"Annex {letter}"
                    sid = f"annex_{letter.lower()}"
                else:
                    title = "Bibliography"
                    sid = "bibliography"

            current = start_section(sid, title, p)
            current["text"] += t + "\n\n"
            continue

        # collect into current section or front
        if current:
            current["text"] += t + "\n\n"
            current["page_end"] = max(current["page_end"], p)
        else:
            front_started = True
            front["text"] += t + "\n\n"
            front["page_end"] = max(front["page_end"], p)

    if current:
        flush()
    else:
        if front["text"].strip():
            sections.append(front)

    return sections

def main():
    doc = fitz.open(PDF_PATH)
    lines = extract_lines(doc, max_pages=None)
    blocks = merge_into_blocks(lines)

    sections = build_sections(blocks)

    manifest = []
    for idx, s in enumerate(sections):
        # stable ordering prefix
        order = str(idx).zfill(2)
        fname = f"{order}_{s['id']}.txt"
        out_path = OUT_DIR / fname
        out_path.write_text(s["text"], encoding="utf-8")

        manifest.append({
            "order": idx,
            "id": s["id"],
            "title": s["title"],
            "file": str(out_path).replace("\\", "/"),
            "page_start": s["page_start"],
            "page_end": s["page_end"],
            "word_count": len(s["text"].split())
        })

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✅ Sections saved in: {OUT_DIR}/")
    print(f"✅ Manifest saved: {MANIFEST_PATH}")
    print(f"Total sections: {len(manifest)}")
    print("Sample:", manifest[:5])

if __name__ == "__main__":
    main()
