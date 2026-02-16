import fitz
import json
import argparse
from pathlib import Path
from collections import Counter
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# --- Watermark keyword check (strong filter) ---
WATERMARK_KEYWORDS = [
    "supplied by",
    "under the license",
    "license from bis",
    "valid upto",
    "book supply bureau",
    "csir-national physical laboratory",
]

# -------------------------------
# Control chars (fix: \x08, etc.)
# -------------------------------
CONTROL_CHAR_RE = re.compile(r"[\x00-\x1F\x7F]")

def strip_control_chars(s: str) -> str:
    return CONTROL_CHAR_RE.sub("", s or "")

# -------------------------------
# Remove rules
# -------------------------------
def load_remove_rules(path: str | None):
    if not path:
        return {"exact": set(), "regex": []}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        exact = set((data.get("exact") or []))
        regex = [re.compile(p) for p in (data.get("regex") or [])]
        return {"exact": exact, "regex": regex}
    except Exception:
        return {"exact": set(), "regex": []}

def should_remove_by_rules(text: str, rules) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if t in rules["exact"]:
        return True
    for rx in rules["regex"]:
        if rx.search(t):
            return True
    return False

# -------------------------------
# Filters
# -------------------------------
def is_watermark_text(t: str) -> bool:
    tl = (t or "").strip().lower()
    return any(k in tl for k in WATERMARK_KEYWORDS)

def in_header_or_footer(y0: float, y1: float, page_h: float) -> bool:
    top = page_h * 0.04
    bottom = page_h * 0.96
    return (y1 <= top) or (y0 >= bottom)

def in_right_stamp_strip(x0: float, x1: float, page_w: float) -> bool:
    return x0 >= page_w * 0.95 or x1 >= page_w * 0.98

def is_bs_stamp_span(text: str) -> bool:
    t = (text or "").strip()
    return t.upper() in {"BS", "BIS"}

def is_footer_page_num_candidate(text: str) -> bool:
    t = (text or "").strip()
    return bool(re.fullmatch(r"\d{1,4}", t))

# -------------------------------
# Text utils
# -------------------------------
def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def normalize_multiline_keep_newlines(s: str) -> str:
    lines = [normalize_ws(x) for x in (s or "").splitlines()]
    return "\n".join(lines).strip()

def is_bold(font: str) -> bool:
    return "bold" in (font or "").lower()

def is_italic(font: str) -> bool:
    f = (font or "").lower()
    return ("italic" in f) or ("oblique" in f)

# -------------------------------
# Generic reflow (NO hardcoded content)
# -------------------------------
LIST_MARKER_RE = re.compile(
    r"""^(
        (\([a-zA-Z0-9]+\))|
        ([a-zA-Z]\))|
        (\d+[\.\)])|
        ([IVXLCDM]+[\.\)])|
        ([\-\u2022\u2023\u25E6\u2043\u2219]\s+)
    )\s+""",
    re.VERBOSE,
)
MARKER_ONLY_RE = re.compile(r"^(\([a-zA-Z0-9]+\)|[a-zA-Z]\)|\d+[\.\)]|[IVXLCDM]+[\.\)])$")
MARKER_END_RE = re.compile(r"\s((?:[a-zA-Z]\)|\(\w+\)|\d+[\.\)]|[IVXLCDM]+[\.\)]))\s*$")

def ends_sentence(s: str) -> bool:
    return (s or "").rstrip().endswith((".", "?", "!", ";", ":"))

def looks_like_continuation(prev: str, cur: str) -> bool:
    if not prev or not cur:
        return False
    if ends_sentence(prev):
        return False
    if LIST_MARKER_RE.match(cur):
        return False
    c0 = cur[:1]
    return (c0.islower() or c0 in "([\"'" or c0.isdigit() or len(cur) <= 12)

def reflow_pdf_text(text: str) -> str:
    raw_lines = (text or "").splitlines()
    lines = [ln.strip() for ln in raw_lines]

    tmp = []
    for ln in lines:
        if not ln:
            tmp.append("")
            continue
        m = MARKER_END_RE.search(ln)
        if m and not LIST_MARKER_RE.match(ln):
            marker = m.group(1)
            base = MARKER_END_RE.sub("", ln).rstrip()
            if base:
                tmp.append(base)
            tmp.append(marker)
        else:
            tmp.append(ln)

    out = []
    i = 0
    while i < len(tmp):
        ln = tmp[i]
        if ln == "":
            out.append("")
            i += 1
            continue

        if MARKER_ONLY_RE.fullmatch(ln):
            j = i + 1
            while j < len(tmp) and tmp[j] == "":
                j += 1
            if j < len(tmp):
                out.append(f"{ln} {tmp[j]}")
                i = j + 1
                continue

        out.append(ln)
        i += 1

    merged = []
    i = 0
    while i < len(out):
        cur = out[i]
        if cur == "":
            merged.append("")
            i += 1
            continue

        if i + 1 < len(out):
            nxt = out[i + 1]
            if nxt != "" and looks_like_continuation(cur, nxt):
                merged.append(cur + " " + nxt)
                i += 2
                continue

        merged.append(cur)
        i += 1

    final = []
    for ln in merged:
        if ln == "" and final and final[-1] == "":
            continue
        final.append(ln)

    return "\n".join(final).strip()

# -------------------------------
# TABLE DETECTION (CAPTION-DRIVEN + RULED GRID)
# -------------------------------
TABLE_CAPTION_RE = re.compile(
    r"^\s*Table\s+(?P<num>\d+)\s*(?:[—\-–:]\s*(?P<title>.*))?$",
    re.IGNORECASE
)
TABLE_CONT_RE = re.compile(
    r"^\s*Table\s+(?P<num>\d+)\s*\(\s*continued\s*\)\s*$",
    re.IGNORECASE
)

@dataclass
class TableHit:
    page: int
    bbox: Tuple[float, float, float, float]  # x0,y0,x1,y1 (fitz coords)
    caption: Optional[str] = None
    id: str = ""         # logical table id (T10)
    hit_id: str = ""     # per-occurrence id (T10_P13) useful later if you want

@dataclass
class CaptionHit:
    page: int
    num: int
    text: str
    bbox: Tuple[float, float, float, float]  # caption bbox

def _bbox_union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

def _bbox_area(b):
    return max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))

def _overlap(a, b, pad=0.0) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 - pad or ax0 > bx1 + pad or ay1 < by0 - pad or ay0 > by1 + pad)

def detect_table_captions(page: fitz.Page, page_num: int) -> List[CaptionHit]:
    """
    Detect caption text spans:
      Table 10 — Title...
      Table 10 (continued)
    """
    out: List[CaptionHit] = []
    d = page.get_text("dict")

    for block in d.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue

            parts = []
            x0 = y0 = 1e9
            x1 = y1 = -1e9

            for sp in spans:
                t = (sp.get("text") or "").replace("\u00ad", "")
                t = strip_control_chars(t)
                if not t.strip():
                    continue
                parts.append(t)

                bb = sp.get("bbox")
                if bb:
                    sx0, sy0, sx1, sy1 = bb
                    x0 = min(x0, sx0); y0 = min(y0, sy0)
                    x1 = max(x1, sx1); y1 = max(y1, sy1)

            if not parts or x1 < x0:
                continue

            line_text = normalize_ws("".join(parts))

            m = TABLE_CAPTION_RE.match(line_text)
            mc = TABLE_CONT_RE.match(line_text)

            if m:
                num = int(m.group("num"))
                out.append(CaptionHit(page=page_num, num=num, text=line_text, bbox=(x0, y0, x1, y1)))
            elif mc:
                num = int(mc.group("num"))
                out.append(CaptionHit(page=page_num, num=num, text=line_text, bbox=(x0, y0, x1, y1)))

    out.sort(key=lambda c: (c.bbox[1], c.bbox[0]))
    return out

def drawings_to_bboxes(page: fitz.Page) -> List[Tuple[float, float, float, float]]:
    """
    Convert all drawing primitives into bboxes.
    For ISO PDFs, grid lines are often rectangles/quads (NOT only lines).
    """
    out = []
    drawings = page.get_drawings()

    for d in drawings:
        items = d.get("items", []) or []

        for item in items:
            kind = item[0]

            if kind == "l":
                p1, p2 = item[1], item[2]
                out.append((min(p1.x, p2.x), min(p1.y, p2.y), max(p1.x, p2.x), max(p1.y, p2.y)))

            elif kind == "re":
                r = item[1]
                out.append((float(r.x0), float(r.y0), float(r.x1), float(r.y1)))

            elif kind == "qu":
                q = item[1]
                xs = [q.ul.x, q.ur.x, q.ll.x, q.lr.x]
                ys = [q.ul.y, q.ur.y, q.ll.y, q.lr.y]
                out.append((min(xs), min(ys), max(xs), max(ys)))

    return out

def detect_ruled_clusters(page: fitz.Page, min_items=18, min_area_ratio=0.02):
    """
    Cluster drawing bboxes into large regions likely representing ruled tables.
    """
    w, h = float(page.rect.width), float(page.rect.height)
    bboxes = drawings_to_bboxes(page)

    if len(bboxes) < min_items:
        return []

    clusters = []
    for bb in bboxes:
        placed = False
        for c in clusters:
            if _overlap(bb, c["bbox"], pad=14):
                c["bbox"] = _bbox_union(c["bbox"], bb)
                c["count"] += 1
                placed = True
                break
        if not placed:
            clusters.append({"bbox": bb, "count": 1})

    out = []
    for c in clusters:
        bb = c["bbox"]
        if c["count"] < min_items:
            continue
        if _bbox_area(bb) < (w * h * min_area_ratio):
            continue
        # ignore page frame
        if bb[0] < 8 and bb[1] < 8 and (w - bb[2]) < 8 and (h - bb[3]) < 8:
            continue
        out.append(bb)

    out.sort(key=lambda b: (b[1], b[0]))

    # merge overlaps
    merged = []
    for bb in out:
        if not merged:
            merged.append(bb)
        else:
            last = merged[-1]
            if _overlap(bb, last, pad=10):
                merged[-1] = _bbox_union(last, bb)
            else:
                merged.append(bb)
    return merged

def pick_table_bbox_below_caption(page: fitz.Page, caption_bbox, clusters):
    """
    Choose nearest ruled cluster below caption bbox.
    """
    _, cap_y0, _, cap_y1 = caption_bbox
    candidates = []
    for bb in clusters:
        x0, y0, x1, y1 = bb
        if y0 >= cap_y0 - 6:
            dist = max(0.0, y0 - cap_y1)
            candidates.append((dist, bb))
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1] if candidates else None

def detect_tables(doc: fitz.Document, max_pages=None) -> List[TableHit]:
    """
    Caption-driven table hits.
    Logical id: T{num} (shared for continued pages)
    """
    hits: List[TableHit] = []
    n = len(doc) if max_pages is None else min(len(doc), max_pages)

    for pno in range(n):
        page = doc[pno]
        page_num = pno + 1

        captions = detect_table_captions(page, page_num)
        if not captions:
            continue

        clusters = detect_ruled_clusters(page)

        for cap in captions:
            table_id = f"T{cap.num}"
            hit_id = f"{table_id}_P{page_num}"

            tb = pick_table_bbox_below_caption(page, cap.bbox, clusters)
            if not tb:
                # fallback bbox: from caption bottom to some height below
                tb = (0.0, cap.bbox[3], float(page.rect.width), min(float(page.rect.height), cap.bbox[3] + 260.0))

            hits.append(TableHit(
                page=page_num,
                bbox=tb,
                caption=cap.text,
                id=table_id,
                hit_id=hit_id
            ))

    hits.sort(key=lambda t: (t.page, t.bbox[1], t.bbox[0]))
    return hits

# -------------------------------
# Extraction
# -------------------------------
RIGHT_MARKER_RE = re.compile(r"^(?:[a-zA-Z]\)|\([a-zA-Z0-9]+\))$")  # a) (a) etc.

def extract_lines(doc, max_pages=None, rules=None, tables_by_page=None):
    if rules is None:
        rules = {"exact": set(), "regex": []}
    if tables_by_page is None:
        tables_by_page = {}

    lines_out = []
    pages = len(doc) if max_pages is None else min(len(doc), max_pages)

    for pno in range(pages):
        page_num = pno + 1
        page = doc[pno]
        w = float(page.rect.width)
        h = float(page.rect.height)

        table_rects = [t.bbox for t in tables_by_page.get(page_num, [])]

        data = page.get_text("dict")
        for block in data.get("blocks", []):
            if block.get("type", 0) != 0:
                continue

            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue

                parts = []
                sizes, bold_flags, italic_flags = [], [], []
                xs, ys = [], []

                for sp in spans:
                    t = sp.get("text", "") or ""
                    t = t.replace("\u00ad", "")
                    t = strip_control_chars(t)
                    if not t.strip():
                        continue

                    bbox = sp.get("bbox", None)
                    if bbox:
                        x0, y0, x1, y1 = bbox
                    else:
                        x0 = y0 = x1 = y1 = 0.0

                    size = float(sp.get("size", 0) or 0.0)
                    font = sp.get("font", "") or ""

                    # --- suppress text inside detected table regions ---
                    if bbox and table_rects:
                        span_rect = (x0, y0, x1, y1)
                        if any(_overlap(span_rect, tr, pad=3.0) for tr in table_rects):
                            continue

                    # Filters
                    if is_watermark_text(t):
                        continue
                    if bbox and in_header_or_footer(y0, y1, h):
                        continue
                    if bbox and in_right_stamp_strip(x0, x1, w):
                        continue

                    # Drop BS/BIS stamp spans
                    if is_bs_stamp_span(t) and bbox and (y0 >= h * 0.80 or x0 >= w * 0.75):
                        continue

                    # Drop footer page number candidates near bottom-right
                    if bbox and is_footer_page_num_candidate(t) and (y0 >= h * 0.85) and (x0 >= w * 0.70):
                        continue

                    # Handle "b)" from far-right: keep as newline marker
                    if bbox and RIGHT_MARKER_RE.fullmatch(t.strip()) and (x0 >= w * 0.80):
                        parts.append(("\n" + t.strip(), x0, y0, x1, y1, size, font))
                    else:
                        parts.append((t, x0, y0, x1, y1, size, font))

                    sizes.append(size)
                    bold_flags.append(is_bold(font))
                    italic_flags.append(is_italic(font))
                    xs.append(x0)
                    ys.append(y0)

                if not parts:
                    continue

                raw = "".join(p[0] for p in parts)
                raw = strip_control_chars(raw)
                text = normalize_ws(raw)

                if should_remove_by_rules(text, rules):
                    continue
                if not text or is_watermark_text(text):
                    continue

                avg_size = sum(sizes) / len(sizes) if sizes else 0.0
                bold_ratio = sum(1 for b in bold_flags if b) / len(bold_flags) if bold_flags else 0.0
                italic_ratio = sum(1 for i in italic_flags if i) / len(italic_flags) if italic_flags else 0.0

                lines_out.append({
                    "page": page_num,
                    "x": min(xs) if xs else 0.0,
                    "y": min(ys) if ys else 0.0,
                    "text": text,
                    "avg_size": avg_size,
                    "bold_ratio": bold_ratio,
                    "italic_ratio": italic_ratio,
                    "page_w": w,
                    "page_h": h,
                })

    lines_out.sort(key=lambda r: (r["page"], round(r["y"], 1), r["x"]))
    return lines_out

def detect_repeated_lines_anywhere(lines, min_pages_ratio=0.5, max_len=200):
    pages_seen = len(set(l["page"] for l in lines))
    texts = [l["text"] for l in lines if len(l["text"]) <= max_len]
    cnt = Counter(texts)

    repeated = set()
    for text, c in cnt.items():
        if pages_seen == 0:
            continue
        if (c / pages_seen) < min_pages_ratio:
            continue
        if len(text) >= 25 or is_watermark_text(text):
            repeated.add(text)
    return repeated

# -------------------------------
# Block building
# -------------------------------
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
    body_size = (sum(sizes) / len(sizes)) if sizes else 11.0

    blocks, current, prev = [], None, None

    def flush():
        nonlocal current
        if current:
            current["text"] = normalize_multiline_keep_newlines(current["text"])
            current["text"] = reflow_pdf_text(current["text"])
            if current["text"]:
                blocks.append(current)
        current = None

    for ln in lines:
        ltype = classify_line(ln, body_font_size_guess=body_size)
        btype = "heading" if ltype == "heading" else "paragraph"

        if current is None:
            current = {"page": ln["page"], "type": btype, "text": ln["text"], "y": float(ln.get("y", 0.0))}
        else:
            y_gap = (ln["y"] - prev["y"]) if (prev and ln["page"] == prev["page"]) else 999
            x_gap = abs(ln["x"] - prev["x"]) if prev else 0

            if current["type"] != btype:
                flush()
                current = {"page": ln["page"], "type": btype, "text": ln["text"], "y": float(ln.get("y", 0.0))}
            else:
                if ln["page"] == prev["page"] and y_gap <= 18 and x_gap <= 35:
                    if y_gap > 10:
                        current["text"] += "\n" + ln["text"]
                    else:
                        current["text"] += " " + ln["text"]
                else:
                    flush()
                    current = {"page": ln["page"], "type": btype, "text": ln["text"], "y": float(ln.get("y", 0.0))}
        prev = ln

    flush()
    return blocks, body_size

# -------------------------------
# Table block insertion helpers
# -------------------------------
def _stable_tables_by_page(table_hits: List[TableHit]) -> dict:
    """
    Group by page but keep logical ids (T10). (No more p16_t0)
    """
    byp = {}
    for th in table_hits:
        byp.setdefault(th.page, []).append(th)

    for page, lst in byp.items():
        lst.sort(key=lambda t: (t.bbox[1], t.bbox[0]))
    return byp

def add_table_blocks(blocks: list, doc: fitz.Document, tables_by_page: dict) -> None:
    for page_num, tables in tables_by_page.items():
        if not tables:
            continue

        for th in tables:
            x0, y0, x1, y1 = th.bbox
            cap = th.caption or f"Table {th.id.replace('T','')} (no caption)"
            placeholder = f"[TABLE:{th.id}] {cap}"

            blocks.append({
                "page": int(page_num),
                "type": "table",
                "text": placeholder,
                "y": float(y0),
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "table_id": th.id,     # logical id: T10
                "hit_id": th.hit_id,   # per occurrence: T10_P13
                "caption": cap,
            })

# -------------------------------
# CLI
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remove_rules", default=None, help="Path to JSON rules file generated by Streamlit")
    ap.add_argument("--pdf", required=True, help="Path to uploaded PDF")
    ap.add_argument("--out", required=True, help="Path to output blocks json (out_blocks.json)")
    ap.add_argument("--max_pages", type=int, default=None, help="Debug: limit pages")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    rules = load_remove_rules(args.remove_rules)

    # 1) Detect tables first (caption-driven + ruled grid)
    table_hits = detect_tables(doc, max_pages=args.max_pages)
    tables_by_page = _stable_tables_by_page(table_hits)

    # 2) Extract lines while suppressing table areas
    lines = extract_lines(doc, max_pages=args.max_pages, rules=rules, tables_by_page=tables_by_page)

    # 3) Remove repeated header/footer-ish lines
    repeated = detect_repeated_lines_anywhere(lines, min_pages_ratio=0.5)
    if repeated:
        lines = [ln for ln in lines if ln["text"] not in repeated]

    # 4) Merge into paragraph/heading blocks
    blocks, body_size = merge_into_blocks(lines)

    # 5) Insert table placeholder blocks in reading order
    add_table_blocks(blocks, doc, tables_by_page)

    # 6) Sort blocks by reading order
    blocks.sort(key=lambda b: (
        int(b.get("page", 0)),
        float(b.get("y", 0.0)),
        1 if b.get("type") == "table" else 0
    ))

    out_path.write_text(json.dumps(blocks, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ PDF: {pdf_path.name}")
    print(f"✅ Body font size guess: {body_size:.2f}")
    print(f"✅ Blocks: {len(blocks)}")
    print(f"✅ Tables detected (caption-based): {len(table_hits)}")

    if table_hits:
        per_page = {}
        for t in table_hits:
            per_page[t.page] = per_page.get(t.page, 0) + 1
        top = sorted(per_page.items(), key=lambda x: x[0])[:20]
        print(f"✅ Tables per page (first pages): {top}")

        # debug first few captions
        for t in table_hits[:8]:
            print(f"  - p{t.page} {t.id} :: {t.caption}")

    print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    main()
