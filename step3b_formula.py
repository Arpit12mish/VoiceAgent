# step3b_formula.py
"""
Step3b: Formula pipeline for Research PDFs (Digital text PDFs)

✅ Key improvements for reliable formula detection (like your screenshots):
1) Anchor-based detection using RIGHT-MARGIN equation numbers: "(12)", "(13)", "(14)", "(15)".
   - This is the most robust for standards/research PDFs where the equation number is on the right.
2) Layout-aware fallback for formulas WITHOUT eq numbers:
   - Detect stacked fraction layout (numerator above denominator) near an "=" line.
   - Detect "math-ish" lines and expand bbox to include nearby lines.
3) Better crops for OCR (zoom/pad), and safer runtime (never fail on one bad formula).
4) Optional: initialize pix2tex model ONCE (faster + less crashes).

USAGE (Streamlit pipeline mode):
  python step3b_formula.py --pdf ISO9735.pdf \
    --blocks_in out_blocks.json \
    --blocks_out out_blocks_with_formulas.json \
    --out_dir runs/job_xxx

Outputs inside out_dir:
  - formula_index.json
  - formulas.json
  - formulas/
  - formula_ocr_cache.json
"""

import base64
import hashlib
import json
import os
import re
import time
import unicodedata
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import fitz  # PyMuPDF


# ---------------------------
# Basic utils
# ---------------------------

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def norm_unicode(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("\u00ad", "")  # soft hyphen
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = s.replace("–", "-").replace("—", "-")
    return s

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default

def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------
# Detection patterns
# ---------------------------

EQUATION_NO_RE = re.compile(r"^\(\s*\d{1,4}\s*\)$")           # "(12)"
EQUATION_INLINE_RE = re.compile(r"\(\s*\d{1,4}\s*\)\s*$")    # "... (12)" at end of line

HAS_MATH_RE = re.compile(r"[=±×*/^_∑∏√∫≈≠≤≥→←↔∞αβγΔμΩπθλ∈∉∩∪∂∇·•]")
MATH_WORDS_RE = re.compile(r"\b(sum|prod|sqrt|lim|log|ln|sin|cos|tan|exp|min|max)\b", re.I)

def looks_like_formula_text(text: str) -> bool:
    t = normalize_ws(norm_unicode(text))
    if not t:
        return False
    if len(t) > 900:
        return False
    if EQUATION_NO_RE.match(t):
        return True
    if HAS_MATH_RE.search(t):
        return True
    if MATH_WORDS_RE.search(t):
        return True
    # variable assignment patterns
    if re.search(r"\b[A-Za-z]{1,6}\s*[_]?\s*[A-Za-z0-9]{0,8}\s*=\s*[^=]", t):
        return True

    op_count = len(re.findall(r"[=+\-*/^_(){}\[\]]", t))
    digit_count = len(re.findall(r"\d", t))
    if op_count >= 2 and digit_count >= 2:
        return True
    return False


# ---------------------------
# Line extraction (dict mode)
# ---------------------------

def extract_line_level_for_page(doc: fitz.Document, pno_1based: int) -> List[Dict[str, Any]]:
    """
    Returns list of lines with bbox + average font size.
    """
    page = doc[pno_1based - 1]
    data = page.get_text("dict")
    lines = []

    for block in data.get("blocks", []):
        if block.get("type", 0) != 0:
            continue

        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue

            texts = []
            xs, ys, xe, ye = [], [], [], []
            sizes = []

            for sp in spans:
                txt = norm_unicode(sp.get("text", ""))
                if txt.strip():
                    texts.append(txt)

                bbox = sp.get("bbox")
                if bbox:
                    xs.append(bbox[0]); ys.append(bbox[1]); xe.append(bbox[2]); ye.append(bbox[3])

                if "size" in sp:
                    try:
                        sizes.append(float(sp["size"]))
                    except Exception:
                        pass

            if not texts or not xs:
                continue

            text = normalize_ws("".join(texts))
            bbox = [min(xs), min(ys), max(xe), max(ye)]
            avg_size = (sum(sizes) / len(sizes)) if sizes else 0.0

            lines.append({"text": text, "bbox": bbox, "size": avg_size})

    # sort by y then x for stable grouping
    lines.sort(key=lambda d: (float(d["bbox"][1]), float(d["bbox"][0])))
    return lines


# ---------------------------
# Geometry helpers
# ---------------------------

def _bbox_union(b1, b2):
    return [min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])]

def _line_h(bbox):
    return max(1.0, float(bbox[3]) - float(bbox[1]))

def _y_mid(bbox):
    return (float(bbox[1]) + float(bbox[3])) / 2.0

def _x_mid(bbox):
    return (float(bbox[0]) + float(bbox[2])) / 2.0

def _overlap_x(b1, b2) -> float:
    """returns overlap ratio over min width"""
    x0 = max(float(b1[0]), float(b2[0]))
    x1 = min(float(b1[2]), float(b2[2]))
    inter = max(0.0, x1 - x0)
    w1 = max(1.0, float(b1[2]) - float(b1[0]))
    w2 = max(1.0, float(b2[2]) - float(b2[0]))
    return inter / min(w1, w2)


# ---------------------------
# NEW: Anchor-based detection using right-margin equation numbers
# ---------------------------

def detect_equations_by_right_eqno(
    lines: List[Dict[str, Any]],
    page_width: float,
) -> List[Dict[str, Any]]:
    """
    Detect display equations using a right-margin "(12)" equation number.
    We then build a bbox region to the left that captures the whole formula block
    (including stacked numerator/denominator).
    """
    eqno_lines = []
    for ln in lines:
        t = (ln.get("text") or "").strip()
        b = ln.get("bbox")
        if not b:
            continue

        # equation number on right (common in standards)
        if EQUATION_NO_RE.match(t) and float(b[0]) >= page_width * 0.72:
            eqno_lines.append(ln)

        # sometimes eqno appears at end of a longer line
        elif EQUATION_INLINE_RE.search(t) and float(b[2]) >= page_width * 0.92:
            # isolate the "(12)"
            m = EQUATION_INLINE_RE.search(t)
            eq = m.group(0).strip()
            eqno_lines.append({"text": eq, "bbox": b, "size": ln.get("size", 0.0)})

    regions = []
    for eqno in eqno_lines:
        nb = eqno["bbox"]
        y = _y_mid(nb)
        tol_y = 90  # bigger tolerance to include stacked fraction parts

        group = []
        for ln in lines:
            b = ln["bbox"]
            if abs(_y_mid(b) - y) <= tol_y:
                group.append(ln)

        # also include lines slightly above (numerator) and below (denominator)
        for ln in lines:
            b = ln["bbox"]
            dy = _y_mid(b) - y
            if -160 <= dy <= 160:
                # keep within main column (avoid headers/footers if possible)
                if float(b[0]) >= page_width * 0.05 and float(b[2]) <= page_width * 0.95:
                    group.append(ln)

        if not group:
            continue

        # region should extend left to capture formula body; right edge near eqno
        x0 = min(float(ln["bbox"][0]) for ln in group)
        y0 = min(float(ln["bbox"][1]) for ln in group)
        x1 = float(nb[2])
        y1 = max(float(ln["bbox"][3]) for ln in group)

        # clamp and pad lightly (actual crop adds more pad)
        regions.append({
            "bbox": [x0, y0, x1, y1],
            "eqno": (eqno.get("text") or "").strip()
        })

    # de-dup by rough bbox
    out = []
    seen = set()
    for r in regions:
        b = r["bbox"]
        key = (int(b[0]//5), int(b[1]//5), int(b[2]//5), int(b[3]//5))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)

    return out


# ---------------------------
# NEW: Fallback detection for formulas WITHOUT equation numbers
# (handles stacked fractions & simple displayed equations)
# ---------------------------

def detect_equations_by_layout_fallback(lines: List[Dict[str, Any]], page_width: float) -> List[Dict[str, Any]]:
    """
    For pages where there is no right-side (12) label.
    Strategy:
      - find an "anchor line" with '=' or strong math,
      - expand bbox to include nearby lines above/below with strong x-overlap
        (captures numerator/denominator).
    """
    anchors = []
    for ln in lines:
        t = (ln.get("text") or "").strip()
        b = ln.get("bbox")
        if not b:
            continue

        # likely displayed equation: not too tiny, inside main column
        w = float(b[2]) - float(b[0])
        if w < page_width * 0.18:
            continue
        if float(b[0]) < page_width * 0.04:
            continue

        if "=" in t or looks_like_formula_text(t):
            anchors.append(ln)

    regions = []
    for a in anchors:
        ab = a["bbox"]
        a_y = _y_mid(ab)
        a_h = _line_h(ab)

        # gather nearby lines with good x overlap (stacked fraction parts)
        group = [a]
        for ln in lines:
            b = ln["bbox"]
            if ln is a:
                continue
            dy = _y_mid(b) - a_y
            if abs(dy) <= max(120, 6 * a_h):
                if _overlap_x(ab, b) >= 0.35:
                    # avoid picking body text sentences
                    tt = (ln.get("text") or "").strip()
                    if len(tt) <= 120:  # equations are usually short
                        group.append(ln)

        # build region bbox
        if len(group) < 1:
            continue

        x0 = min(float(ln["bbox"][0]) for ln in group)
        y0 = min(float(ln["bbox"][1]) for ln in group)
        x1 = max(float(ln["bbox"][2]) for ln in group)
        y1 = max(float(ln["bbox"][3]) for ln in group)

        # filter out obvious normal paragraphs (too tall and too wide can be paragraph)
        w = x1 - x0
        h = y1 - y0
        if w > page_width * 0.92 and h > 160:
            continue

        regions.append({"bbox": [x0, y0, x1, y1], "eqno": ""})

    # de-dup
    out = []
    seen = set()
    for r in regions:
        b = r["bbox"]
        key = (int(b[0]//8), int(b[1]//8), int(b[2]//8), int(b[3]//8))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)

    return out


# ---------------------------
# Page scan that uses NEW detection first, then old heuristic fallback
# ---------------------------

def _scan_page_for_formula_regions(doc: fitz.Document, page_1based: int) -> List[Dict[str, Any]]:
    lines = extract_line_level_for_page(doc, page_1based)
    if not lines:
        return []

    page = doc[page_1based - 1]
    page_width = float(page.rect.width)

    # ✅ 1) Best: right-margin equation-number anchored
    anchored = detect_equations_by_right_eqno(lines, page_width)
    if anchored:
        return [{"page": page_1based, "bbox": r["bbox"], "text": r.get("eqno", "")} for r in anchored]

    # ✅ 2) Layout fallback for unlabeled equations
    layout = detect_equations_by_layout_fallback(lines, page_width)
    if layout:
        return [{"page": page_1based, "bbox": r["bbox"], "text": ""} for r in layout]

    # 3) Final fallback: old text heuristic
    flagged = []
    for ln in lines:
        t = (ln.get("text") or "").strip()
        b = ln.get("bbox")
        if not b:
            continue
        # require math-ish text + not too short
        if looks_like_formula_text(t):
            w = float(b[2]) - float(b[0])
            if w >= page_width * 0.18:
                flagged.append({"text": t, "bbox": b})

    if not flagged:
        return []

    # merge consecutive flagged lines by y proximity
    merged = []
    cur = None
    for ln in flagged:
        if cur is None:
            cur = {"bbox": ln["bbox"], "text": ln["text"]}
            continue

        prev_bbox = cur["bbox"]
        next_bbox = ln["bbox"]
        gap = float(next_bbox[1]) - float(prev_bbox[3])
        if gap >= -2 and gap <= _line_h(prev_bbox) * 2.2:
            cur["bbox"] = _bbox_union(cur["bbox"], ln["bbox"])
            cur["text"] = normalize_ws(cur["text"] + " " + ln["text"])
        else:
            merged.append(cur)
            cur = {"bbox": ln["bbox"], "text": ln["text"]}

    if cur:
        merged.append(cur)

    return [{"page": page_1based, "bbox": m["bbox"], "text": m["text"]} for m in merged]


# ---------------------------
# Blocks and selection
# ---------------------------

def load_blocks(blocks_path: Path) -> List[Dict[str, Any]]:
    return json.loads(blocks_path.read_text(encoding="utf-8"))

def select_formula_regions(doc: fitz.Document, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Priority:
      1) explicit formula blocks from Step2 (if any)
      2) relaxed block heuristic (rarely good for fractions)
      3) ✅ page scan with right-margin equation-number anchored detection
    """
    out = []

    # 1) explicit
    for b in blocks:
        if (b.get("type") or "").lower() == "formula" and b.get("page") and b.get("bbox"):
            out.append({"page": int(b["page"]), "bbox": b["bbox"], "text": (b.get("text") or "")})
    if out:
        return out

    # 2) relaxed heuristic on blocks
    for b in blocks:
        page = int(b.get("page") or 0)
        bbox = b.get("bbox")
        if not page or not bbox:
            continue
        text = (b.get("text") or "").strip()
        if looks_like_formula_text(text):
            out.append({"page": page, "bbox": bbox, "text": text})
    if out:
        return out

    # 3) page scan fallback (robust)
    for p in range(1, len(doc) + 1):
        out.extend(_scan_page_for_formula_regions(doc, p))

    return out


# ---------------------------
# Render formula image
# ---------------------------

def crop_png(
    doc: fitz.Document,
    page_1based: int,
    bbox: List[float],
    out_png: Path,
    zoom: float = 4.0,   # ✅ better for OCR
    pad: float = 10.0    # ✅ better for OCR, capture fraction bars
) -> None:
    page = doc[page_1based - 1]
    x0, y0, x1, y1 = map(float, bbox)
    rect = fitz.Rect(
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(page.rect.width, x1 + pad),
        min(page.rect.height, y1 + pad),
    )
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    out_png.write_bytes(pix.tobytes("png"))


# ---------------------------
# OCR: pix2tex (offline) - INIT ONCE
# ---------------------------

_PIX2TEX_OCR = None
_PIX2TEX_INIT_ERR = None

def _init_pix2tex_once():
    global _PIX2TEX_OCR, _PIX2TEX_INIT_ERR
    if _PIX2TEX_OCR is not None or _PIX2TEX_INIT_ERR is not None:
        return
    try:
        from pix2tex.cli import LatexOCR
        _PIX2TEX_OCR = LatexOCR()
    except Exception as e:
        _PIX2TEX_INIT_ERR = str(e)

def pix2tex_image_to_latex(image_path: Path) -> Tuple[Optional[str], Optional[str]]:
    _init_pix2tex_once()
    if _PIX2TEX_OCR is None:
        return None, f"pix2tex not available: {_PIX2TEX_INIT_ERR}"

    try:
        from PIL import Image
    except Exception as e:
        return None, f"PIL not available: {e}"

    try:
        img = Image.open(image_path).convert("RGB")
        latex = (_PIX2TEX_OCR(img) or "").strip()
        if latex:
            return latex, None
        return None, "pix2tex returned empty"
    except Exception as e:
        return None, f"pix2tex error: {e}"


# ---------------------------
# OCR: Mathpix (online fallback)
# ---------------------------

def mathpix_image_to_latex(image_path: Path, timeout_s: int = 40) -> Tuple[Optional[str], Optional[str]]:
    app_id = os.getenv("MATHPIX_APP_ID", "").strip()
    app_key = os.getenv("MATHPIX_APP_KEY", "").strip()
    if not app_id or not app_key:
        return None, "Mathpix keys missing"

    try:
        import requests
    except Exception as e:
        return None, f"requests not available: {e}"

    try:
        b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "src": f"data:image/png;base64,{b64}",
            "formats": ["latex_simplified", "asciimath"],
            "ocr": ["math", "text"],
            "skip_recrop": True
        }
        headers = {"app_id": app_id, "app_key": app_key, "Content-type": "application/json"}
        url = "https://api.mathpix.com/v3/text"
        r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        if r.status_code != 200:
            return None, f"Mathpix HTTP {r.status_code}: {r.text[:300]}"
        data = r.json()
        latex = (data.get("latex_simplified") or data.get("latex") or "").strip()
        if latex:
            return latex, None
        asciimath = (data.get("asciimath") or "").strip()
        if asciimath:
            return asciimath, None
        return None, "Mathpix returned no latex"
    except Exception as e:
        return None, f"Mathpix error: {e}"


# ---------------------------
# Quality gate
# ---------------------------

def latex_quality_ok(latex: str) -> bool:
    if not latex:
        return False
    t = latex.strip()
    if len(t) < 3:
        return False
    if t.count("?") >= 2:
        return False
    return True


# ---------------------------
# LaTeX -> Spoken -> SSML (minimal)
# ---------------------------

def latex_to_spoken(latex: str) -> str:
    s = (latex or "").strip()
    s = s.replace("$$", "").replace("$", "")
    s = s.replace("=", " equals ")
    return normalize_ws(s)

def escape_ssml(text: str) -> str:
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))

def spoken_to_ssml(spoken: str) -> str:
    t = normalize_ws(spoken)
    if not t:
        return "<speak></speak>"
    t = escape_ssml(t)
    return f"<speak>{t}</speak>"


# ---------------------------
# Injection into blocks
# ---------------------------

def inject_formula_tokens_into_blocks(blocks: List[Dict[str, Any]], formula_index: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = list(blocks)
    for it in formula_index:
        fid = it["formula_id"]
        out.append({
            "type": "formula",
            "page": it["page"],
            "bbox": it.get("bbox"),
            "text": f"[[FORMULA|{fid}|]]"
        })
    return out


# ---------------------------
# Main pipeline
# ---------------------------

def run_pipeline(pdf_path: Path, blocks_in: Path, blocks_out: Path, out_dir: Path, zoom: float = 4.0):
    doc = fitz.open(str(pdf_path))
    blocks = load_blocks(blocks_in)

    print(f"[Step3b] blocks loaded: {len(blocks)}")

    formulas_dir = out_dir / "formulas"
    ensure_dir(formulas_dir)

    cache_path = out_dir / "formula_ocr_cache.json"
    cache = load_json(cache_path, default={})

    candidates = select_formula_regions(doc, blocks)
    print(f"[Step3b] formula candidates: {len(candidates)}")

    formula_index: List[Dict[str, Any]] = []
    formulas_full: Dict[str, Any] = {}

    seq = 0
    for cand in candidates:
        seq += 1
        try:
            page = int(cand["page"])
            bbox = cand["bbox"]
            raw_text = normalize_ws(cand.get("text", ""))

            # use equation number (if present) for stable id, else seq
            eqno = ""
            if raw_text and EQUATION_NO_RE.match(raw_text.strip()):
                eqno = re.findall(r"\d+", raw_text)
                if eqno:
                    fid = f"F{eqno[0]}"
                else:
                    fid = f"F{seq}"
            else:
                fid = f"F{seq}"

            # ensure uniqueness (same formula number can appear multiple times)
            fid_unique = f"{fid}_p{page}_{seq}"

            img_path = formulas_dir / f"{fid_unique}.png"

            # ✅ better crop defaults for your PDF style
            crop_png(doc, page, bbox, img_path, zoom=zoom, pad=10.0)

            img_hash = sha256_file(img_path)
            cached = cache.get(img_hash)

            if cached and cached.get("latex"):
                latex = cached["latex"]
                provider = cached.get("provider", "cache")
            else:
                latex, _ = pix2tex_image_to_latex(img_path)
                provider = "pix2tex" if latex else None

                if not latex_quality_ok(latex or ""):
                    latex = None
                    provider = None

                if not latex:
                    latex2, _ = mathpix_image_to_latex(img_path)
                    if latex2 and latex_quality_ok(latex2):
                        latex = latex2
                        provider = "mathpix"
                    else:
                        latex = None

                if not latex:
                    latex = raw_text or ""
                    provider = "fallback_text"

                if provider in {"pix2tex", "mathpix"} and latex:
                    cache[img_hash] = {"latex": latex, "provider": provider, "ts": int(time.time())}

            spoken = latex_to_spoken(latex)
            ssml = spoken_to_ssml(spoken)

            formula_index.append({
                "formula_id": fid_unique,
                "page": page,
                "bbox": bbox,
                "latex": latex
            })

            formulas_full[fid_unique] = {
                "page": page,
                "bbox": bbox,
                "image": str(img_path.relative_to(out_dir)),
                "latex": latex,
                "provider": provider,
                "spoken": spoken,
                "ssml": ssml
            }

        except Exception as e:
            print(f"[Step3b] ERROR cand={cand} err={e}")
            traceback.print_exc()
            continue

    save_json(out_dir / "formula_index.json", formula_index)
    save_json(out_dir / "formulas.json", formulas_full)
    save_json(cache_path, cache)

    out_blocks = inject_formula_tokens_into_blocks(blocks, formula_index)
    save_json(blocks_out, out_blocks)

    print(f"[Step3b] saved formula_index: {out_dir / 'formula_index.json'} ({len(formula_index)})")
    print(f"[Step3b] saved blocks_out: {blocks_out}")


def legacy(pdf_path: Path, blocks_path: Path, out_path: Path, zoom: float = 4.0):
    out_dir = out_path.parent
    blocks_out = out_dir / "out_blocks_with_formulas.json"
    run_pipeline(pdf_path, blocks_path, blocks_out, out_dir, zoom=zoom)
    (out_dir / "formulas.json").replace(out_path)


# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to PDF")

    # new args (streamlit)
    ap.add_argument("--blocks_in", help="Input blocks JSON path")
    ap.add_argument("--blocks_out", help="Output blocks JSON path (with formula tokens)")
    ap.add_argument("--out_dir", help="Job dir to write formula_index.json etc")

    # legacy args
    ap.add_argument("--blocks", help="Legacy input blocks")
    ap.add_argument("--out", help="Legacy output formulas.json")

    ap.add_argument("--zoom", type=float, default=4.0)

    args = ap.parse_args()
    pdf = Path(args.pdf)

    if args.blocks_in and args.blocks_out and args.out_dir:
        run_pipeline(pdf, Path(args.blocks_in), Path(args.blocks_out), Path(args.out_dir), zoom=float(args.zoom))
    elif args.blocks and args.out:
        legacy(pdf, Path(args.blocks), Path(args.out), zoom=float(args.zoom))
    else:
        raise SystemExit("Provide either (--blocks_in --blocks_out --out_dir) OR (--blocks --out)")
