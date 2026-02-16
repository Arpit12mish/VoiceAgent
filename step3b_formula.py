# step3b_formula.py
"""
Step3b: Formula narration for Research PDFs (Hybrid)
Flow:
PDF → detect formula region → render image → LaTeX OCR (pix2tex first, Mathpix fallback)
→ LaTeX → spoken math → SSML

USAGE:
  python step3b_formula.py --pdf ISO9735.pdf --blocks out_blocks.json --out formulas.json

ENV (for Mathpix fallback):
  MATHPIX_APP_ID=...
  MATHPIX_APP_KEY=...

OUTPUT (formulas.json):
{
  "F3": {
    "page": 12,
    "bbox": [x0,y0,x1,y1],
    "image": "formulas/F3_p12.png",
    "latex": "...",
    "provider": "pix2tex" | "mathpix" | "fallback_text",
    "spoken": "...",
    "ssml": "<speak>...</speak>"
  },
  ...
}
"""

import base64
import hashlib
import json
import os
import re
import time
import unicodedata
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
# Formula detection (light)
# ---------------------------

EQUATION_NO_RE = re.compile(r"^\(\s*\d{1,4}\s*\)$")
HAS_MATH_RE = re.compile(r"[=±×*/^_∑∏√∫≈≠≤≥→←↔∞αβγΔμΩπθλ]")

def looks_like_formula_text(text: str) -> bool:
    t = normalize_ws(norm_unicode(text))
    if not t:
        return False
    if len(t) > 240:
        return False
    if EQUATION_NO_RE.match(t):
        return True
    if HAS_MATH_RE.search(t):
        return True
    if re.search(r"\b[A-Za-z]{1,3}\s*[_]?\s*[A-Za-z0-9]{0,3}\s*=", t):
        return True
    return False

def is_centerish(x0: float, x1: float, page_width: float, tol_ratio: float = 0.22) -> bool:
    mid = (x0 + x1) / 2.0
    return abs(mid - (page_width / 2.0)) <= page_width * tol_ratio

def extract_line_level_for_page(doc: fitz.Document, pno_1based: int) -> List[Dict[str, Any]]:
    page = doc[pno_1based - 1]
    data = page.get_text("dict")
    lines = []
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            texts = []
            xs, ys, xe, ye = [], [], [], []
            for sp in spans:
                txt = norm_unicode(sp.get("text", ""))
                if txt.strip():
                    texts.append(txt)
                bbox = sp.get("bbox")
                if bbox:
                    xs.append(bbox[0]); ys.append(bbox[1]); xe.append(bbox[2]); ye.append(bbox[3])
            if not texts:
                continue
            text = normalize_ws("".join(texts))
            if xs and ys and xe and ye:
                bbox = [min(xs), min(ys), max(xe), max(ye)]
            else:
                bbox = [0, 0, 0, 0]
            lines.append({"text": text, "bbox": bbox})
    return lines

def infer_equation_number_from_nearby(
    lines: List[Dict[str, Any]],
    formula_bbox: List[float],
    page_width: float
) -> Optional[str]:
    x0, y0, x1, y1 = map(float, formula_bbox)
    y_mid = (y0 + y1) / 2.0

    candidates = []
    for ln in lines:
        t = (ln.get("text") or "").strip()
        if not EQUATION_NO_RE.match(t):
            continue
        bx0, by0, bx1, by1 = ln.get("bbox", [0, 0, 0, 0])
        by_mid = (by0 + by1) / 2.0
        if abs(by_mid - y_mid) <= 18 and bx0 >= page_width * 0.70:
            candidates.append((bx0, t))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


# ---------------------------
# Render formula image
# ---------------------------

def crop_png(
    doc: fitz.Document,
    page_1based: int,
    bbox: List[float],
    out_png: Path,
    zoom: float = 3.0,
    pad: float = 2.0
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
# OCR: pix2tex (offline)
# ---------------------------

def pix2tex_image_to_latex(image_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns: (latex, error)
    """
    try:
        from PIL import Image
        from pix2tex.cli import LatexOCR
    except Exception as e:
        return None, f"pix2tex not available: {e}"

    try:
        ocr = LatexOCR()
        img = Image.open(image_path).convert("RGB")
        latex = (ocr(img) or "").strip()
        if latex:
            return latex, None
        return None, "pix2tex returned empty"
    except Exception as e:
        return None, f"pix2tex error: {e}"


# ---------------------------
# OCR: Mathpix (online fallback)
# ---------------------------

def mathpix_image_to_latex(image_path: Path, timeout_s: int = 40) -> Tuple[Optional[str], Optional[str]]:
    """
    Uses Mathpix OCR (API).
    Needs env vars:
      MATHPIX_APP_ID, MATHPIX_APP_KEY

    Returns: (latex, error)
    """
    app_id = os.getenv("MATHPIX_APP_ID", "").strip()
    app_key = os.getenv("MATHPIX_APP_KEY", "").strip()
    if not app_id or not app_key:
        return None, "Mathpix keys missing (set MATHPIX_APP_ID and MATHPIX_APP_KEY)"

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
        headers = {
            "app_id": app_id,
            "app_key": app_key,
            "Content-type": "application/json"
        }
        # Mathpix endpoint
        url = "https://api.mathpix.com/v3/text"

        r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        if r.status_code != 200:
            return None, f"Mathpix HTTP {r.status_code}: {r.text[:300]}"

        data = r.json()

        # Prefer latex_simplified
        latex = (data.get("latex_simplified") or data.get("latex") or "").strip()
        if latex:
            return latex, None

        # Sometimes asciimath exists even if latex doesn't
        asciimath = (data.get("asciimath") or "").strip()
        if asciimath:
            return asciimath, None

        return None, f"Mathpix returned no latex: keys={list(data.keys())}"
    except Exception as e:
        return None, f"Mathpix error: {e}"


# ---------------------------
# Quality gate: decide if pix2tex output is acceptable
# ---------------------------

def latex_quality_ok(latex: str) -> bool:
    """
    Heuristic:
    - reject extremely short junk
    - reject outputs dominated by '?' or single chars
    - reject if contains many unknown tokens
    """
    if not latex:
        return False
    t = latex.strip()
    if len(t) < 3:
        return False
    if t.count("?") >= 2:
        return False
    # too many spaces but no operators
    if len(t.split()) >= 8 and not re.search(r"[=\\frac\\sqrt\\sum\\int\+\-\*/_^\(\)\[\]]", t):
        return False
    return True


# ---------------------------
# LaTeX → Spoken → SSML
# ---------------------------

GREEK = {
    r"\alpha": "alpha", r"\beta": "beta", r"\gamma": "gamma", r"\delta": "delta",
    r"\Delta": "delta", r"\mu": "mu", r"\pi": "pi", r"\theta": "theta",
    r"\lambda": "lambda", r"\Omega": "omega", r"\omega": "omega",
}

LATEX_SIMPLE = {
    r"\times": " multiplied by ",
    r"\cdot": " multiplied by ",
    r"\pm": " plus or minus ",
    r"\div": " divided by ",
    r"\leq": " less than or equal to ",
    r"\geq": " greater than or equal to ",
    r"\neq": " not equal to ",
    r"\approx": " approximately equal to ",
    r"\rightarrow": " tends to ",
    r"\to": " tends to ",
}

def latex_to_spoken(latex: str) -> str:
    s = (latex or "").strip()
    if not s:
        return ""

    s = s.replace("$$", "").replace("$", "")
    s = s.replace(r"\,", " ").replace(r"\;", " ").replace(r"\:", " ")

    for k, v in GREEK.items():
        s = s.replace(k, f" {v} ")

    for k, v in LATEX_SIMPLE.items():
        s = s.replace(k, v)

    # \mathrm{mol} -> mol
    s = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", s)

    # Fractions
    def frac_repl(m):
        a = m.group(1)
        b = m.group(2)
        a_sp = latex_to_spoken(a)
        b_sp = latex_to_spoken(b)
        return f" {a_sp} divided by {b_sp} "
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", frac_repl, s)

    # Subscripts
    s = re.sub(r"([A-Za-z])_\{([^}]+)\}", r"\1 sub \2", s)
    s = re.sub(r"([A-Za-z])_([A-Za-z0-9]+)", r"\1 sub \2", s)

    # Superscripts
    s = re.sub(r"([A-Za-z0-9])\^\{([^}]+)\}", r"\1 to the power of \2", s)
    s = re.sub(r"([A-Za-z0-9])\^([A-Za-z0-9]+)", r"\1 to the power of \2", s)

    # Brackets
    s = s.replace("{", " ").replace("}", " ")
    s = s.replace("(", " open bracket ").replace(")", " close bracket ")
    s = s.replace("[", " open bracket ").replace("]", " close bracket ")

    s = s.replace("=", " equals ")

    # Remove backslashes left
    s = s.replace("\\", " ")

    # Basic chemical: CO2 -> C O 2
    s = re.sub(r"\b([A-Z]{1,3})(\d+)\b", lambda m: " ".join(list(m.group(1))) + f" {m.group(2)}", s)

    return normalize_ws(s)

def escape_ssml(text: str) -> str:
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))

def spoken_to_ssml(spoken: str) -> str:
    t = normalize_ws(spoken)
    if not t:
        return "<speak></speak>"

    # Insert light pauses around key math words (placeholders first)
    t = re.sub(r"\b(equals)\b", r" __BRK120__ \1 __BRK120__ ", t, flags=re.I)
    t = re.sub(r"\b(divided by|multiplied by|plus|minus)\b", r" __BRK90__ \1 __BRK90__ ", t, flags=re.I)

    # Escape, then restore tags
    t = escape_ssml(normalize_ws(t))
    t = t.replace("__BRK120__", '<break time="120ms"/>')
    t = t.replace("__BRK90__", '<break time="90ms"/>')

    return f"<speak>{t}</speak>"


# ---------------------------
# Load blocks & select formula regions
# ---------------------------

def load_blocks(blocks_path: Path) -> List[Dict[str, Any]]:
    return json.loads(blocks_path.read_text(encoding="utf-8"))

def select_formula_regions(doc: fitz.Document, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Priority:
    1) blocks with type == "formula" and bbox exists
    2) heuristic fallback: formula-like + centerish + bbox exists
    """
    out = []

    # 1) explicit
    for b in blocks:
        if (b.get("type") or "").lower() == "formula" and b.get("page") and b.get("bbox"):
            out.append({"page": int(b["page"]), "bbox": b["bbox"], "text": (b.get("text") or "")})
    if out:
        return out

    # 2) heuristic fallback
    for b in blocks:
        page = int(b.get("page") or 0)
        bbox = b.get("bbox")
        if not page or not bbox:
            continue
        text = (b.get("text") or "").strip()
        if not looks_like_formula_text(text):
            continue
        page_width = float(doc[page - 1].rect.width)
        x0, y0, x1, y1 = map(float, bbox)
        if is_centerish(x0, x1, page_width, tol_ratio=0.22):
            out.append({"page": page, "bbox": bbox, "text": text})

    return out


# ---------------------------
# Main hybrid pipeline with caching
# ---------------------------

def main(pdf_path: Path, blocks_path: Path, out_path: Path, zoom: float = 3.0):
    pdf_path = Path(pdf_path)
    blocks_path = Path(blocks_path)
    out_path = Path(out_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not blocks_path.exists():
        raise FileNotFoundError(f"Blocks JSON not found: {blocks_path}")

    doc = fitz.open(str(pdf_path))
    blocks = load_blocks(blocks_path)

    formulas_dir = out_path.parent / "formulas"
    ensure_dir(formulas_dir)

    cache_path = out_path.parent / "formula_ocr_cache.json"
    cache = load_json(cache_path, default={})
    # cache schema:
    # { "<sha256>": {"latex": "...", "provider": "pix2tex|mathpix", "ts": 1234567890} }

    candidates = select_formula_regions(doc, blocks)

    results: Dict[str, Any] = {}
    seq = 0

    for cand in candidates:
        seq += 1
        page = int(cand["page"])
        bbox = cand["bbox"]
        raw_text = normalize_ws(cand.get("text", ""))

        # Try infer equation number "(3)"
        page_lines = extract_line_level_for_page(doc, page)
        page_width = float(doc[page - 1].rect.width)
        eqno = infer_equation_number_from_nearby(page_lines, bbox, page_width)

        if eqno:
            fid = f"F{int(re.sub(r'[^0-9]', '', eqno))}"
        else:
            fid = f"F{seq}"

        img_path = formulas_dir / f"{fid}_p{page}.png"
        try:
            crop_png(doc, page, bbox, img_path, zoom=zoom, pad=2.0)
        except Exception:
            # skip if crop fails
            continue

        img_hash = sha256_file(img_path)

        # Cached?
        cached = cache.get(img_hash)
        if cached and cached.get("latex"):
            latex = cached["latex"]
            provider = cached.get("provider", "cache")
        else:
            # 1) pix2tex
            latex, err = pix2tex_image_to_latex(img_path)
            provider = "pix2tex" if latex else None

            # Quality gate: if pix2tex returns weak latex -> fallback
            if not latex_quality_ok(latex or ""):
                latex = None
                provider = None

            # 2) Mathpix fallback
            if not latex:
                latex2, err2 = mathpix_image_to_latex(img_path)
                if latex2 and latex_quality_ok(latex2):
                    latex = latex2
                    provider = "mathpix"
                else:
                    latex = latex2 or None
                    provider = "mathpix" if latex else None

            # 3) final fallback: use raw text
            if not latex:
                latex = raw_text or ""
                provider = "fallback_text"

            # write cache (only if pix2tex/mathpix)
            if provider in {"pix2tex", "mathpix"} and latex:
                cache[img_hash] = {"latex": latex, "provider": provider, "ts": int(time.time())}

        spoken = latex_to_spoken(latex)
        ssml = spoken_to_ssml(spoken)

        results[fid] = {
            "page": page,
            "bbox": bbox,
            "image": str(img_path.relative_to(out_path.parent)),
            "latex": latex,
            "provider": provider,
            "spoken": spoken,
            "ssml": ssml
        }

    save_json(out_path, results)
    save_json(cache_path, cache)
    print(f"Saved {len(results)} formulas to: {out_path}")
    print(f"Cache updated: {cache_path}")


# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to PDF")
    ap.add_argument("--blocks", required=True, help="Path to out_blocks.json from Step2")
    ap.add_argument("--out", required=True, help="Output formulas.json path")
    ap.add_argument("--zoom", type=float, default=3.0, help="Render zoom for formula crops")
    args = ap.parse_args()

    main(Path(args.pdf), Path(args.blocks), Path(args.out), zoom=float(args.zoom))
