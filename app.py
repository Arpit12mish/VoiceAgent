# app.py (FINAL - FIXED)
# ✅ Fix 1: build_table_index_from_blocks now dedupes TOC-vs-caption by canonical "T{n}" id
# ✅ Fix 2: render_table_explanations_ui uses rendered=set() guard (never renders textarea twice per table_id)
# ✅ Fix 3: ui_scope is REQUIRED + used in ALL widget keys + BOTH call-sites pass different ui_scope
#
# Result: No StreamlitDuplicateElementKey even when:
# - the same table is referenced in Table of Contents AND above the table
# - you render the table UI twice in same run (after-processing + job-level)

import streamlit as st
import json
import subprocess
import sys
import os
import wave
from pathlib import Path
from datetime import datetime

import fitz  # PyMuPDF
import pandas as pd
import re
import unicodedata

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)

STEP2 = ROOT / "step2_blocks.py"
STEP2C = ROOT / "step2c_sections.py"
STEP5A = ROOT / "step5a_make_text_chunks_from_sections.py"
STEP5B = ROOT / "step5b_synthesize_wav_chunks.py"
STEP6 = ROOT / "step6_merge_wavs.py"  # optional; we merge inside app.py too

# ---------------------------
# TABLE TOKENS (supports older formats too)
# 1) New format: [[TABLE|p8_t0|Caption...]]
# 2) Old format: [TABLE:p8_t0] Caption...
# ---------------------------
TABLE_TOKEN_NEW_RE = re.compile(r"\[\[TABLE\|([^|]+)\|([^\]]+)\]\]")
TABLE_TOKEN_OLD_RE = re.compile(r"^\[TABLE:([^\]]+)\]\s*(.*)$")

# canonical table number extraction: "Table 1", "TABLE 10:", "Table 2 (continued)" etc.
TABLE_NO_RE = re.compile(r"\btable\s*([0-9]{1,4})\b", re.IGNORECASE)


# ---------------------------
# PDF caching + table image crop
# ---------------------------
@st.cache_resource(show_spinner=False)
def _open_pdf_cached(pdf_path_str: str):
    return fitz.open(pdf_path_str)


def crop_table_png_bytes(pdf_path: Path, page_num_1based: int, bbox, zoom: float = 2.0) -> bytes:
    """
    Crop a table region from pdf using bbox (fitz coords), return PNG bytes.
    bbox: [x0,y0,x1,y1]
    """
    doc = _open_pdf_cached(str(pdf_path))
    page = doc[page_num_1based - 1]
    rect = fitz.Rect(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

    # small padding to avoid cutting borders
    pad = 2.0
    rect = fitz.Rect(
        max(0, rect.x0 - pad),
        max(0, rect.y0 - pad),
        min(page.rect.width, rect.x1 + pad),
        min(page.rect.height, rect.y1 + pad),
    )

    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    return pix.tobytes("png")


# ---------------------------
# Canonical table_id logic
# ---------------------------
def canonical_table_id(raw_table_id: str | None, caption: str | None) -> str | None:
    """
    Unifies TOC entry and real table caption into ONE logical id:
      - If caption contains "Table N" => returns "T{N}"
      - Else if raw_table_id contains "T{N}" pattern => returns that "T{N}"
      - Else falls back to raw_table_id (p8_t0 etc.)

    This is the key fix for "Table of contents" vs "Above the table" duplication.
    """
    rid = (raw_table_id or "").strip()
    cap = (caption or "").strip()

    # 1) Prefer caption "Table N"
    m = TABLE_NO_RE.search(cap)
    if m:
        return f"T{int(m.group(1))}"

    # 2) Try raw id containing Table N or Tn
    #    e.g. "20260215_210600_T1" or "job_..._T12"
    m2 = re.search(r"\bT([0-9]{1,4})\b", rid, flags=re.IGNORECASE)
    if m2:
        return f"T{int(m2.group(1))}"

    m3 = TABLE_NO_RE.search(rid)
    if m3:
        return f"T{int(m3.group(1))}"

    # 3) Fallback
    return rid or None


# ---------------------------
# Table extraction/grouping from blocks (FIX 1)
# ---------------------------
def build_table_index_from_blocks(blocks):
    """
    Returns dict keyed by logical table_id (canonical):
      {
        "T10": {
           "caption": "Table 10 — ...",
           "occurrences": [
              {"page": 13, "bbox": [...], "hit_id": "p13_t0", "caption": "..."},
              {"page": 14, "bbox": [...], "hit_id": "p14_t0", "caption": "Table 10 (continued)"},
           ]
        },
        ...
      }

    Supports:
    - step2 explicit blocks: type=table with table_id/caption/bbox/hit_id/page
    - older inline tokens fallback (no bbox -> screenshots not possible)

    IMPORTANT:
    - Canonicalizes TOC-vs-caption duplication into one id via canonical_table_id()
    - Dedupes identical occurrences (same page + bbox) defensively
    """
    idx = {}

    def _ensure(tid: str):
        if tid not in idx:
            idx[tid] = {"caption": "", "occurrences": []}

    def _add_occ(tid: str, occ: dict):
        # Defensive de-dupe: same page + same bbox (or both None) + same hit_id
        page = int(occ.get("page") or 0)
        bbox = occ.get("bbox")
        hit_id = (occ.get("hit_id") or "").strip()
        key = (page, tuple(map(float, bbox)) if bbox else None, hit_id)

        existing_keys = idx[tid].setdefault("_seen_keys", set())
        if key in existing_keys:
            return
        existing_keys.add(key)
        idx[tid]["occurrences"].append(occ)

    for b in blocks:
        text = (b.get("text") or "").strip()

        # Best case: Step2 explicit fields
        if b.get("type") == "table":
            raw_tid = str(b.get("table_id") or "").strip()
            caption = (b.get("caption") or b.get("text") or "").strip()
            page = int(b.get("page") or 0)
            bbox = b.get("bbox")
            hit_id = (b.get("hit_id") or raw_tid or "").strip()  # preserve original as hit_id

            tid = canonical_table_id(raw_tid, caption)
            if tid:
                _ensure(tid)

                # prefer the "real" caption (longer, non-empty)
                if caption:
                    if not idx[tid]["caption"] or len(caption) > len(idx[tid]["caption"]):
                        idx[tid]["caption"] = caption

                _add_occ(tid, {"page": page, "bbox": bbox, "hit_id": hit_id, "caption": caption})
            continue

        # Fallback 1: token inside text (no bbox)
        for m in TABLE_TOKEN_NEW_RE.finditer(text):
            raw_tid = m.group(1).strip()
            caption = m.group(2).strip()
            tid = canonical_table_id(raw_tid, caption)
            if tid:
                _ensure(tid)
                if caption and (not idx[tid]["caption"] or len(caption) > len(idx[tid]["caption"])):
                    idx[tid]["caption"] = caption
                _add_occ(tid, {
                    "page": int(b.get("page") or 0),
                    "bbox": None,
                    "hit_id": raw_tid,
                    "caption": caption
                })

        # Fallback 2: old whole-line placeholder (no bbox)
        m2 = TABLE_TOKEN_OLD_RE.match(text)
        if m2:
            raw_tid = (m2.group(1) or "").strip()
            caption = (m2.group(2) or "").strip()
            tid = canonical_table_id(raw_tid, caption)
            if tid:
                _ensure(tid)
                if caption and (not idx[tid]["caption"] or len(caption) > len(idx[tid]["caption"])):
                    idx[tid]["caption"] = caption
                _add_occ(tid, {
                    "page": int(b.get("page") or 0),
                    "bbox": None,
                    "hit_id": raw_tid,
                    "caption": caption
                })

    # Clean helper key
    for tid in list(idx.keys()):
        idx[tid].pop("_seen_keys", None)

    # Stable sort occurrences: by page then y0 if bbox
    def occ_sort_key(o):
        y = 0.0
        if o.get("bbox"):
            try:
                y = float(o["bbox"][1])
            except Exception:
                y = 0.0
        return (int(o.get("page") or 0), y)

    items = []
    for tid, v in idx.items():
        v["occurrences"] = sorted(v["occurrences"], key=occ_sort_key)
        items.append((tid, v))

    # Sort tables by first occurrence page then id
    items.sort(key=lambda kv: (kv[1]["occurrences"][0]["page"] if kv[1]["occurrences"] else 10**9, kv[0]))
    return dict(items)


def load_table_explanations(job_dir: Path):
    path = job_dir / "table_explanations.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_table_explanations(job_dir: Path, data: dict):
    path = job_dir / "table_explanations.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------
# UI: Table explanations (FIX 2 + FIX 3)
# ---------------------------
def render_table_explanations_ui(
    job_dir: Path,
    pdf_path: Path,
    blocks_data: list,
    title="🧾 Table Explanations",
    ui_scope: str = "job"   # ✅ FIX 3: scope is part of widget keys
):
    """
    UI:
    - shows all detected table placeholders (per occurrence)
    - shows screenshot crop if bbox exists
    - one textarea per logical table_id (canonical)
    """

    st.divider()
    st.header(title)

    table_index = build_table_index_from_blocks(blocks_data)

    if not table_index:
        st.info("No tables detected for this job.")
        return

    existing = load_table_explanations(job_dir)
    updated = dict(existing)

    st.caption("Write the explanation ONCE per table_id. The same explanation will be reused for continued pages.")
    st.info(f"Detected {len(table_index)} unique table_id(s).")

    # ✅ FIX 2: never render a textarea twice per table_id within a run (defensive)
    rendered = set()

    for table_id, meta in table_index.items():
        if table_id in rendered:
            continue
        rendered.add(table_id)

        caption = (meta.get("caption") or "").strip()
        occs = meta.get("occurrences") or []
        occ_count = len(occs)

        st.subheader(f"{table_id}  •  {occ_count} occurrence(s)")
        if caption:
            st.caption(caption)

        with st.expander("📌 Show table placeholders & screenshots", expanded=False):
            for i, occ in enumerate(occs, start=1):
                page = occ.get("page") or "?"
                oc_caption = (occ.get("caption") or "").strip()
                hit_id = (occ.get("hit_id") or "").strip()

                left, right = st.columns([1, 2], vertical_alignment="top")
                with left:
                    st.markdown(f"**Occurrence {i}**")
                    st.write(f"Page: **{page}**")
                    if hit_id:
                        st.write(f"hit_id: `{hit_id}`")
                with right:
                    if oc_caption:
                        st.code(f"[TABLE:{table_id}] {oc_caption}", language="text")
                    else:
                        st.code(f"[TABLE:{table_id}] (no caption)", language="text")

                    bbox = occ.get("bbox")
                    if bbox and pdf_path and pdf_path.exists():
                        try:
                            png = crop_table_png_bytes(pdf_path, int(page), bbox, zoom=2.0)
                            st.image(png, caption=f"{table_id} • page {page}", use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not render screenshot for {table_id} on page {page}: {e}")
                    else:
                        st.info("No bbox available for this occurrence (cannot crop screenshot).")

        # ✅ FIX 3: key includes ui_scope + job + table_id
        default_text = existing.get(table_id, "")
        explanation = st.text_area(
            f"Explanation for {table_id}",
            value=default_text,
            height=140,
            key=f"table_exp_{ui_scope}_{job_dir.name}_{table_id}"
        )
        updated[table_id] = (explanation or "").strip()

    cA, cB = st.columns(2)

    # ✅ FIX 3: button keys include ui_scope too
    if cA.button("💾 Save Table Explanations", key=f"save_table_expl_{ui_scope}_{job_dir.name}"):
        save_table_explanations(job_dir, updated)
        st.success(f"Saved: {job_dir / 'table_explanations.json'}")

    if cB.button("🧹 Clear all explanations", key=f"clear_table_expl_{ui_scope}_{job_dir.name}"):
        save_table_explanations(job_dir, {})
        st.warning("Cleared. (table_explanations.json reset)")


# ---------------------------
# Helpers
# ---------------------------
def run_cmd(cmd, cwd=None, env=None):
    """
    Run a command and return (ok, stdout+stderr).
    """
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
            env=env
        )
        out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
        return (p.returncode == 0), out.strip()
    except Exception as e:
        return False, f"Exception running command: {e}"


def safe_name(s: str, max_len=80) -> str:
    s = (s or "").strip().replace(" ", "_")
    s = "".join(ch for ch in s if ch.isalnum() or ch in "_-")
    return s[:max_len] if s else "section"


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def ensure_normalized_and_ssml(job_dir: Path, sections: list):
    """
    If sections_normalized/ or sections_ssml_preview/ is missing,
    generate them from sections_original/ using the existing normalize/ssml functions.
    """
    orig_dir = job_dir / "sections_original"
    sections_norm = job_dir / "sections_normalized"
    sections_ssml = job_dir / "sections_ssml_preview"

    # Create dirs if missing
    sections_norm.mkdir(exist_ok=True)
    sections_ssml.mkdir(exist_ok=True)

    # If already populated, skip
    already_has_norm = any(sections_norm.glob("*.txt"))
    already_has_ssml = any(sections_ssml.glob("*.xml"))

    if already_has_norm and already_has_ssml:
        return

    # Generate missing files
    for s in sections:
        sec_file = Path(s["file"])
        if not sec_file.exists():
            sec_file = orig_dir / Path(s["file"]).name

        raw = sec_file.read_text(encoding="utf-8")
        norm_text = normalize_section_text(raw)

        (sections_norm / sec_file.name).write_text(norm_text, encoding="utf-8")
        (sections_ssml / f"{sec_file.stem}.xml").write_text(
            text_to_ssml_preview(norm_text),
            encoding="utf-8"
        )


def detect_header_footer_candidates(
    pdf_path: Path,
    top_pct=0.08,
    bottom_pct=0.92,
    max_len=160,
    min_pages_ratio=0.6,
    sample_pages=6
):
    """
    Detect repeated header/footer lines in top/bottom bands.
    Also returns common page-number regex suggestions.
    """
    doc = fitz.open(str(pdf_path))
    pages = len(doc)

    seen = {}         # (band, text) -> set(pages)
    page_samples = {} # (band, text) -> [page nos]

    for pno in range(pages):
        page = doc[pno]
        h = float(page.rect.height)

        data = page.get_text("dict")
        for block in data.get("blocks", []):
            if block.get("type", 0) != 0:
                continue

            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue

                parts = []
                ys = []
                for sp in spans:
                    t = (sp.get("text", "") or "").replace("\u00ad", "")
                    if t.strip():
                        parts.append(t)
                        bbox = sp.get("bbox")
                        if bbox:
                            ys.append(bbox[1])

                if not parts:
                    continue

                text = normalize_ws("".join(parts))
                if not text or len(text) > max_len:
                    continue

                y0 = min(ys) if ys else 0.0

                if y0 <= h * top_pct:
                    band = "TOP"
                elif y0 >= h * bottom_pct:
                    band = "BOTTOM"
                else:
                    continue

                key = (band, text)
                seen.setdefault(key, set()).add(pno + 1)

                if key not in page_samples:
                    page_samples[key] = []
                if len(page_samples[key]) < sample_pages:
                    page_samples[key].append(pno + 1)

    rows = []
    for (band, text), pset in seen.items():
        ratio = (len(pset) / pages) if pages else 0.0
        if ratio >= min_pages_ratio and len(text) >= 2:
            rows.append({
                "remove": True,
                "band": band,
                "text": text,
                "pages_count": len(pset),
                "pages_ratio": round(ratio, 2),
                "sample_pages": ", ".join(map(str, page_samples.get((band, text), [])[:sample_pages]))
            })

    page_number_regex = [
        r"^\s*\d+\s*$",
        r"^\s*page\s*\d+\s*$",
        r"^\s*\d+\s*/\s*\d+\s*$",
        r"^\s*-\s*\d+\s*-\s*$",
        r"^\s*\d{1,4}\s*$",
        r"^\s*[ivxlcdm]{1,8}\s*$",
        r"^\s*page\s*\d{1,4}\s*$",
        r"^\s*\d{1,4}\s*/\s*\d{1,4}\s*$",
        r"^\s*\d{1,4}\s+of\s+\d{1,4}\s*$",
        r"^\s*-\s*\d{1,4}\s*-\s*$",
        r"^\s*\d{1,4}\s*[\|\-]\s*\d{1,4}\s*$",
        r"^\s*\d{1,4}\s*(?:BS|BIS)\s*$",
        r"^\s*\d{1,4}\s*(?:BS|BIS)\s*[^\w\s]?\s*$",
        r"^\s*(?:BS|BIS)\s*\d{1,4}\s*$",
        r"^\s*\d{1,4}(?:BS|BIS)\s*$",
        r"^\s*\d{1,4}\s*[\.\-\|\u00b7:]?\s*(?:BS|BIS)\s*$",
    ]

    return rows, page_number_regex


def merge_wavs_in_order(wav_files, out_path: Path):
    """
    Merge WAV files in order using wave module (no ffmpeg needed).
    All chunks must have same params.
    """
    wav_files = [Path(p) for p in wav_files if Path(p).exists()]
    if not wav_files:
        raise FileNotFoundError("No WAV files found to merge.")

    with wave.open(str(wav_files[0]), "rb") as w0:
        params = w0.getparams()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as out:
        out.setparams(params)
        for f in wav_files:
            with wave.open(str(f), "rb") as w:
                if w.getparams() != params:
                    raise ValueError(f"WAV params mismatch: {f.name}")
                out.writeframes(w.readframes(w.getnframes()))



# ---------------------------
# Step3 normalize + Step4 SSML preview helpers
# ---------------------------
LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
}

ABBREV = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.",
    "e.g.", "i.e.", "etc.", "vs.", "no.", "fig.", "eq.",
    "inc.", "ltd.", "pvt.", "co.", "dept.", "gov."
}

TOC_START_TITLES = {"contents", "table of contents"}


def norm_unicode(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    for k, v in LIGATURES.items():
        s = s.replace(k, v)
    s = s.replace("\u00ad", "")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = s.replace("–", "-").replace("—", "-")
    return s


def fix_decimal_spacing(s: str) -> str:
    return re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", s or "")


def remove_dot_leaders(s: str) -> str:
    s = re.sub(r"\.{5,}", " ", s or "")
    return normalize_ws(s)


PAGE_MARKER_PATTERNS = [
    r"^\s*\d{1,4}\s*$",
    r"^\s*[ivxlcdm]{1,8}\s*$",
    r"^\s*page\s*\d{1,4}\s*$",
    r"^\s*\d{1,4}\s*/\s*\d{1,4}\s*$",
    r"^\s*\d{1,4}\s+of\s+\d{1,4}\s*$",
    r"^\s*-\s*\d{1,4}\s*-\s*$",
    r"^\s*\d{1,4}\s*[\|\-]\s*\d{1,4}\s*$",
    r"^\s*\d{1,4}\s*(?:BS|BIS)\s*$",
    r"^\s*\d{1,4}\s*(?:BS|BIS)\s*[^\w\s]?\s*$",
    r"^\s*(?:BS|BIS)\s*\d{1,4}\s*$",
    r"^\s*\d{1,4}(?:BS|BIS)\s*$",
    r"^\s*\d{1,4}\s*[\.\-\|\u00b7:]?\s*(?:BS|BIS)\s*$",
]


def looks_like_page_marker(s: str) -> bool:
    t = (s or "").strip().lower()
    if t in {"page"}:
        return True
    for pat in PAGE_MARKER_PATTERNS:
        if re.fullmatch(pat, t, flags=re.IGNORECASE):
            return True
    return False


def is_boilerplate_line(s: str) -> bool:
    t = (s or "").strip().lower()
    boiler = [
        "for bis use only",
        "copyright protected document",
        "all rights reserved",
        "© iso",
        "published in switzerland",
    ]
    return any(b in t for b in boiler)


def drop_toc_from_text(text: str) -> str:
    lines = [ln.rstrip() for ln in (text or "").splitlines()]

    def is_toc_start(line: str) -> bool:
        return line.strip().lower() in TOC_START_TITLES

    def looks_like_toc_entry(line: str) -> bool:
        t = line.strip()
        if not t:
            return False
        if re.search(r"\.{5,}", t):
            return True
        return bool(re.search(r"\s(\d{1,4}|[ivxlcdm]{1,8})\s*$", t.lower()))

    def looks_like_paragraph(line: str) -> bool:
        t = line.strip()
        return bool(len(t.split()) >= 10 and re.search(r"[a-zA-Z]", t))

    toc_start_idx = None
    for i, ln in enumerate(lines[:400]):
        if is_toc_start(ln):
            toc_start_idx = i
            break
    if toc_start_idx is None:
        return "\n".join(lines).strip()

    toc_like_count = 0
    for ln in lines[toc_start_idx + 1: toc_start_idx + 120]:
        if looks_like_toc_entry(ln):
            toc_like_count += 1
    if toc_like_count < 5:
        return "\n".join(lines).strip()

    out = []
    in_toc = False

    i = 0
    while i < len(lines):
        ln = lines[i]

        if is_toc_start(ln):
            in_toc = True
            i += 1
            continue

        if in_toc:
            if looks_like_toc_entry(ln) or re.fullmatch(r"\d+(\.\d+)+", ln.strip()):
                i += 1
                continue

            if looks_like_paragraph(ln):
                in_toc = False
                out.append(ln)
                i += 1
                continue

            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and looks_like_paragraph(lines[j]):
                in_toc = False
                out.append(ln)
                i += 1
                continue

            i += 1
            continue

        out.append(ln)
        i += 1

    return "\n".join(out).strip()


def split_sentences(text: str):
    t = text or ""
    t = re.sub(r"(\d)\.(\d)", r"\1<DEC>\2", t)

    for ab in ABBREV:
        ab_esc = re.escape(ab[:-1])
        t = re.sub(
            rf"(?i)\b{ab_esc}\.",
            lambda m: m.group(0).replace(".", "<ABBR>"),
            t
        )

    parts = re.split(r"(?<=[\.\?\!])\s+", t)

    out = []
    for p in parts:
        p = p.replace("<DEC>", ".").replace("<ABBR>", ".")
        p = normalize_ws(p)
        if p:
            out.append(p)
    return out


# -------------------------------
# ✅ NEW: Plural-aware unit normalization used by app.py Step-3
# -------------------------------
def normalize_decimal_commas(text: str) -> str:
    # 0,01 -> 0.01 (only between digits)
    return re.sub(r"(?<=\d),(?=\d)", ".", text or "")


def _is_plural_number(num_str: str) -> bool:
    try:
        v = float(num_str)
        return abs(v - 1.0) > 1e-12
    except Exception:
        return True


def _maybe_pluralize(unit_word: str, plural: bool) -> str:
    if not plural:
        return unit_word
    parts = unit_word.split()
    if not parts:
        return unit_word
    last = parts[-1]
    if last.endswith("s"):
        return unit_word
    if last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
        parts[-1] = last[:-1] + "ies"
    else:
        parts[-1] = last + "s"
    return " ".join(parts)


def normalize_symbols_units_plural(text: str) -> str:
    """
    Converts:
      250 ml  -> 250 millilitres
      30 g    -> 30 grams
      0,3 cm  -> 0.3 centimetres
      1,5 cm  -> 1.5 centimetres
      250 cm3 / cm³ / cm^3 -> cubic centimetre(s)
      mol/l   -> mole per litre
    """
    if not text:
        return text

    text = normalize_decimal_commas(text)

    # normalize special glyph units and superscripts
    text = (text
            .replace("㎜", "mm")
            .replace("㎝", "cm")
            .replace("㎞", "km")
            .replace("㎡", "m2")
            .replace("㎟", "mm2")
            .replace("㎠", "cm2")
            .replace("㎢", "km2")
            .replace("㎥", "m3")
            .replace("㎣", "mm3")
            .replace("㎤", "cm3")
            .replace("²", "2")
            .replace("³", "3")
            )

    # direct safe replacements
    safe_direct = {
        "°C": " degree celsius ",
        "°c": " degree celsius ",
        "°F": " degree fahrenheit ",
        "°f": " degree fahrenheit ",
        "μm": " micrometer ",
        "ppm": " parts per million ",
        "ppb": " parts per billion ",
    }
    for k, v in safe_direct.items():
        text = text.replace(k, v)

    unit_volume = {"mm": "cubic millimetre", "cm": "cubic centimetre", "m": "cubic metre", "km": "cubic kilometre"}
    unit_area   = {"mm": "square millimetre", "cm": "square centimetre", "m": "square metre", "km": "square kilometre"}
    unit_length = {"mm": "millimetre", "cm": "centimetre", "m": "metre", "km": "kilometre"}
    unit_mass   = {"mg": "milligram", "g": "gram", "kg": "kilogram"}
    unit_liquid = {"ml": "millilitre", "l": "litre"}

    def repl_powered(m, kind_map):
        num = m.group("num")
        unit = m.group("unit").lower()
        plural = _is_plural_number(num)
        base = kind_map.get(unit, unit)
        return f"{num} {_maybe_pluralize(base, plural)}"

    def repl_plain(m, kind_map):
        num = m.group("num")
        unit = m.group("unit").lower()
        plural = _is_plural_number(num)
        base = kind_map.get(unit, unit)
        return f"{num} {_maybe_pluralize(base, plural)}"

    # volume: 250 cm3, 250 cm^3, 250 cm 3
    text = re.sub(
        r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m|km)\s*(?:\^?\s*3)\b",
        lambda m: repl_powered(m, unit_volume),
        text,
        flags=re.IGNORECASE
    )

    # area: 10 cm2 etc
    text = re.sub(
        r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m|km)\s*(?:\^?\s*2)\b",
        lambda m: repl_powered(m, unit_area),
        text,
        flags=re.IGNORECASE
    )

    # length: 0.3 cm, 5 cm
    text = re.sub(
        r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m|km)\b",
        lambda m: repl_plain(m, unit_length),
        text,
        flags=re.IGNORECASE
    )

    # mass: 30 g
    text = re.sub(
        r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>mg|g|kg)\b",
        lambda m: repl_plain(m, unit_mass),
        text,
        flags=re.IGNORECASE
    )

    # liquids: 250 ml
    text = re.sub(
        r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>ml|l)\b",
        lambda m: repl_plain(m, unit_liquid),
        text,
        flags=re.IGNORECASE
    )

    # mol/l (common in standards)
    text = re.sub(r"(?i)\bmol\s*/\s*l\b", " mole per litre ", text)

    return re.sub(r"\s+", " ", text).strip()


def normalize_section_text(text: str) -> str:
    text = norm_unicode(text)
    text = drop_toc_from_text(text)

    lines = text.splitlines()
    kept = []
    for ln in lines:
        t = remove_dot_leaders(ln)
        t = fix_decimal_spacing(t)

        # ✅ THE MISSING STEP (unit conversion happens here)
        t = normalize_symbols_units_plural(t)

        t = normalize_ws(t)

        if not t:
            kept.append("")
            continue

        if looks_like_page_marker(t) or is_boilerplate_line(t):
            continue

        kept.append(t)

    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def escape_ssml(text: str) -> str:
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))


def text_to_ssml_preview(text: str) -> str:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    parts = []
    for p in paras:
        sents = split_sentences(p)
        inner = " ".join(escape_ssml(s) for s in sents)
        parts.append(inner + '<break time="350ms"/>')
    return "<speak>\n" + "\n".join(parts) + "\n</speak>"



# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="VoiceAgent - PDF → Sections → Speech", layout="wide")
st.title("📘 VoiceAgent (Upload PDF → Edit Sections → Generate Speech)")

# ---------------------------
# 1) Upload PDF
# ---------------------------
st.header("1) Upload PDF")
uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

# Persist job folder across reruns
if "current_job_dir" not in st.session_state:
    st.session_state["current_job_dir"] = None
if "current_pdf_name" not in st.session_state:
    st.session_state["current_pdf_name"] = None

job_dir = None
pdf_path = None

if uploaded:
    # Create a stable job folder for THIS upload
    if st.session_state["current_pdf_name"] != uploaded.name or not st.session_state["current_job_dir"]:
        job_id = datetime.now().strftime("job_%Y%m%d_%H%M%S")
        job_dir = RUNS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        st.session_state["current_job_dir"] = str(job_dir)
        st.session_state["current_pdf_name"] = uploaded.name
    else:
        job_dir = Path(st.session_state["current_job_dir"])

    pdf_path = job_dir / uploaded.name
    if not pdf_path.exists():
        pdf_path.write_bytes(uploaded.getbuffer())

    st.success(f"Saved PDF to: {pdf_path}")

    # ---------------------------
    # 1A) Detect Header/Footer & Page No rules
    # ---------------------------
    st.subheader("1A) Header/Footer detection (editable)")

    if st.button("🔎 Detect header/footer candidates"):
        rows, regex_suggestions = detect_header_footer_candidates(pdf_path)
        st.session_state["hf_rows"] = rows
        st.session_state["hf_regex_suggestions"] = regex_suggestions

    if "hf_rows" in st.session_state:
        rows = st.session_state["hf_rows"]
        regex_suggestions = st.session_state.get("hf_regex_suggestions", [])

        if not rows:
            st.info("No strong repeated header/footer candidates detected (based on repetition in top/bottom bands).")
        else:
            df = pd.DataFrame(rows)
            st.write("✅ Detected repeated lines (top/bottom). Uncheck anything you want to KEEP.")
            edited_df = st.data_editor(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "remove": st.column_config.CheckboxColumn("Remove?", default=True),
                    "band": st.column_config.TextColumn("Band"),
                    "text": st.column_config.TextColumn("Text"),
                    "pages_count": st.column_config.NumberColumn("Pages"),
                    "pages_ratio": st.column_config.NumberColumn("Ratio"),
                    "sample_pages": st.column_config.TextColumn("Sample pages"),
                },
            )

            st.markdown("### Page-number removal rules (regex)")
            st.caption("Enable/disable these patterns. These are applied globally while extracting text.")
            regex_enabled = []
            for i, rx in enumerate(regex_suggestions):
                checked = st.checkbox(f"{rx}", value=True, key=f"rx_{i}")
                if checked:
                    regex_enabled.append(rx)

            rules_path = job_dir / "remove_rules.json"

            if st.button("💾 Save removal rules"):
                exact_remove = edited_df[edited_df["remove"] == True]["text"].tolist()
                rules = {"exact": exact_remove, "regex": regex_enabled}
                rules_path.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
                st.success(f"Saved rules to: {rules_path}")
                st.session_state["remove_rules_path"] = str(rules_path)

    if st.button("📥 Start Processing"):
        st.info("Running Step 2 (blocks) + Step 2c (sections)...")

        out_blocks = job_dir / "out_blocks.json"
        sections_orig = job_dir / "sections_original"
        manifest_path = job_dir / "sections_manifest.json"
        sections_orig.mkdir(exist_ok=True)

        remove_rules_path = st.session_state.get("remove_rules_path", None)

        cmd1 = [
            sys.executable, str(STEP2),
            "--pdf", str(pdf_path),
            "--out", str(out_blocks),
        ]
        if remove_rules_path:
            cmd1 += ["--remove_rules", str(remove_rules_path)]

        ok1, log1 = run_cmd(cmd1, cwd=ROOT)

        st.subheader("Logs: Step2 blocks")
        st.code(log1 or "(no output)", language="text")

        if not ok1:
            st.error("Step2 failed. Fix logs above.")
            st.stop()

        cmd2 = [
            sys.executable, str(STEP2C),
            "--blocks", str(out_blocks),
            "--out_dir", str(sections_orig),
            "--manifest", str(manifest_path),
        ]
        ok2, log2 = run_cmd(cmd2, cwd=ROOT)

        st.subheader("Logs: Step2c sections")
        st.code(log2 or "(no output)", language="text")

        if not ok2:
            st.error("Step2c failed. Fix logs above.")
            st.stop()

        sections_norm = job_dir / "sections_normalized"
        sections_ssml = job_dir / "sections_ssml_preview"
        sections_norm.mkdir(exist_ok=True)
        sections_ssml.mkdir(exist_ok=True)

        st.info("Running Step 3 (normalize sections) + Step 4 (SSML preview per section)...")

        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

        for s in manifest_data:
            sec_file = Path(s["file"])
            if not sec_file.exists():
                sec_file = sections_orig / Path(s["file"]).name

            raw = sec_file.read_text(encoding="utf-8")

            norm_text = normalize_section_text(raw)
            (sections_norm / sec_file.name).write_text(norm_text, encoding="utf-8")

            (sections_ssml / f"{sec_file.stem}.xml").write_text(
                text_to_ssml_preview(norm_text),
                encoding="utf-8"
            )

        st.success("✅ Step2 + Step2c + Step3 + Step4 completed. Scroll down to edit sections.")
        st.session_state["active_job_dir"] = str(job_dir)

        # ---------------------------
        # Table Explanations UI (after processing)
        # ✅ FIX 3: ui_scope is DIFFERENT from job-level section so keys never collide
        # ---------------------------
        if out_blocks.exists():
            blocks_data = json.loads(out_blocks.read_text(encoding="utf-8"))
            render_table_explanations_ui(
                job_dir=job_dir,
                pdf_path=pdf_path,
                blocks_data=blocks_data,
                title="🧾 Table Explanations (New upload)",
                ui_scope="after_process"  # ✅ FIX 3
            )


# ---------------------------
# 2) Select Job
# ---------------------------
st.divider()
st.header("2) Open a processed job")

jobs = sorted([p for p in RUNS_DIR.glob("job_*") if p.is_dir()], reverse=True)
job_options = [str(p) for p in jobs]

default_job = st.session_state.get("active_job_dir", job_options[0] if job_options else None)
selected_job = None
if job_options:
    selected_job = st.selectbox(
        "Select Job Folder",
        job_options,
        index=job_options.index(default_job) if default_job in job_options else 0
    )

if not selected_job:
    st.info("No processed job found yet. Upload a PDF and click Start Processing.")
    st.stop()

job_dir = Path(selected_job)
manifest = job_dir / "sections_manifest.json"
orig_dir = job_dir / "sections_original"
user_dir = job_dir / "sections_user"
user_dir.mkdir(exist_ok=True)

if not manifest.exists():
    st.warning("This job is not processed yet. Run processing first.")
    st.stop()

# Resolve job PDF path (first pdf inside job folder)
pdf_candidates = list(job_dir.glob("*.pdf"))
job_pdf_path = pdf_candidates[0] if pdf_candidates else None

# ---------------------------
# Table explanations UI for existing jobs too
# ✅ FIX 3: ui_scope="job_level" so keys are unique from after_process UI
# ---------------------------
out_blocks_path = job_dir / "out_blocks.json"
if out_blocks_path.exists() and job_pdf_path and job_pdf_path.exists():
    blocks_data = json.loads(out_blocks_path.read_text(encoding="utf-8"))
    render_table_explanations_ui(
        job_dir=job_dir,
        pdf_path=job_pdf_path,
        blocks_data=blocks_data,
        title="🧾 Table Explanations (Job-level)",
        ui_scope="job_level"  # ✅ FIX 3
    )
elif out_blocks_path.exists():
    st.divider()
    st.header("🧾 Table Explanations (Job-level)")
    st.warning("out_blocks.json exists, but I couldn't find the PDF file in this job folder to render screenshots.")
    blocks_data = json.loads(out_blocks_path.read_text(encoding="utf-8"))
    render_table_explanations_ui(
        job_dir=job_dir,
        pdf_path=job_pdf_path if job_pdf_path else (job_dir / "missing.pdf"),
        blocks_data=blocks_data,
        title="🧾 Table Explanations (Job-level)",
        ui_scope="job_level"  # ✅ FIX 3
    )
else:
    st.divider()
    st.header("🧾 Table Explanations (Job-level)")
    st.info("out_blocks.json not found in this job folder (run processing first).")

sections = json.loads(manifest.read_text(encoding="utf-8"))
ensure_normalized_and_ssml(job_dir, sections)

# ---------------------------
# 3) Sections Editor
# ---------------------------
st.divider()
st.header("3) Edit Sections")

st.sidebar.header("Sections")
search = st.sidebar.text_input("Search section title", "")

filtered = [s for s in sections if search.lower() in s["title"].lower()]
selected = st.sidebar.selectbox(
    "Select section",
    filtered,
    format_func=lambda s: f"{s['order']:02d}. {s['title']} (p{s['page_start']}-{s['page_end']})"
)

sections_norm = job_dir / "sections_normalized"
sections_ssml = job_dir / "sections_ssml_preview"

sec_file = Path(selected["file"])
if not sec_file.exists():
    sec_file = orig_dir / Path(selected["file"]).name

user_file = user_dir / sec_file.name
norm_file = sections_norm / sec_file.name
ssml_file = sections_ssml / f"{sec_file.stem}.xml"

if user_file.exists():
    text = user_file.read_text("utf-8")
elif norm_file.exists():
    text = norm_file.read_text("utf-8")
else:
    text = sec_file.read_text("utf-8")

st.subheader(selected["title"])
st.caption(f"Pages: {selected['page_start']}–{selected['page_end']} | Words: {selected['word_count']}")

if ssml_file.exists():
    with st.expander("🔎 SSML preview (generated after Step4)", expanded=False):
        st.code(ssml_file.read_text("utf-8"), language="xml")

if user_file.exists():
    st.success("Showing: ✅ User edited version (sections_user)")
elif norm_file.exists():
    st.info("Showing: ✨ Normalized version (sections_normalized)")
else:
    st.warning("Showing: ⚠ Original extracted version (sections_original)")

edited = st.text_area("Edit text (this will be used for speech)", value=text, height=520)

c1, c2, c3 = st.columns(3)
if c1.button("💾 Save Changes"):
    user_file.write_text(edited, encoding="utf-8")
    st.success(f"Saved to: {user_file}")

if c2.button("♻ Reset to Default (remove user edits)"):
    if user_file.exists():
        user_file.unlink()
    st.info("Reset done. Reload section.")

if c3.button("🧾 Mark as SKIP (empty it)"):
    user_file.write_text("", encoding="utf-8")
    st.warning("This section will produce no speech.")

# ---------------------------
# 4) Select Sections for Speech (ORDER PRESERVED)
# ---------------------------
st.divider()
st.header("4) Select Sections for Speech")

st.write("Tick sections you want to include (in document order):")

selected_items = []
for s in sections:
    checked = st.checkbox(f"{s['order']:02d}. {s['title']}", value=True, key=f"sec_{s['order']}")
    if checked:
        f = orig_dir / Path(s["file"]).name
        uf = user_dir / f.name
        nf = (job_dir / "sections_normalized") / f.name

        if uf.exists():
            chosen = uf
        elif nf.exists():
            chosen = nf
        else:
            chosen = f

        selected_items.append({
            "order": s["order"],
            "title": s["title"],
            "file": str(chosen)
        })

st.write("Selected section files:", [x["file"] for x in selected_items])

# ---------------------------
# 5) Generate Speech (PER SECTION + FINAL)
# ---------------------------
st.divider()
st.header("5) Generate Speech")

engine = st.selectbox("TTS Engine", ["xtts", "vits"], index=0)
max_chars = st.number_input("Chunk max chars", min_value=500, max_value=4000, value=1800, step=100)

speaker_wav = st.text_input("XTTS speaker wav (optional)", value="voice_sample.wav")
xtts_lang = st.text_input("XTTS language", value="en")

speech_work_dir = job_dir / "speech_work"
section_audio_dir = job_dir / "section_audio"
final_wav = job_dir / "final.wav"

if st.button("🚀 Generate Speech Now"):
    if not selected_items:
        st.error("No sections selected.")
        st.stop()

    selected_items = sorted(selected_items, key=lambda x: x["order"])

    speech_work_dir.mkdir(exist_ok=True)
    section_audio_dir.mkdir(exist_ok=True)

    st.info("Generating per-section audio...")
    section_wavs = []

    for item in selected_items:
        order = int(item["order"])
        title = item["title"]
        src_file = Path(item["file"])

        if src_file.exists() and src_file.read_text(encoding="utf-8").strip() == "":
            st.warning(f"Skipping empty section: {order:02d}. {title}")
            continue

        section_name = f"{order:02d}_{safe_name(title)}"
        work_dir = speech_work_dir / section_name

        text_chunks_dir = work_dir / "text_chunks"
        wav_chunks_dir = work_dir / "wav_chunks"
        work_dir.mkdir(parents=True, exist_ok=True)
        text_chunks_dir.mkdir(exist_ok=True)
        wav_chunks_dir.mkdir(exist_ok=True)

        for p in text_chunks_dir.glob("chunk_*.txt"):
            p.unlink()
        for p in wav_chunks_dir.glob("chunk_*.wav"):
            p.unlink()

        st.write(f"▶️ Section {order:02d}: {title}")

        cmd5a = [
            sys.executable, str(STEP5A),
            "--inputs", json.dumps([str(src_file)]),
            "--out_dir", str(text_chunks_dir),
            "--max_chars", str(int(max_chars)),
        ]
        ok5a, log5a = run_cmd(cmd5a, cwd=ROOT)
        with st.expander(f"Logs: Step5a ({order:02d})", expanded=False):
            st.code(log5a or "(no output)", language="text")
        if not ok5a:
            st.error(f"Step5a failed for section {order:02d}.")
            st.stop()

        env = dict(os.environ)
        env["VOICE_ENGINE"] = engine
        env["XTTS_MODEL"] = env.get("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
        env["VITS_MODEL"] = env.get("VITS_MODEL", "tts_models/en/ljspeech/vits")
        env["SPEAKER_WAV"] = speaker_wav
        env["XTTS_LANG"] = xtts_lang

        cmd5b = [sys.executable, str(STEP5B)]
        ok5b, log5b = run_cmd(cmd5b, cwd=work_dir, env=env)
        with st.expander(f"Logs: Step5b ({order:02d})", expanded=False):
            st.code(log5b or "(no output)", language="text")
        if not ok5b:
            st.error(f"Step5b failed for section {order:02d}.")
            st.stop()

        section_out = section_audio_dir / f"{section_name}.wav"
        try:
            chunk_files = sorted(wav_chunks_dir.glob("chunk_*.wav"))
            if not chunk_files:
                st.warning(f"No wav chunks produced for section {order:02d}.")
                continue
            merge_wavs_in_order(chunk_files, section_out)
            section_wavs.append(section_out)
            st.success(f"✅ Section audio ready: {section_out.name}")
        except Exception as e:
            st.error(f"Failed to merge section {order:02d}: {e}")
            st.stop()

    if not section_wavs:
        st.error("No section audio produced. (All may be empty/failed).")
        st.stop()

    st.info("Merging final audio...")
    try:
        merge_wavs_in_order(section_wavs, final_wav)
        st.success("✅ Final speech generated!")
        st.session_state["final_wav"] = str(final_wav)
        st.session_state["section_wavs"] = [str(p) for p in section_wavs]
    except Exception as e:
        st.error(f"Final merge failed: {e}")
        st.stop()

# ---------------------------
# Output players + downloads
# ---------------------------
if "section_wavs" in st.session_state:
    st.subheader("🎧 Per-section audio")
    for p in st.session_state["section_wavs"]:
        fp = Path(p)
        if fp.exists():
            st.write(fp.name)
            st.audio(fp.read_bytes(), format="audio/wav")
            st.download_button(f"⬇ Download {fp.name}", fp.read_bytes(), file_name=fp.name, mime="audio/wav")

if "final_wav" in st.session_state:
    fw = Path(st.session_state["final_wav"])
    if fw.exists():
        st.subheader("🎬 Final combined audio")
        st.audio(fw.read_bytes(), format="audio/wav")
        st.download_button("⬇ Download FINAL WAV", fw.read_bytes(), file_name=fw.name, mime="audio/wav")
