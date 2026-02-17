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

WATERMARK_PATTERNS = [
    r"supplied by .*? under the license",
    r"under the license from bis",
    r"csir-?national physical laboratory",
    r"valid upto",
    r"\(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\)",  # IP address
]


def looks_like_watermark(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(re.search(p, t) for p in WATERMARK_PATTERNS)


def is_margin_noise(line: dict) -> bool:
    """
    Removes header/footer + vertical stamp style lines by position & size.
    """
    y = float(line.get("y", 0))
    x = float(line.get("x", 0))
    size = float(line.get("avg_size", 0))
    # Typical A4: y~0 top, y~840 bottom
    if size <= 9.0 and (y <= 35 or y >= 805):
        return True
    # Vertical stamp often near right edge with small size
    if size <= 9.0 and x >= 540:
        return True
    return False


# -------------------------------
# ✅ TABLE TOKEN PRESERVATION
# -------------------------------
# We must NOT modify anything inside:
#   [[TABLE|p8_t0|Table 1 – ...]]
TABLE_TOKEN_RE = re.compile(r"\[\[TABLE\|[^\]]+\]\]")


def _protect_table_tokens(text: str):
    """
    Replace table tokens with placeholders so normalization doesn't break them.
    Returns (protected_text, mapping).
    """
    mapping = {}

    def repl(m):
        key = f"__TABLETOKEN_{len(mapping)}__"
        mapping[key] = m.group(0)
        return key

    protected = TABLE_TOKEN_RE.sub(repl, text or "")
    return protected, mapping


def _restore_table_tokens(text: str, mapping: dict):
    out = text or ""
    for k, v in mapping.items():
        out = out.replace(k, v)
    return out


def norm_unicode(s: str) -> str:
    # Normalize unicode + replace common typography chars
    s = unicodedata.normalize("NFKC", s or "")
    for k, v in LIGATURES.items():
        s = s.replace(k, v)
    s = s.replace("\u00ad", "")  # soft hyphen
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = s.replace("–", "-").replace("—", "-")
    return s


def normalize_ws(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "")
    return s.strip()


def fix_decimal_spacing(s: str) -> str:
    # 3 . 14 -> 3.14
    return re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", s or "")


def remove_dot_leaders(s: str) -> str:
    # TOC style: "Foreword........iv" -> "Foreword iv"
    s = re.sub(r"\.{5,}", " ", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def looks_like_page_marker(s: str) -> bool:
    # "ii", "iv", "1", "Page", etc.
    t = (s or "").strip().lower()
    if t in {"page"}:
        return True
    if re.fullmatch(r"[ivxlcdm]+", t) and len(t) <= 6:  # roman numerals
        return True
    if re.fullmatch(r"\d{1,3}", t):  # plain page number
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


# -------------------------------
# ✅ Plural-aware unit normalization (SAFE)
# -------------------------------

def normalize_decimal_commas(text: str) -> str:
    """
    Convert decimal comma to decimal dot: 0,01 -> 0.01
    Only when comma is between digits.
    """
    return re.sub(r"(?<=\d),(?=\d)", ".", text or "")


def _is_plural_number(num_str: str) -> bool:
    """
    Decide pluralization:
    - 1 or 1.0 -> singular
    - everything else -> plural
    """
    try:
        v = float(num_str)
        return abs(v - 1.0) > 1e-12
    except Exception:
        # if parse fails, default to plural (safer for TTS)
        return True


def _maybe_pluralize(unit_word: str, plural: bool) -> str:
    """
    Pluralize simple English units:
    - centimetre -> centimetres
    - cubic centimetre -> cubic centimetres
    - square metre -> square metres
    """
    if not plural:
        return unit_word

    # pluralize the last token only ("cubic centimetre" -> "cubic centimetres")
    parts = unit_word.split()
    if not parts:
        return unit_word
    last = parts[-1]

    # basic pluralization rules; extend later if needed
    if last.endswith("s"):
        # already plural-ish
        parts[-1] = last
    elif last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
        parts[-1] = last[:-1] + "ies"
    else:
        parts[-1] = last + "s"

    return " ".join(parts)


def normalize_symbols(text: str) -> str:
    """
    Context-aware + plural-aware unit normalization:
    - Converts cm3 / cm³ / cm^3 / cm 3 / ㎤ variants -> "cubic centimetre(s)" in numeric context
    - Converts cm/mm/m/km ONLY when they follow a number (prevents breaking words like "Sampling")
    - Converts decimal commas (0,01 -> 0.01)
    """
    if not text:
        return text

    # 0) decimal commas first (common in ISO PDFs)
    text = normalize_decimal_commas(text)

    # 1) normalize special unit glyphs + superscripts
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

    # 2) safe direct replacements (NOT single-letter units)
    safe_direct = {
        "@": " at the rate ",
        "&": " and ",
        "%": " percent ",
        "₹": " rupees ",
        "$": " dollars ",
        "°C": " degree celsius ",
        "°c": " degree celsius ",
        "°F": " degree fahrenheit ",
        "°f": " degree fahrenheit ",
        "μm": " micrometer ",
        "ppm": " parts per million ",
        "ppb": " parts per billion ",
        "mg/l": "milligrams per liter",
        "kg/l": "kilograms per liter",
        "°": " degree ", 
        "rad": " radian ",
        "g/mol": "grams/mole",
    }
    for k, v in safe_direct.items():
        text = text.replace(k, v)

    # maps for powered & plain units
    unit_volume = {
        "mm": "cubic millimetre",
        "cm": "cubic centimetre",
        "m":  "cubic metre",
        "km": "cubic kilometre",
    }
    unit_area = {
        "mm": "square millimetre",
        "cm": "square centimetre",
        "m":  "square metre",
        "km": "square kilometre",
    }
    unit_length = {
        "mm": "millimetre",
        "cm": "centimetre",
        "m":  "metre",
        "km": "kilometre",
    }
    unit_mass = {
        "mg": "milligrams",
        "g":  "grams",
        "kg": "kilograms",
    }
    unit_volume_liquid = {
        "ml": "millilitre",
        "l":  "litre",
    }
    unit_time = {
        "sec": "second",
        "s":   "second",
        "min": "minute",
        "hr":  "hour",
        "h":   "hour",
        "yr":  "year",
        "y":   "year",
    }

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

    # 3) Volume: number + unit + 3 (cm3, cm^3, cm 3, cm³ already normalized to cm3)
    text = re.sub(
        r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m|km)\s*(?:\^?\s*3)\b",
        lambda m: repl_powered(m, unit_volume),
        text,
        flags=re.IGNORECASE
    )

    # 4) Area: number + unit + 2
    text = re.sub(
        r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m|km)\s*(?:\^?\s*2)\b",
        lambda m: repl_powered(m, unit_area),
        text,
        flags=re.IGNORECASE
    )

    # 5) Length: number + unit (no power)
    text = re.sub(
        r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m|km)\b",
        lambda m: repl_plain(m, unit_length),
        text,
        flags=re.IGNORECASE
    )

    # 6) Mass: number + unit
    text = re.sub(
        r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>mg|g|kg)\b",
        lambda m: repl_plain(m, unit_mass),
        text,
        flags=re.IGNORECASE
    )

    # 7) Liquid volume: number + unit
    text = re.sub(
        r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>ml|l)\b",
        lambda m: repl_plain(m, unit_volume_liquid),
        text,
        flags=re.IGNORECASE
    )

    # 8) Time: number + unit
    text = re.sub(
        r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>sec|min|hr|yr|s|h|y)\b",
        lambda m: repl_plain(m, unit_time),
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_slash_context(text: str) -> str:
    """
    Context-aware normalization of '/'.
    """
    # URLs
    text = re.sub(
        r'((https?://|www\.)[^\s]+)',
        lambda m: m.group(0).replace('/', ' slash '),
        text or ""
    )

    # file paths
    text = re.sub(
        r'([A-Za-z]:\\[^\s]+|\/[A-Za-z0-9_\-\/\.]+)',
        lambda m: m.group(0).replace('/', ' slash '),
        text
    )

    # Units like kg/m, km/h, m/s -> per
    text = re.sub(
        r'(?<=\b[A-Za-z]+)\s*/\s*(?=[A-Za-z]+\b)',
        ' per ',
        text
    )

    # Fractions 3/4 -> upon
    text = re.sub(
        r'(?<=\d)\s*/\s*(?=\d)',
        ' upon ',
        text
    )
    return text


def normalize_numeric_ranges(text: str) -> str:
    """
    Replace dash with 'to' when used as numeric range.
    """
    return re.sub(r'(?<=\d)\s*[-–—]\s*(?=\d)', ' to ', text or "")


def normalize_dots(text: str) -> str:
    """
    Context-aware normalization of dot '.'
    """
    text = text or ""

    # IP addresses -> point
    text = re.sub(
        r'\b(\d{1,3}(?:\.\d{1,3}){3})\b',
        lambda m: m.group(0).replace('.', ' point '),
        text
    )

    # Version numbers -> point
    text = re.sub(
        r'\b\d+(?:\.\d+){1,}\b',
        lambda m: m.group(0).replace('.', ' point '),
        text
    )

    # Decimal numbers -> point
    text = re.sub(
        r'(?<=\d)\.(?=\d)',
        ' point ',
        text
    )

    # Multiple dots -> dot dot ...
    text = re.sub(
        r'\.{2,}',
        lambda m: ' dot ' * len(m.group(0)),
        text
    )

    text = re.sub(r'\s+', ' ', text)
    return text.strip()


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
                        xs.append(bbox[0])
                        ys.append(bbox[1])

                if not texts:
                    continue

                text = normalize_ws("".join(texts))
                if not text:
                    continue

                avg_size = sum(sizes) / len(sizes) if sizes else 0.0
                bold_ratio = sum(1 for b in bolds if b) / len(bolds) if bolds else 0.0
                italic_ratio = sum(1 for i in italics if i) / len(italics) if italics else 0.0

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
    body_size = (sum(sizes) / len(sizes)) if sizes else 11.0

    blocks = []
    cur = None
    prev = None

    def flush():
        nonlocal cur
        if cur:
            cur["text"] = cur["text"].strip()
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
                    if y_gap > 10:
                        cur["text"] += "\n" + ln["text"]
                    else:
                        cur["text"] += " " + ln["text"]
                else:
                    flush()
                    cur = {"type": btype, "page": ln["page"], "text": ln["text"]}

        prev = ln

    flush()
    return blocks


# -------------------------------
# Reflow helper (GENERAL, no hardcoded words)
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
    return (s or "").rstrip().endswith((".", "?", "!", ";"))


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


def normalize_blocks(blocks):
    """
    ✅ Preserves table tokens [[TABLE|...]] exactly.
    ✅ Plural-aware unit normalization (10 centimetres, 1 centimetre)
    ✅ Handles cm³ / cm3 / ㎤ etc + decimal commas (0,01)
    """
    cleaned = []
    for b in blocks:
        t = b["text"] or ""

        # protect table tokens
        t, mapping = _protect_table_tokens(t)

        # normalize (safe)
        t = norm_unicode(t)
        t = remove_dot_leaders(t)
        t = fix_decimal_spacing(t)
        t = normalize_numeric_ranges(t)
        t = normalize_slash_context(t)
        t = normalize_dots(t)
        t = normalize_symbols(t)

        # Keep newlines; normalize each line separately
        t = "\n".join(normalize_ws(x) for x in t.splitlines())
        t = re.sub(r"\n{3,}", "\n\n", t).strip()

        # Apply reflow fix for list markers + wrapping
        t = reflow_pdf_text(t)

        # restore table tokens
        t = _restore_table_tokens(t, mapping)

        # Drop obvious noise
        if not t:
            continue

        # If block is just a page marker (single-line)
        if "\n" not in t and looks_like_page_marker(t):
            continue

        # boilerplate check line-wise (but don't kill table tokens)
        if any(is_boilerplate_line(x) for x in t.splitlines() if x.strip()):
            non_boiler = [x for x in t.splitlines() if x.strip() and not is_boilerplate_line(x)]
            t = "\n".join(non_boiler).strip()
            if not t:
                continue

        if len(t.replace("\n", "").strip()) <= 2:
            continue

        if b["type"] == "heading" and t.strip().lower() == "contents":
            continue

        cleaned.append({**b, "text": t})
    return cleaned


def main():
    doc = fitz.open(PDF_PATH)
    lines = extract_lines(doc, max_pages=8)
    lines = [ln for ln in lines if not is_margin_noise(ln) and not looks_like_watermark(ln["text"])]

    repeated = detect_repeated_lines_anywhere(lines, min_pages_ratio=0.5)
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
    lines = [ln for ln in lines if not is_margin_noise(ln) and not looks_like_watermark(ln["text"])]

    repeated = detect_repeated_lines_anywhere(lines, min_pages_ratio=0.5)
    lines = [ln for ln in lines if ln["text"] not in repeated and not is_boilerplate_line(ln["text"])]

    blocks = merge_into_blocks(lines)
    blocks2 = normalize_blocks(blocks)
    return blocks2


if __name__ == "__main__":
    main()
