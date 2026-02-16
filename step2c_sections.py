import json
import re
import argparse
from pathlib import Path

SEC_NUM_HEADING = re.compile(r"^\s*(\d+(\.\d+)*)\s+(.+)$", re.IGNORECASE)
ANNEX_HEADING = re.compile(r"^\s*annex\s+([A-Z])\b", re.IGNORECASE)
BIBLIO = re.compile(r"^\s*bibliography\s*$", re.IGNORECASE)

# Table placeholder from step2_blocks.py:
#   "[TABLE:p8_t0] Table 1 – ...."
TABLE_PLACEHOLDER_RE = re.compile(r"^\s*\[TABLE:([^\]]+)\]\s*(.*)\s*$", re.IGNORECASE)

def slugify(s: str, max_len=60) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s_]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s[:max_len] if s else "section"

def is_section_heading(text: str) -> bool:
    t = (text or "").strip()
    return bool(SEC_NUM_HEADING.match(t) or ANNEX_HEADING.match(t) or BIBLIO.match(t))

def build_sections(blocks):
    """
    Builds sections as before, BUT:
    - preserves table blocks (type="table")
    - inserts a machine-readable token in section text:
        [[TABLE|table_id|caption]]
      so later steps/UI can replace it with user-provided explanation.
    - also collects per-section table list in metadata.
    """
    sections = []
    current = None

    def start_section(sec_id, title, page_start):
        return {
            "id": sec_id,
            "title": title,
            "page_start": page_start,
            "page_end": page_start,
            "text": "",
            "tables": []  # each: {table_id, caption, page, bbox}
        }

    def flush():
        nonlocal current
        if current and current["text"].strip():
            current["text"] = current["text"].strip()
            sections.append(current)
        current = None

    front = start_section("00_front_matter", "Front matter", 1)

    for b in blocks:
        btype = (b.get("type") or "").strip().lower()
        t = (b.get("text") or "").strip()
        if not t:
            continue
        p = int(b.get("page", 1))

        # --------------------------
        # 1) TABLE BLOCKS
        # --------------------------
        if btype == "table":
            table_id = (b.get("table_id") or "").strip()
            caption = (b.get("caption") or "").strip()

            # fallback: parse from "[TABLE:...]" text if needed
            if (not table_id) or (not caption):
                m = TABLE_PLACEHOLDER_RE.match(t)
                if m:
                    if not table_id:
                        table_id = m.group(1).strip()
                    if not caption:
                        caption = (m.group(2) or "").strip()

            if not caption:
                caption = f"Table on page {p}"
            if not table_id:
                # still keep something stable-ish (but ideally step2 already sets table_id)
                table_id = f"p{p}_t_unknown"

            token = f"[[TABLE|{table_id}|{caption}]]"

            table_meta = {
                "table_id": table_id,
                "caption": caption,
                "page": p,
                "bbox": b.get("bbox"),
            }

            if current:
                current["text"] += token + "\n\n"
                current["page_end"] = max(current["page_end"], p)
                current["tables"].append(table_meta)
            else:
                front["text"] += token + "\n\n"
                front["page_end"] = max(front["page_end"], p)
                front["tables"].append(table_meta)

            continue

        # --------------------------
        # 2) SECTION HEADINGS
        # --------------------------
        if btype == "heading" and is_section_heading(t):
            if current:
                flush()
            else:
                if front["text"].strip():
                    sections.append(front)

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

        # --------------------------
        # 3) NORMAL TEXT BLOCKS
        # --------------------------
        if current:
            current["text"] += t + "\n\n"
            current["page_end"] = max(current["page_end"], p)
        else:
            front["text"] += t + "\n\n"
            front["page_end"] = max(front["page_end"], p)

    if current:
        flush()
    else:
        if front["text"].strip():
            sections.append(front)

    return sections

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", required=True, help="Path to out_blocks.json")
    ap.add_argument("--out_dir", required=True, help="Directory to write sections_original/")
    ap.add_argument("--manifest", required=True, help="Path to sections_manifest.json")
    args = ap.parse_args()

    blocks_path = Path(args.blocks)
    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if not blocks_path.exists():
        raise FileNotFoundError(f"Blocks json not found: {blocks_path}")

    blocks = json.loads(blocks_path.read_text(encoding="utf-8"))
    sections = build_sections(blocks)

    manifest = []
    for idx, s in enumerate(sections):
        order = str(idx).zfill(2)
        fname = f"{order}_{s['id']}.txt"
        out_path = out_dir / fname
        out_path.write_text(s["text"], encoding="utf-8")

        manifest.append({
            "order": idx,
            "id": s["id"],
            "title": s["title"],
            "file": str(out_path).replace("\\", "/"),
            "page_start": s["page_start"],
            "page_end": s["page_end"],
            "word_count": len(s["text"].split()),
            "tables": s.get("tables", []),          # ✅ NEW
            "table_count": len(s.get("tables", [])) # ✅ NEW
        })

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✅ Sections: {len(manifest)}")
    print(f"✅ Saved sections -> {out_dir}")
    print(f"✅ Saved manifest -> {manifest_path}")

    # quick debug
    total_tables = sum(m.get("table_count", 0) for m in manifest)
    print(f"✅ Total table placeholders: {total_tables}")

if __name__ == "__main__":
    main()
