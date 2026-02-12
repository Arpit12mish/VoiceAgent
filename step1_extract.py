import fitz  # PyMuPDF
from pathlib import Path
import json

PDF_PATH = Path("IS60867.pdf")
OUT_JSON = Path("out_spans.json")  # saved output (for debugging & reuse)

def style_flags(font: str):
    f = (font or "").lower()
    is_bold = "bold" in f
    is_italic = ("italic" in f) or ("oblique" in f)
    return is_bold, is_italic

def extract_spans(pdf_path: Path, max_pages: int = 2):
    doc = fitz.open(pdf_path)
    pages = min(len(doc), max_pages)

    all_spans = []
    for pno in range(pages):
        page = doc[pno]
        w = float(page.rect.width)
        h = float(page.rect.height)

        data = page.get_text("dict")

        for block in data.get("blocks", []):
            # images have "type": 1, text blocks are usually type 0
            if block.get("type", 0) != 0:
                continue

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text or not text.strip():
                        continue

                    text = text.strip()
                    font = span.get("font", "")
                    size = float(span.get("size", 0.0))
                    bbox = span.get("bbox", None)  # [x0,y0,x1,y1]

                    is_bold, is_italic = style_flags(font)

                    all_spans.append({
                        "page": pno + 1,
                        "text": text,
                        "font": font,
                        "size": size,
                        "bold": is_bold,
                        "italic": is_italic,
                        "bbox": bbox,
                        "page_w": w,
                        "page_h": h,
                    })

    return all_spans, len(doc)

def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Put a PDF named {PDF_PATH} next to this script.")

    spans, total_pages = extract_spans(PDF_PATH, max_pages=2)

    total_chars = sum(len(s["text"]) for s in spans)
    print(f"Total pages: {total_pages}")
    print(f"Extracted spans (first 2 pages): {len(spans)}")
    print(f"Extracted chars (first 2 pages): {total_chars}")

    if total_chars < 200:
        print("\n❌ Looks like scanned / image-only PDF.")
        return

    # Save spans to json so Step 2 can optionally reuse for debugging
    OUT_JSON.write_text(json.dumps(spans, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ Saved spans JSON -> {OUT_JSON}")

    print("\n✅ Digital PDF confirmed. Sample spans:")
    for s in spans[:25]:
        style = []
        if s["bold"]:
            style.append("BOLD")
        if s["italic"]:
            style.append("ITALIC")
        style = ",".join(style) if style else "normal"
        print(f'[p{s["page"]}] ({style}, {s["size"]:.1f}) {s["text"]}')

if __name__ == "__main__":
    main()
