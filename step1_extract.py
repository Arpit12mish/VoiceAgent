import fitz  # PyMuPDF
from pathlib import Path

PDF_PATH = Path("ISO9735.pdf")  # <-- put your pdf here

def extract_spans(pdf_path: Path, max_pages: int = 2):
    doc = fitz.open(pdf_path)
    pages = min(len(doc), max_pages)
    all_spans = []

    for pno in range(pages):
        page = doc[pno]
        data = page.get_text("dict")

        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue

                    font = span.get("font", "")
                    size = span.get("size", 0)
                    # style inference (good enough for v1)
                    is_bold = "bold" in font.lower()
                    is_italic = ("italic" in font.lower()) or ("oblique" in font.lower())

                    all_spans.append({
                        "page": pno + 1,
                        "text": text,
                        "font": font,
                        "size": size,
                        "bold": is_bold,
                        "italic": is_italic
                    })

    return all_spans, len(doc)

def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Put a PDF named {PDF_PATH} next to this script.")

    spans, total_pages = extract_spans(PDF_PATH, max_pages=2)

    # Digital-PDF check: do we have enough text?
    total_chars = sum(len(s["text"]) for s in spans)
    print(f"Total pages: {total_pages}")
    print(f"Extracted spans (first 2 pages): {len(spans)}")
    print(f"Extracted chars (first 2 pages): {total_chars}")

    if total_chars < 200:  # heuristic
        print("\n❌ This looks like an image-only/scanned PDF (or very low text).")
        print("We said digital PDFs only, so this PDF is not suitable.")
        return

    print("\n✅ Digital PDF confirmed. Sample spans:")
    for s in spans[:25]:
        style = []
        if s["bold"]: style.append("BOLD")
        if s["italic"]: style.append("ITALIC")
        style = ",".join(style) if style else "normal"
        print(f'[p{s["page"]}] ({style}, {s["size"]:.1f}) {s["text"]}')

if __name__ == "__main__":
    main()
