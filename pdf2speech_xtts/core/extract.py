import fitz
import re
import uuid
from typing import List, Dict, Any

def _is_heading(text: str) -> bool:
    t = text.strip()
    if len(t) < 3:
        return False
    if len(t) <= 70 and not t.endswith("."):
        if t.isupper():
            return True
        words = [w for w in re.split(r"\s+", t) if w]
        if 1 <= len(words) <= 10:
            upperish = sum(1 for w in words if w[:1].isupper())
            return (upperish / max(1, len(words))) >= 0.6
    return False

def extract_blocks(pdf_path: str) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    blocks: List[Dict[str, Any]] = []

    for p in range(len(doc)):
        page = doc[p]
        raw = page.get_text("blocks")
        for b in raw:
            x0, y0, x1, y1, text, block_no, _ = b
            text = (text or "").strip()
            if not text:
                continue
            bid = f"p{p+1}_b{block_no}_{uuid.uuid4().hex[:6]}"
            kind = "heading" if _is_heading(text) else "paragraph"
            blocks.append({
                "block_id": bid,
                "page": p + 1,
                "bbox": [x0, y0, x1, y1],
                "type": kind,
                "text": text
            })
    return blocks

def build_topics(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    topics = []
    current = None

    for blk in blocks:
        if blk["type"] == "heading":
            current = {
                "topic_id": blk["block_id"],
                "title": blk["text"],
                "page": blk["page"],
                "block_ids": []
            }
            topics.append(current)
        else:
            if current is None:
                if not topics or topics[0].get("topic_id") != "topic_default":
                    topics.insert(0, {
                        "topic_id": "topic_default",
                        "title": "Introduction / Unsorted",
                        "page": 1,
                        "block_ids": []
                    })
                topics[0]["block_ids"].append(blk["block_id"])
            else:
                current["block_ids"].append(blk["block_id"])

    return topics

def get_block(blocks: List[Dict[str, Any]], block_id: str) -> Dict[str, Any]:
    for b in blocks:
        if b["block_id"] == block_id:
            return b
    raise KeyError(f"Block not found: {block_id}")
