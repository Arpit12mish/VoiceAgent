import os
import uuid

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def new_doc_dir(base="data"):
    doc_id = uuid.uuid4().hex[:10]
    doc_dir = os.path.join(base, doc_id)
    ensure_dir(doc_dir)
    return doc_id, doc_dir
