import os
import base64
import streamlit.components.v1 as components

_component_func = components.declare_component(
    "pdf_selector",
    path=os.path.join(os.path.dirname(__file__), "frontend"),
)

def pdf_selector(pdf_bytes: bytes, height: int = 650, key=None):
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    return _component_func(
        pdf_b64=pdf_b64,
        height=height,
        key=key,
        default={"selectedText": "", "pageNumber": None},
    )
