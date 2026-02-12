import os
import streamlit as st

from core.utils import new_doc_dir
from core.dictionary import default_rules, apply_dictionary
from core.normalize import normalize_text
from core.ssml import to_ssml_preview
from components.pdf_selector import pdf_selector

st.set_page_config(page_title="PDF → Speech (XTTS)", layout="wide")
st.title("PDF → Speech (Coqui XTTS Voice Clone)")

# ---------------- Session init ----------------
if "doc" not in st.session_state:
    st.session_state.doc = None

if "rules" not in st.session_state:
    st.session_state.rules = default_rules()

if "normalize_opts" not in st.session_state:
    st.session_state.normalize_opts = {
        "fix_hyphen_linebreaks": True,
        "collapse_whitespace": True,
        "normalize_punct": True,
    }

# NEW: store highlighted selection (from PDF) + page
if "selected_text" not in st.session_state:
    st.session_state.selected_text = ""

if "selected_page" not in st.session_state:
    st.session_state.selected_page = None

if "processed_text" not in st.session_state:
    st.session_state.processed_text = ""

if "edited_text" not in st.session_state:
    st.session_state.edited_text = ""

if "ssml" not in st.session_state:
    st.session_state.ssml = ""

if "speaker_wav_path" not in st.session_state:
    st.session_state.speaker_wav_path = None

if "audio_path" not in st.session_state:
    st.session_state.audio_path = None

# ---------------- Sidebar: steps ----------------
with st.sidebar:
    st.header("Flow")
    step = st.radio(
        "Step",
        [
            "1) Upload PDF",
            "2) Open PDF + Highlight text",
            "3) Dictionary + Normalize + SSML preview",
            "4) Edit text",
            "5) Generate audio (XTTS clone)",
        ],
        index=0,
    )

# ---------------- STEP 1 ----------------
if step.startswith("1"):
    st.subheader("1) Upload PDF")
    pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    if pdf:
        doc_id, doc_dir = new_doc_dir("data")
        pdf_path = os.path.join(doc_dir, "input.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf.getbuffer())

        st.session_state.doc = {
            "doc_id": doc_id,
            "doc_dir": doc_dir,
            "pdf_path": pdf_path,
        }

        # reset pipeline state on new upload
        st.session_state.selected_text = ""
        st.session_state.selected_page = None
        st.session_state.processed_text = ""
        st.session_state.edited_text = ""
        st.session_state.ssml = ""
        st.session_state.audio_path = None

        st.success("PDF uploaded ✅")
        st.info("Go to step 2: Open PDF + Highlight text")

    st.caption("Step 2 will open the PDF in-app and allow highlight selection (no pre-extraction list).")

# ---------------- STEP 2 ----------------
elif step.startswith("2"):
    st.subheader("2) Open PDF + Highlight text")
    doc = st.session_state.doc
    if not doc:
        st.warning("Upload a PDF first.")
        st.stop()

    # Read PDF bytes and render selector component
    with open(doc["pdf_path"], "rb") as f:
        pdf_bytes = f.read()

    st.caption("Highlight (select) a paragraph inside the PDF. Your selection will appear below.")
    result = pdf_selector(pdf_bytes=pdf_bytes, height=750, key="pdfsel")

    selected = (result or {}).get("selectedText", "").strip()
    page_num = (result or {}).get("pageNumber", None)

    if selected:
        st.session_state.selected_text = selected
        st.session_state.selected_page = page_num

    if st.session_state.selected_text:
        st.success(f"Selected text captured ✅ (Page: {st.session_state.selected_page})")
        st.session_state.selected_text = st.text_area(
            "Selected text (you can tweak here before processing)",
            value=st.session_state.selected_text,
            height=180,
        )
        st.info("Go to step 3 to apply Dictionary + Normalize and preview SSML.")
    else:
        st.info("No text selected yet. Highlight some text in the PDF to continue.")

# ---------------- STEP 3 ----------------
elif step.startswith("3"):
    st.subheader("3) Dictionary + Normalize + SSML preview")

    doc = st.session_state.doc
    if not doc:
        st.warning("Upload a PDF first.")
        st.stop()

    raw_text = (st.session_state.selected_text or "").strip()
    if not raw_text:
        st.warning("Select/highlight some text in step 2 first.")
        st.stop()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### Dictionary rules")
        rules = st.session_state.rules
        new_rules = []

        for i, r in enumerate(rules):
            with st.container(border=True):
                c1, c2 = st.columns([1, 1])
                find = c1.text_input("Find", r.get("find", ""), key=f"find_{i}")
                repl = c2.text_input("Replace", r.get("replace", ""), key=f"repl_{i}")

                c3, c4, c5 = st.columns([1, 1, 1])
                mt = c3.selectbox(
                    "Match",
                    ["exact", "word", "regex"],
                    index=["exact", "word", "regex"].index(r.get("match_type", "exact")),
                    key=f"mt_{i}",
                )
                cs = c4.checkbox("Case sensitive", value=bool(r.get("case_sensitive", False)), key=f"cs_{i}")
                en = c5.checkbox("Enabled", value=bool(r.get("enabled", True)), key=f"en_{i}")

                new_rules.append(
                    {"find": find, "replace": repl, "match_type": mt, "case_sensitive": cs, "enabled": en}
                )

        if st.button("+ Add rule"):
            new_rules.append({"find": "", "replace": "", "match_type": "exact", "case_sensitive": False, "enabled": True})

        st.session_state.rules = new_rules

        st.markdown("#### Normalize options")
        opts = st.session_state.normalize_opts
        opts["fix_hyphen_linebreaks"] = st.checkbox("Fix hyphen line breaks", value=opts["fix_hyphen_linebreaks"])
        opts["collapse_whitespace"] = st.checkbox("Collapse whitespace", value=opts["collapse_whitespace"])
        opts["normalize_punct"] = st.checkbox("Normalize punctuation spacing", value=opts["normalize_punct"])
        st.session_state.normalize_opts = opts

        pause_ms = st.slider("SSML paragraph pause (ms)", 0, 1500, 250, 50)

    processed = apply_dictionary(raw_text, st.session_state.rules)
    processed = normalize_text(processed, st.session_state.normalize_opts)
    ssml = to_ssml_preview(processed, break_ms=pause_ms)

    st.session_state.processed_text = processed
    st.session_state.ssml = ssml

    # seed edited text if empty OR if it still matches the old processed text scenario
    if not st.session_state.edited_text:
        st.session_state.edited_text = processed

    with col2:
        st.markdown("#### Preview")
        with st.expander("Raw selected text"):
            st.text(raw_text)
        with st.expander("Processed text", expanded=True):
            st.text(processed)
        with st.expander("SSML preview (preview only)"):
            st.code(ssml, language="xml")

    st.info("Go to step 4 to edit the processed text before generating audio.")

# ---------------- STEP 4 ----------------
elif step.startswith("4"):
    st.subheader("4) Edit text")

    doc = st.session_state.doc
    if not doc:
        st.warning("Upload a PDF first.")
        st.stop()

    if not st.session_state.processed_text:
        st.warning("Run step 3 first to generate processed text.")
        st.stop()

    st.session_state.edited_text = st.text_area(
        "Edit the final text (this will be spoken by XTTS)",
        value=st.session_state.edited_text,
        height=320,
    )

    c1, c2 = st.columns([1, 1])
    if c1.button("Reset to processed"):
        st.session_state.edited_text = st.session_state.processed_text
        st.rerun()

    if c2.button("Clear"):
        st.session_state.edited_text = ""
        st.rerun()

    st.info("Go to step 5 for voice clone + audio generation using Coqui XTTS.")

# ---------------- STEP 5 ----------------
else:
    st.subheader("5) Generate audio (Coqui XTTS voice clone)")

    doc = st.session_state.doc
    if not doc:
        st.warning("Upload a PDF first.")
        st.stop()

    if not st.session_state.edited_text.strip():
        st.warning("No text to speak. Complete steps 2–4 first.")
        st.stop()

    st.markdown("### Upload voice sample (speaker WAV)")
    voice_file = st.file_uploader("Upload speaker voice (WAV recommended)", type=["wav", "mp3", "m4a"])

    lang = st.selectbox("Language", ["en", "hi"], index=0)
    use_gpu = st.checkbox("Use GPU (CUDA)", value=False)

    if voice_file:
        speaker_path = os.path.join(doc["doc_dir"], f"speaker_{voice_file.name}")
        with open(speaker_path, "wb") as f:
            f.write(voice_file.getbuffer())
        st.session_state.speaker_wav_path = speaker_path
        st.success("Voice sample saved ✅")

    if st.button("Generate audio"):
        if not st.session_state.speaker_wav_path:
            st.error("Please upload a voice sample first.")
            st.stop()

        # Lazy import so Streamlit doesn't hang on startup due to heavy TTS imports
        from core.tts_xtts import synthesize_xtts

        with st.spinner("Generating audio with XTTS-v2 (voice clone)..."):
            audio_path = synthesize_xtts(
                text=st.session_state.edited_text.strip(),
                speaker_wav_path=st.session_state.speaker_wav_path,
                out_dir=doc["doc_dir"],
                language=lang,
                use_gpu=use_gpu,
            )
            st.session_state.audio_path = audio_path

        st.success("Audio generated ✅")

    if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
        st.audio(st.session_state.audio_path)
        with open(st.session_state.audio_path, "rb") as f:
            st.download_button("Download WAV", f, file_name=os.path.basename(st.session_state.audio_path))
