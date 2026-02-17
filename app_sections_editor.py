import streamlit as st
import json
from pathlib import Path

MANIFEST = Path("sections_manifest.json")
ORIG_DIR = Path("sections_original")
USER_DIR = Path("sections_user")
USER_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="VoiceAgent - Sections Editor", layout="wide")
st.title("📘 PDF Sections (Edit → Speech)")

if not MANIFEST.exists():
    st.error("sections_manifest.json not found. Run: python step2c_sections.py")
    st.stop()

sections = json.loads(MANIFEST.read_text(encoding="utf-8"))

# Sidebar list
st.sidebar.header("Sections")
search = st.sidebar.text_input("Search section title")
filtered = [s for s in sections if search.lower() in s["title"].lower()]

selected = st.sidebar.selectbox(
    "Select section",
    filtered,
    format_func=lambda s: f"{s['order']:02d}. {s['title']} (p{s['page_start']}-{s['page_end']})"
)

sec_file = Path(selected["file"])
sec_id = sec_file.stem  # like 01_1_scope
user_file = USER_DIR / f"{sec_file.name}"  # same name inside user folder

text = user_file.read_text("utf-8") if user_file.exists() else sec_file.read_text("utf-8")

# Main editor
st.subheader(selected["title"])
st.caption(f"Pages: {selected['page_start']}–{selected['page_end']} | Words: {selected['word_count']}")

edited = st.text_area("Edit text (this will be used for speech)", value=text, height=500)

c1, c2, c3 = st.columns(3)

if c1.button("💾 Save Changes"):
    user_file.write_text(edited, encoding="utf-8")
    st.success("Saved to sections_user/")

if c2.button("♻ Reset to Original"):
    if user_file.exists():
        user_file.unlink()
    st.info("Reset done. Reload section.")

if c3.button("🧾 Mark as SKIP (empty it)"):
    user_file.write_text("", encoding="utf-8")
    st.warning("This section will produce no speech.")

st.divider()

# Generate speech trigger (we connect to pipeline in next step)
st.subheader("🎧 Generate speech")
st.write("Select which sections to include:")

selected_ids = []
for s in sections:
    default_on = True
    checked = st.checkbox(f"{s['order']:02d}. {s['title']}", value=default_on)
    if checked:
        selected_ids.append(s["file"])

if st.button("🚀 Generate Speech for Selected Sections"):
    st.info("Next step: we will wire this to step5a → step5b → step6 pipeline.")
    st.write("Selected files:", selected_ids)
