from pathlib import Path
from TTS.api import TTS

TEXT_DIR = Path("text_chunks")
OUT_DIR = Path("wav_chunks")
OUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "tts_models/en/ljspeech/vits"

def main():
    files = sorted(TEXT_DIR.glob("chunk_*.txt"))
    if not files:
        raise FileNotFoundError("No chunk_*.txt found. Run step5a_make_text_chunks_from_step3.py first.")

    print("Loading model:", MODEL_NAME)
    tts = TTS(model_name=MODEL_NAME, progress_bar=True, gpu=False)

    for i, f in enumerate(files, start=1):
        text = f.read_text(encoding="utf-8").strip()
        if not text:
            continue

        out_path = OUT_DIR / f"chunk_{i:04d}.wav"

        # resume support
        if out_path.exists() and out_path.stat().st_size > 2000:
            print(f"Skipping existing {out_path.name}")
            continue

        print(f"Generating {out_path.name} | chars={len(text)}")
        tts.tts_to_file(text=text, file_path=str(out_path))

    print(f"\n✅ WAV chunks ready in {OUT_DIR}/")

if __name__ == "__main__":
    main()

