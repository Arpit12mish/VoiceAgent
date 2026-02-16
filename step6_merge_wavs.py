from pathlib import Path
import argparse
import wave

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    wav_dir = Path(args.wav_dir)
    out_path = Path(args.out)

    files = sorted(wav_dir.glob("chunk_*.wav"))
    if not files:
        raise FileNotFoundError("No chunk_*.wav found to merge.")

    with wave.open(str(files[0]), "rb") as w0:
        params = w0.getparams()

    with wave.open(str(out_path), "wb") as out:
        out.setparams(params)
        for f in files:
            with wave.open(str(f), "rb") as w:
                if w.getparams() != params:
                    raise ValueError(f"WAV params mismatch in {f.name}. All chunks must match.")
                out.writeframes(w.readframes(w.getnframes()))

    print(f"✅ Merged {len(files)} chunks into: {out_path}")

if __name__ == "__main__":
    main()
