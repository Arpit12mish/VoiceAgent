import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.api import TTS

# ✅ Allowlist XTTS config classes for PyTorch safe loading
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig])

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
TEXT_FILE = "text_chunks/chunk_0003.txt"
VOICE_SAMPLE = ["manan_tts_audio.wav", "manan_tts_audio_02.wav", "voice_sample_clean.wav"]
OUTPUT_FILE = "chunk3_xtts_indian_01.wav"

print("Loading XTTS model...")
tts = TTS(model_name=MODEL_NAME, progress_bar=True, gpu=False)

text = open(TEXT_FILE, encoding="utf-8").read()

print("Generating Indian voice audio...")
tts.tts_to_file(
    text=text,
    speaker_wav=VOICE_SAMPLE,
    language="en",
    file_path=OUTPUT_FILE
)

print("✅ Done. Output:", OUTPUT_FILE)
