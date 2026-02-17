import os
import uuid
from typing import Optional

from TTS.api import TTS

# Cache the model in-memory so we don't reload each click
_MODEL = None
_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

def get_model(use_gpu: bool):
    global _MODEL
    if _MODEL is None:
        _MODEL = TTS(_MODEL_NAME, gpu=use_gpu)
    return _MODEL

def synthesize_xtts(
    text: str,
    speaker_wav_path: str,
    out_dir: str,
    language: str = "en",
    use_gpu: bool = False,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"xtts_{uuid.uuid4().hex[:10]}.wav")

    tts = get_model(use_gpu=use_gpu)
    tts.tts_to_file(
        text=text,
        file_path=out_path,
        speaker_wav=speaker_wav_path,
        language=language,
    )
    return out_path
