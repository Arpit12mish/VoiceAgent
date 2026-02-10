#!/usr/bin/env bash
set -euo pipefail

mkdir -p mp3_chunks
rm -f mp3_chunks/concat_list.txt

for f in wav_chunks/chunk_*.wav; do
  base=$(basename "$f" .wav)
  ffmpeg -y -i "$f" -codec:a libmp3lame -q:a 4 "mp3_chunks/${base}.mp3" >/dev/null 2>&1
done

for f in mp3_chunks/chunk_*.mp3; do
  echo "file '$f'" >> mp3_chunks/concat_list.txt
done

ffmpeg -y -f concat -safe 0 -i mp3_chunks/concat_list.txt -c copy final.mp3 >/dev/null 2>&1
echo "✅ final.mp3 created"
