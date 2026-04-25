"""End-to-end single-turn smoke for VllmOmniRealtimeDuplexView.

Simulates the demo's audio_duplex flow: 1-second PCM16 chunks in,
generate per chunk, drain until end_of_turn.

Prerequisites
- vllm-omni server running at ws://localhost:8000 (set --url to override).
- Test fixture WAV at /tmp/test_input.wav (PCM16 16kHz mono). Override with --wav.

Run from repo root::

    .venv/base/bin/python examples/vllm_omni_realtime_single_turn.py
"""
import argparse
import os
import sys
import time
import wave

import numpy as np

# Make repo root importable when run from examples/ or arbitrary cwd
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.processors.vllm_omni_realtime import VllmOmniRealtimeProcessor

_DEFAULT_WAV = "/tmp/test_input.wav"
_DEFAULT_URL = "ws://localhost:8000"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=_DEFAULT_URL, help="vllm-omni server base URL")
    parser.add_argument("--wav", default=_DEFAULT_WAV, help="PCM16 16kHz mono WAV fixture")
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    args = parser.parse_args()

    with wave.open(args.wav, "rb") as w:
        sr = w.getframerate()
        pcm16 = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    print(f"input: {len(pcm16) / sr:.2f}s @ {sr}Hz, {len(pcm16)} samples")

    p = VllmOmniRealtimeProcessor(args.url, model=args.model)
    v = p.set_duplex_mode()
    try:
        v.prepare(system_prompt_text="You are a helpful assistant. Keep replies very short.")

        # Phase 1 — feed audio in 1-second chunks like the demo frontend
        chunk_size = sr  # 1s @ 16kHz = 16000 samples
        for i in range(0, len(pcm16), chunk_size):
            chunk = pcm16[i:i + chunk_size]
            v.prefill(audio_waveform=chunk)
            r = v.generate(force_listen=False)
            print(
                f"  feed chunk #{i // chunk_size}: is_listen={r.is_listen} "
                f"eot={r.end_of_turn} audio={'yes' if r.audio_data else 'no'} "
                f"silence={v._silence_ms:.0f}ms buffer={v._buffer_ms:.0f}ms"
            )
            if r.end_of_turn:
                break

        # Phase 2 — keep generating: silence accumulates → commit → drain response
        print("-- post-audio drain --")
        deadline = time.time() + 30
        empty_silent = np.zeros(int(sr * 0.5), dtype=np.int16)  # 500ms silence
        total_audio_bytes = 0
        full_text = ""
        n_calls = 0
        while time.time() < deadline:
            v.prefill(audio_waveform=empty_silent)
            r = v.generate(force_listen=False)
            n_calls += 1
            audio_b64 = r.audio_data or ""
            audio_size = len(audio_b64) * 3 // 4 if audio_b64 else 0  # base64 → bytes
            total_audio_bytes += audio_size
            full_text += r.text or ""
            print(
                f"  drain #{n_calls}: is_listen={r.is_listen} eot={r.end_of_turn} "
                f"audio={audio_size}B text={r.text!r} silence={v._silence_ms:.0f}ms "
                f"committed={v._committed} response_active={v._response_active}"
            )
            if r.end_of_turn:
                break
            time.sleep(0.5)

        print()
        print(f"=== summary ===")
        print(f"  total audio (float32 base64-decoded): {total_audio_bytes} bytes")
        print(f"  transcript: {full_text!r}")
        print(f"  end_of_turn reached: {r.end_of_turn}")
        return 0 if r.end_of_turn and total_audio_bytes > 0 else 1
    finally:
        v.stop()
        v.cleanup()


sys.exit(main())
