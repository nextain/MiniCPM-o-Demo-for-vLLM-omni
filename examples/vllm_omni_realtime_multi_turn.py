"""Multi-turn smoke for VllmOmniRealtimeDuplexView.

Validates that:
1. Turn 1 commits + completes properly.
2. Inter-turn silence (no speech) does NOT trigger a spurious commit.
3. Turn 2 starts cleanly after turn 1's reset_turn_state.
4. _has_speech_seen guard blocks silence-only commits.

Prerequisites
- vllm-omni server running at ws://localhost:8000 (set --url to override).
- Test fixture WAV at /tmp/test_input.wav (PCM16 16kHz mono). Override with --wav.

Run from repo root::

    .venv/base/bin/python examples/vllm_omni_realtime_multi_turn.py
"""
import argparse
import os
import sys
import time
import wave

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.processors.vllm_omni_realtime import VllmOmniRealtimeProcessor

_DEFAULT_WAV = "/tmp/test_input.wav"
_DEFAULT_URL = "ws://localhost:8000"


def feed_turn(view, pcm16, sr, label: str):
    """Feed one turn's worth of audio in 1-second chunks like the demo."""
    print(f"\n--- {label}: feeding {len(pcm16) / sr:.2f}s audio in 1s chunks ---")
    chunk_size = sr
    for i in range(0, len(pcm16), chunk_size):
        chunk = pcm16[i:i + chunk_size]
        view.prefill(audio_waveform=chunk)
        r = view.generate(force_listen=False)
        print(
            f"  feed #{i // chunk_size}: is_listen={r.is_listen} eot={r.end_of_turn} "
            f"audio={'yes' if r.audio_data else 'no'} "
            f"speech_seen={view._has_speech_seen} silence={view._silence_ms:.0f}ms "
            f"buffer={view._buffer_ms:.0f}ms committed={view._committed}"
        )


def drain_until_eot(view, sr, label: str, max_seconds: float = 30):
    """Pad with 500ms silence chunks until end_of_turn."""
    print(f"\n--- {label}: drain until end_of_turn ---")
    silent = np.zeros(int(sr * 0.5), dtype=np.int16)
    deadline = time.time() + max_seconds
    total_audio = 0
    full_text = ""
    n = 0
    while time.time() < deadline:
        view.prefill(audio_waveform=silent)
        r = view.generate(force_listen=False)
        n += 1
        audio_size = (len(r.audio_data) * 3 // 4) if r.audio_data else 0
        total_audio += audio_size
        full_text += r.text or ""
        print(
            f"  drain #{n}: is_listen={r.is_listen} eot={r.end_of_turn} "
            f"audio={audio_size}B silence={view._silence_ms:.0f}ms "
            f"committed={view._committed} active={view._response_active}"
        )
        if r.end_of_turn:
            break
        time.sleep(0.5)
    return total_audio, full_text, r.end_of_turn


def inter_turn_silence(view, sr, seconds: float):
    """Feed silence chunks between turns. Should NOT trigger commit."""
    print(f"\n--- inter-turn silence: {seconds}s of silence ---")
    silent = np.zeros(int(sr * 1.0), dtype=np.int16)
    n = int(seconds)
    for i in range(n):
        view.prefill(audio_waveform=silent)
        r = view.generate(force_listen=False)
        print(
            f"  inter #{i + 1}: is_listen={r.is_listen} eot={r.end_of_turn} "
            f"speech_seen={view._has_speech_seen} silence={view._silence_ms:.0f}ms "
            f"committed={view._committed}"
        )
        if r.end_of_turn or view._committed:
            print("  !! UNEXPECTED: silence-only triggered commit/eot. Bug.")
            return False
        time.sleep(0.1)
    print(f"  ✓ silence-only inter-turn safe (no commit, no eot)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=_DEFAULT_URL)
    parser.add_argument("--wav", default=_DEFAULT_WAV)
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    args = parser.parse_args()

    with wave.open(args.wav, "rb") as w:
        sr = w.getframerate()
        pcm16 = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

    p = VllmOmniRealtimeProcessor(args.url, model=args.model)
    v = p.set_duplex_mode()
    try:
        v.prepare(system_prompt_text="You are a helpful assistant. Reply briefly.")

        # Turn 1
        feed_turn(v, pcm16, sr, "TURN 1")
        a1, t1, eot1 = drain_until_eot(v, sr, "TURN 1")
        if not eot1:
            print("FAIL: turn 1 did not reach end_of_turn")
            return 1
        print(f"  turn 1 result: {a1} bytes audio, transcript={t1!r}")

        # Inter-turn silence — must NOT trigger commit
        if not inter_turn_silence(v, sr, seconds=5):
            return 2

        # Turn 2
        feed_turn(v, pcm16, sr, "TURN 2")
        a2, t2, eot2 = drain_until_eot(v, sr, "TURN 2")
        if not eot2:
            print("FAIL: turn 2 did not reach end_of_turn")
            return 3
        print(f"  turn 2 result: {a2} bytes audio, transcript={t2!r}")

        print()
        print(f"=== multi-turn summary ===")
        print(f"  turn 1: audio={a1}B, transcript len={len(t1)}")
        print(f"  turn 2: audio={a2}B, transcript len={len(t2)}")
        print(f"  inter-turn silence safe: yes")
        return 0 if (a1 > 0 and a2 > 0) else 4
    finally:
        v.stop()
        v.cleanup()


sys.exit(main())
