# examples/

Smoke scripts for verifying the demo against an external **vllm-omni** server (i.e. `service.backend = "vllm_omni"`).

## Prerequisites

- An **vllm-omni server** reachable at `ws://localhost:8000` (or pass `--url`):

  ```bash
  cd projects/vllm-omni
  distrobox enter vllm-dev -- bash scripts/serve_async_chunk.sh
  ```

  Wait for `Application startup complete` (~90s on 2× RTX 3090 with `minicpmo_async_chunk` stage config).

- A PCM16 16kHz mono WAV fixture at `/tmp/test_input.wav` (override with `--wav`).

These scripts exercise `core/processors/vllm_omni_realtime.py` directly — no demo gateway/worker needed. They are the unit-level verification of the proxy view.

## Single-turn

```bash
.venv/base/bin/python examples/vllm_omni_realtime_single_turn.py
```

Expects: `turn.start` (implicit) → `transcript.delta` × N → `audio.delta` × N → `response.done` → `end_of_turn=True`.

Sample summary:

```
audio: ~3.7 MB float32 base64 (≈ 39s of 24kHz PCM16 → float32 doubled)
transcript: 'Hello! I am Qwen, …Tongyi Lab.'
end_of_turn reached: True
```

## Multi-turn

```bash
.venv/base/bin/python examples/vllm_omni_realtime_multi_turn.py
```

Walks turn 1 → 5s of inter-turn silence → turn 2. Verifies:

1. Turn 1 reaches `end_of_turn=True`.
2. **Inter-turn silence does NOT trigger a spurious commit** (the `_has_speech_seen` guard in the view).
3. Turn 2 starts cleanly after `reset_turn_state()` and reaches `end_of_turn=True`.

If either turn fails to commit or the inter-turn silence accidentally triggers `commit/eot`, the script exits non-zero with an explanatory message.

## What gets exercised

| Behavior | Single | Multi |
|---|:---:|:---:|
| WS connect + `session.update` | ✓ | ✓ |
| `input_audio_buffer.append` chunking | ✓ | ✓ |
| RMS silence detection (`SPEECH_RMS_THRESHOLD=200`, `SILENCE_TIMEOUT_MS=1500`, `MAX_BUFFER_MS=6000`) | ✓ | ✓ |
| `input_audio_buffer.commit` + `response.create` | ✓ | ✓ |
| `response.audio.delta` PCM16 24kHz → float32 base64 | ✓ | ✓ |
| `response.audio_transcript.delta` accumulation | ✓ | ✓ |
| `response.done` → `end_of_turn=True` + `reset_turn_state()` | ✓ | ✓ |
| `_has_speech_seen` guard on inter-turn silence |   | ✓ |
| Second turn after first turn complete |   | ✓ |
| `stop()` + `cleanup()` |  ✓ | ✓ |

## Browser-level smoke (Stage 2C)

These Python smokes verify the view in isolation. The full browser flow (mic → audio_duplex page → gateway → worker → view → vllm-omni) is the next step:

1. vllm-omni server up (as above).
2. Set `"backend": "vllm_omni"` in `config.json` (under `service`).
3. `bash start_all.sh` to launch the demo's gateway + worker.
4. Open `https://localhost:8006/audio_duplex`.
5. Speak one turn. Expect transcript + assistant audio reply through the demo's audio-player.

Hardware: vllm-omni holds the model on its GPUs; the demo with `backend=vllm_omni` loads no weights — `model_loaded=false` in `/health`, but `worker_ready=true`. They co-exist on 2× 24 GB.
