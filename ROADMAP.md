# Roadmap — wire the demo's Audio Full-Duplex onto vllm-omni

> Tracker has moved to GitHub Issues. The phase plan and acceptance criteria are kept as the issue bodies.

## Goal

Connect the demo's **Audio Full-Duplex** mode to **vllm-omni's `/v1/realtime`** endpoint (OpenAI Realtime API spec). The work is the demo-side client + worker.py forwarding — vllm-omni server is left untouched (it already streams audio output chunk-by-chunk via `ResponseAudioDelta`).

The demo's other modes (Turn-based / Half-Duplex / Omni-Full-Duplex) are **not part of this work**. naia-os alignment (Phase 3) is downstream and not a goal of Phase 2 itself.

## Issues

- [Tracking — demo ↔ vllm-omni full-duplex interface](https://github.com/nextain/MiniCPM-o-Demo-for-vLLM-omni/issues/1) (epic)
- [Phase 1 — Run web demo as-is, capture full-duplex reference](https://github.com/nextain/MiniCPM-o-Demo-for-vLLM-omni/issues/2)
- [Phase 2 — Wire demo's Audio Full-Duplex onto vllm-omni's `/v1/realtime`](https://github.com/nextain/MiniCPM-o-Demo-for-vLLM-omni/issues/3)
- [Phase 3 — Apply verified pattern in naia-os `minicpm-o.ts`](https://github.com/nextain/MiniCPM-o-Demo-for-vLLM-omni/issues/4) (downstream, not part of Phase 2)
- [Cross-phase consistency tests](https://github.com/nextain/MiniCPM-o-Demo-for-vLLM-omni/issues/5)

## Hardware constraint (recurring)

24 GB × 2 cannot host both the demo's HF transformers model **and** vllm-omni's PagedAttention model simultaneously. Whichever side we are working on, the other must be killed. Convention:

- Phase 1 work / reference capture → demo on, vllm-omni off
- Phase 2 work → vllm-omni on, demo on with `backend=vllm_omni` (no in-process model load)

## Phase 2 scope (the actual interface work)

vllm-omni side:
- **No changes.** `/v1/realtime` (`vllm_omni/entrypoints/openai/realtime/omni_connection.py`) already streams audio output chunk-by-chunk per OpenAI Realtime spec. naia-os already uses this path in production.
- Reference clients: `examples/online_serving/minicpm_o/realtime_e2e_test.py` (vllm-omni) and `naia-os/shell/src/lib/voice/minicpm-o.ts`.

Demo side:
- New `core/processors/vllm_omni_realtime.py` — `VllmOmniRealtimeProcessor` + `VllmOmniRealtimeDuplexView`. The view holds a `/v1/realtime` WebSocket connection and matches the existing `DuplexView` interface (`prepare / prefill / generate / finalize / stop / cleanup`).
- `worker.py` selects the proxy processor when `service.backend == "vllm_omni"`; otherwise stays on the in-process `UnifiedProcessor` (Phase 1).
- `config.py` adds `service.backend / vllm_omni_url / vllm_omni_model`. Default `backend=inproc` keeps Phase 1 behavior.
- `core/processors/unified.py:1376` Multi-GPU dispatch is irrelevant when `backend=vllm_omni` (no local model load); kept as Phase 1 reference.

### Stage breakdown

| Stage | Status | Scope |
|---|---|---|
| **2A** | ✅ done | config field + `VllmOmniRealtimeProcessor` / `…DuplexView` stub + worker dispatch. Result: `backend=vllm_omni` boots the worker without loading a local model; duplex calls land on the stub view (raises `NotImplementedError`). |
| **2B** | next | Implement `VllmOmniRealtimeDuplexView` against `/v1/realtime`. **Pattern: naia-os `minicpm-o.ts`** — RMS-based client-side silence detection (`SPEECH_RMS_THRESHOLD=200`, `silenceTimer=1500ms`, `maxBufferTimer=6000ms`) decides when to emit `input_audio_buffer.commit` + `response.create`. The demo frontend has no silence detector of its own and emits 1-second chunks blindly, so the view must inject one. `force_listen=True` from the demo UI maps to "do not commit yet" (extends silenceTimer). |
| **2C** | after 2B | Smoke: serve vllm-omni with `minicpmo_async_chunk` config, run demo with `backend=vllm_omni`, drive the audio_duplex page from a browser. Validate transcript + audio output, multi-turn behavior. |
| **2D** | after 2C | Document the wire mapping (demo audio_chunk → `input_audio_buffer.append`; demo `force_listen` → silence-timer extend; demo `is_listen/end_of_turn` ← derived from response phase) so naia-os Phase 3 can reuse the choices. |

### Wire flow (Phase 2)

```
demo client (browser audio_duplex page)
  ↓ existing demo WebSocket protocol
demo worker.py duplex_prefill / duplex_generate
  ↓ NEW: forward via vllm_omni_realtime processor
vllm-omni /v1/realtime WebSocket
  - session.update (modalities, audio formats)
  - input_audio_buffer.append (base64 PCM16 16kHz, chunked)
  - input_audio_buffer.commit  (end of user turn)
  - response.create
  ↑ ResponseAudioTranscriptDelta + ResponseAudioDelta + ResponseAudioDone
  ↓ demo worker.py converts back to its own audio chunk events
demo client plays audio
```

## Out of scope

- Streamlit / mobile variants of the demo.
- Demo's Turn-based / Half-Duplex / Omni-Full-Duplex modes (only **Audio Full-Duplex** matters).
- naia-os audio-playback debugging — separate Phase 3, separate issue track.
- vllm-omni server changes — reverted on 2026-04-25 (see vllm-omni `.agents/contribution-journey.md` Phase 12). archive `/v1/omni` revival was attempted then discarded after cross-review showed `/v1/realtime` is functionally equivalent and spec-compliant.

## History

| Date | Direction | Why |
|---|---|---|
| 2026-04-25 (early) | archive `/v1/omni` revival on vllm-omni side | misread of vllm-omni Phase 11 lesson 17 ("batch-only" → assumed audio output WS streaming was unsupported) |
| 2026-04-25 (later) | reverted; `/v1/realtime` from demo side | cross-review (Plan + Explore agents) showed omni_connection.py streams chunk-by-chunk; archive `/v1/omni` is redundant + not spec-compliant |
