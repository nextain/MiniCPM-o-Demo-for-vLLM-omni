# Roadmap — wire the demo's full-duplex onto vllm-omni's full-duplex

> Tracker has moved to GitHub Issues. The phase plan and acceptance criteria are kept as the issue bodies.

## Goal

Connect the demo's **Audio Full-Duplex** mode to the **full-duplex implementation already living in our `vllm-omni` fork** (`projects/vllm-omni`). Both halves exist; the work is the interface in between, plus reviving the archived `/v1/omni` server endpoint or its equivalent so the demo's `worker.py duplex_*` methods can call into vllm-omni instead of running the model in-process via HF transformers.

The demo's other modes (Turn-based / Half-Duplex / Omni Full-Duplex) are **not part of this work**. naia-os alignment is a downstream consequence (Phase 3) and not a goal of Phase 2 itself.

## Issues

- [Tracking — demo ↔ vllm-omni full-duplex interface](https://github.com/nextain/MiniCPM-o-Demo-for-vLLM-omni/issues/1) (epic)
- [Phase 1 — Run web demo as-is, capture full-duplex reference](https://github.com/nextain/MiniCPM-o-Demo-for-vLLM-omni/issues/2)
- [Phase 2 — Wire demo's Audio Full-Duplex onto vllm-omni's full-duplex backend](https://github.com/nextain/MiniCPM-o-Demo-for-vLLM-omni/issues/3)
- [Phase 3 — Apply verified pattern in naia-os `minicpm-o.ts`](https://github.com/nextain/MiniCPM-o-Demo-for-vLLM-omni/issues/4) (downstream, not part of Phase 2)
- [Cross-phase consistency tests](https://github.com/nextain/MiniCPM-o-Demo-for-vLLM-omni/issues/5)

## Hardware constraint (recurring)

24 GB × 2 cannot host both the demo's HF transformers model **and** vllm-omni's PagedAttention model simultaneously. Whichever side we are working on, the other must be killed. Convention:

- Phase 1 work / reference capture → demo on, vllm-omni off
- Phase 2 work → vllm-omni on, demo on with `backend=vllm_omni` (no in-process model load)

## Phase 2 scope (the actual interface work)

vllm-omni side:
- `projects/vllm-omni/ref/omni_duplex_v1/serving_omni_duplex.py` is the previous `/v1/omni` full-duplex implementation (archived 2026-04-08, commit `4b4f351e`).
- It uses the model's streaming API (`MiniCPMO45.duplex.streaming_prefill` / `streaming_generate`), unlike `omni_connection.py` which is batch-only (Phase 11 lesson 17).
- Decision: revive/port that endpoint (or its equivalent) into the live server so `worker.py duplex_*` has something concrete to call.

Demo side:
- `worker.py duplex_prefill / duplex_generate / streaming_*` → forward chunks to the revived endpoint instead of going through `core/processors/unified.py` DuplexView.
- `core/processors/unified.py:1376` Multi-GPU dispatch is irrelevant when `backend=vllm_omni` (no local model load), but the file stays as Phase 1 reference.

## Out of scope

- Streamlit / mobile variants of the demo.
- Demo's Turn-based / Half-Duplex / Omni-Full-Duplex modes (only **Audio Full-Duplex** matters for this).
- naia-os audio-playback debugging — separate Phase 3, separate issue track.
- vllm-omni's `/v1/realtime` `omni_connection.py` batch flow — leave as-is; full-duplex is the parallel revived path.
