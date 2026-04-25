# Roadmap — vllm-omni alignment + naia-os reference

> Tracker has moved to GitHub Issues. The phase plan and acceptance criteria are kept as the issue bodies.

## Goal

This fork validates **vllm-omni `/v1/realtime` parity with the official MiniCPM-o 4.5 reference web demo**, and produces a known-good reference that the downstream naia-os Tauri client can align against.

## Issues

- [Tracking — 3-phase vllm-omni alignment + naia-os reference](https://github.com/nextain/MiniCPM-o-Demo-for-vLLM-omni/issues/1) (epic)
- [Phase 1 — Run web demo as-is, capture baseline](https://github.com/nextain/MiniCPM-o-Demo-for-vLLM-omni/issues/2)
- [Phase 2 — Swap model loading for vllm-omni `/v1/realtime`](https://github.com/nextain/MiniCPM-o-Demo-for-vLLM-omni/issues/3)
- [Phase 3 — Port verified pattern into naia-os `minicpm-o.ts`](https://github.com/nextain/MiniCPM-o-Demo-for-vLLM-omni/issues/4)
- [Cross-phase consistency tests — multi-turn English, image+audio, audio duration](https://github.com/nextain/MiniCPM-o-Demo-for-vLLM-omni/issues/5)

## Why these phases (not skip-ahead)

Skipping straight to vllm-omni-only ↔ naia-os comparison was tried (issue chain in nextain/naia-os#216, #219) and produced "transcript correct, audio garbled / repeats / chops" symptoms with no anchor for "what correct sounds like". Phase 1 is that anchor.

## Out of scope

- Streamlit / mobile variants of the demo.
- Functional changes to the demo UI beyond what Phase 2 forces (model-loading swap).
