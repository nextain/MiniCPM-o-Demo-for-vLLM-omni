"""Cross-phase consistency test helpers.

The same scenario JSON drives:
  - Phase 1 = HF transformers via the demo gateway HTTP/WS endpoints.
  - Phase 2 = vllm-omni `/v1/realtime` WebSocket directly.
  - Phase 3 = naia-os equivalent via the verified ts-client wrapper.

Each adapter returns a `TurnResult` so the assertion logic in
`tests/test_cross_phase_consistency.py` is identical for all three phases.
"""

from .phase_adapters import (
    PhaseAdapter,
    Phase1HFAdapter,
    Phase2VllmOmniAdapter,
    Phase3NaiaTsClientAdapter,
    Turn,
    TurnResult,
    load_scenario,
    assert_turn_against_expected,
    audio_rms,
    audio_duration_seconds,
)

__all__ = [
    "PhaseAdapter",
    "Phase1HFAdapter",
    "Phase2VllmOmniAdapter",
    "Phase3NaiaTsClientAdapter",
    "Turn",
    "TurnResult",
    "load_scenario",
    "assert_turn_against_expected",
    "audio_rms",
    "audio_duration_seconds",
]
