"""vllm-omni realtime backend processor (proxy to /v1/realtime).

Used when service.backend == "vllm_omni": no in-process model is loaded;
audio duplex calls are forwarded to an external vllm-omni server over
WebSocket using the OpenAI Realtime API protocol.

Hardware rationale: 2× 24 GB GPUs cannot host both the demo's HF transformers
model and vllm-omni's PagedAttention model at the same time. Setting
backend='vllm_omni' makes the demo skip its own model load entirely; vllm-omni
holds the model, the demo only runs the UI + worker glue + WS proxy.

Step 1 (this file as committed now): processor + view shaped like
core.processors.unified.UnifiedProcessor / DuplexView so that worker.py can
hold one without loading a model. The view's methods raise NotImplementedError;
the real /v1/realtime WebSocket client lands in Step 2.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class VllmOmniRealtimeDuplexView:
    """Duplex view proxying to vllm-omni /v1/realtime.

    Interface-compatible with `core.processors.unified.DuplexView`:
    prepare / prefill / generate / finalize / stop / cleanup /
    set_break / clear_break / is_break_set / is_stopped.

    Step 2 will replace the NotImplementedError stubs with an OpenAI Realtime
    API client (session.update -> input_audio_buffer.append -> commit ->
    response.create -> response.audio.delta + response.audio_transcript.delta
    -> response.done), following the pattern used by naia-os
    `shell/src/lib/voice/minicpm-o.ts` and vllm-omni
    `examples/online_serving/minicpm_o/realtime_e2e_test.py`.
    """

    def __init__(self, url: str, model: str):
        self.url = url
        self.model = model

    def prepare(
        self,
        system_prompt_text: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        prompt_wav_path: Optional[str] = None,
    ) -> str:
        raise NotImplementedError("VllmOmniRealtimeDuplexView.prepare — pending Step 2")

    def prefill(
        self,
        audio_waveform=None,
        audio_path: Optional[str] = None,
        frame_list=None,
        max_slice_nums: int = 1,
    ) -> dict:
        raise NotImplementedError("VllmOmniRealtimeDuplexView.prefill — pending Step 2")

    def generate(self, force_listen: bool = False):
        raise NotImplementedError("VllmOmniRealtimeDuplexView.generate — pending Step 2")

    def finalize(self) -> None:
        raise NotImplementedError("VllmOmniRealtimeDuplexView.finalize — pending Step 2")

    def stop(self) -> None:
        raise NotImplementedError("VllmOmniRealtimeDuplexView.stop — pending Step 2")

    def cleanup(self) -> None:
        raise NotImplementedError("VllmOmniRealtimeDuplexView.cleanup — pending Step 2")

    def set_break(self) -> None:
        raise NotImplementedError("VllmOmniRealtimeDuplexView.set_break — pending Step 2")

    def clear_break(self) -> None:
        raise NotImplementedError("VllmOmniRealtimeDuplexView.clear_break — pending Step 2")

    def is_break_set(self) -> bool:
        return False

    def is_stopped(self) -> bool:
        return False


class VllmOmniRealtimeProcessor:
    """Processor that proxies audio duplex to a vllm-omni server.

    Drop-in replacement for `UnifiedProcessor` when `service.backend == "vllm_omni"`.
    Does NOT load any local model — only `set_duplex_mode()` is supported
    (chat and half-duplex modes are out of scope for this proxy; the demo
    must use `backend='inproc'` if those modes are needed).
    """

    def __init__(self, url: str, model: str = "openbmb/MiniCPM-o-4_5"):
        self.url = url
        self.model = model
        self._duplex_view: Optional[VllmOmniRealtimeDuplexView] = None
        logger.info(
            f"VllmOmniRealtimeProcessor ready (url={url}, model={model}); "
            f"no local model loaded — audio duplex proxies to {url}/v1/realtime"
        )

    def set_duplex_mode(self) -> VllmOmniRealtimeDuplexView:
        if self._duplex_view is None:
            self._duplex_view = VllmOmniRealtimeDuplexView(self.url, self.model)
        return self._duplex_view

    def set_chat_mode(self):
        raise NotImplementedError(
            "vllm_omni backend: chat mode is not implemented (audio duplex only). "
            "Set service.backend='inproc' if chat is required."
        )

    def set_half_duplex_mode(self):
        raise NotImplementedError(
            "vllm_omni backend: half-duplex mode is not implemented (audio duplex only). "
            "Set service.backend='inproc' if half-duplex is required."
        )
