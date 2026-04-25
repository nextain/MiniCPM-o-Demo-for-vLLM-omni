"""vllm-omni backend processor — thin proxy, no GPU model load.

Mirrors `UnifiedProcessor`'s mode-switch surface (`set_chat_mode`,
`set_half_duplex_mode`, `set_duplex_mode`) so worker.py can swap
backends with no other code changes.

Stages (issue #3):

- Stage 1 (this file): chat (REST) — `/v1/chat/completions`. Text /
  image / audio input + optional TTS audio output.
- Stage 2: half-duplex — `/v1/realtime` (single-turn audio I/O).
- Stage 3 (deferred): full-duplex — vllm-omni `omni_connection.py` is
  batch-only (Phase 11 lesson 17); needs upstream change.

Selection:

    config.json `service.backend = "vllm_omni"`
        OR
    env `MINICPMO45_BACKEND=vllm_omni`

Reachability:

    config.json `service.vllm_omni_api_base` (default `http://localhost:8000`)
        OR env `VLLM_OMNI_API_BASE`
    Model id: `service.vllm_omni_model` (default `openbmb/MiniCPM-o-4_5`)
        OR env `VLLM_OMNI_MODEL`
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
import wave
from typing import Any, Optional

import httpx
import numpy as np

from core.processors.base import BaseProcessor, ProcessorMode
from core.schemas.chat import ChatRequest, ChatResponse
from core.schemas.common import (
    AudioContent,
    ImageContent,
    Message,
    TextContent,
    VideoContent,
)

logger = logging.getLogger(__name__)


# ============================================================
# Defaults
# ============================================================


_DEFAULT_API_BASE = "http://localhost:8000"
_DEFAULT_MODEL_ID = "openbmb/MiniCPM-o-4_5"
_HTTP_TIMEOUT_S = 180.0


# ============================================================
# Helpers — message conversion
# ============================================================


def _float32_pcm_to_wav_b64(float32_b64: str, sample_rate: int) -> str:
    """Re-pack base64 float32 PCM (the demo's AudioContent shape) into a
    WAV file (PCM16 LE) and return base64 of the WAV bytes — the format
    expected by `input_audio` content blocks.
    """
    raw = base64.b64decode(float32_b64)
    if len(raw) % 4 != 0:
        raw = raw[: len(raw) - (len(raw) % 4)]
    f32 = np.frombuffer(raw, dtype=np.float32)
    f32 = np.clip(f32, -1.0, 1.0)
    pcm16 = (f32 * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _content_to_openai_blocks(content: Any) -> list[dict[str, Any]]:
    """Convert a demo `Message.content` (str | list[ContentItem]) into
    OpenAI-style content blocks consumable by vllm-omni
    `/v1/chat/completions`.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return [{"type": "text", "text": str(content)}]

    blocks: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, TextContent):
            blocks.append({"type": "text", "text": item.text})
        elif isinstance(item, ImageContent):
            # demo: base64 PNG/JPEG bytes; OpenAI: data URL.
            # Default to png; the model decoder is content-sniffing so the
            # MIME hint mostly drives caching keys.
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{item.data}"},
                }
            )
        elif isinstance(item, AudioContent):
            # demo: base64 float32 PCM @ sample_rate; OpenAI: input_audio
            # data block as base64 WAV (or mp3). Repack into a minimal
            # WAV blob.
            wav_b64 = _float32_pcm_to_wav_b64(item.data, item.sample_rate)
            blocks.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": wav_b64, "format": "wav"},
                }
            )
        elif isinstance(item, VideoContent):
            # Defer video. The omni model accepts it but the demo's video
            # pipeline today only feeds it through duplex/streaming, not
            # turn-based chat.
            raise NotImplementedError(
                "VllmOmniProcessor: video content in turn-based chat is not "
                "yet supported (use Half-Duplex / Duplex modes)."
            )
        else:
            raise ValueError(f"unknown content item: {item!r}")
    return blocks


def _messages_to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        out.append(
            {
                "role": m.role.value if hasattr(m.role, "value") else str(m.role),
                "content": _content_to_openai_blocks(m.content),
            }
        )
    return out


# ============================================================
# Chat view — thin REST adapter
# ============================================================


class VllmOmniChatView:
    """ChatView surface (parity with `UnifiedProcessor.set_chat_mode()`)
    backed by vllm-omni `POST /v1/chat/completions`."""

    def __init__(self, processor: "VllmOmniProcessor"):
        self._processor = processor

    @property
    def kv_cache_length(self) -> int:
        # vllm-omni manages KV cache server-side; client-visible length
        # is not meaningful here.
        return 0

    def chat(
        self,
        request: ChatRequest,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        generate_audio: Optional[bool] = None,
    ) -> ChatResponse:
        start = time.time()
        try:
            return self._chat_impl(
                request, max_new_tokens, do_sample, generate_audio, start
            )
        except Exception as e:
            logger.exception("VllmOmniChatView chat failed")
            return ChatResponse(
                success=False,
                error=str(e),
                text="",
                duration_ms=(time.time() - start) * 1000,
            )

    def _chat_impl(
        self,
        request: ChatRequest,
        max_new_tokens: int,
        do_sample: bool,
        generate_audio: Optional[bool],
        start: float,
    ) -> ChatResponse:
        tts_cfg = request.tts if (request.tts is not None) else None
        tts_enabled = (
            generate_audio
            if generate_audio is not None
            else (bool(tts_cfg.enabled) if tts_cfg else False)
        )

        body: dict[str, Any] = {
            "model": self._processor.model_id,
            "messages": _messages_to_openai(request.messages),
            "max_tokens": max(1, int(max_new_tokens or 256)),
        }

        # Generation params (mirror the demo's GenerationConfig where
        # vllm-omni supports them; ignore the rest silently).
        gen = request.generation if request.generation is not None else None
        if gen is not None:
            if gen.temperature is not None:
                body["temperature"] = float(gen.temperature)
            if gen.top_p is not None:
                body["top_p"] = float(gen.top_p)
            # do_sample → temperature 0 if False
            if not do_sample and "temperature" not in body:
                body["temperature"] = 0.0

        # TTS — OpenAI-compatible audio modality.
        if tts_enabled:
            body["modalities"] = ["text", "audio"]
            body["audio"] = {"format": "pcm16"}
            # MiniCPM-o uses a reference audio for voice cloning; pass
            # via the "voice" field if vllm-omni supports it. Fallback:
            # the model uses its baked-in default voice when the field
            # is omitted.
            if tts_cfg is not None and (tts_cfg.ref_audio_path or tts_cfg.ref_audio_data):
                voice = tts_cfg.ref_audio_path or "ref"
                body["audio"]["voice"] = voice

        url = f"{self._processor.api_base.rstrip('/')}/v1/chat/completions"
        with httpx.Client(timeout=_HTTP_TIMEOUT_S) as client:
            resp = client.post(url, json=body)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"vllm-omni /v1/chat/completions {resp.status_code}: "
                f"{resp.text[:500]}"
            )
        data = resp.json()

        # OpenAI ChatCompletion → ChatResponse
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"vllm-omni response had no choices: {data}")
        msg = choices[0].get("message") or {}
        text = msg.get("content") or ""
        # Some servers return `content` as None when modality is audio.
        if text is None:
            text = ""

        audio_data: Optional[str] = None
        audio_sample_rate = 24000
        audio_msg = msg.get("audio")
        if isinstance(audio_msg, dict):
            audio_data = audio_msg.get("data")
            # OpenAI spec lacks an explicit sample_rate field on audio;
            # vllm-omni emits 24 kHz PCM16 (Phase 11 lesson 13). If a
            # future build emits a different rate it should expose it
            # under `audio.sample_rate`; honor that when present.
            sr = audio_msg.get("sample_rate")
            if isinstance(sr, int):
                audio_sample_rate = sr

        usage = data.get("usage") or {}
        return ChatResponse(
            success=True,
            text=text,
            audio_data=audio_data,
            audio_sample_rate=audio_sample_rate,
            tokens_generated=int(usage.get("completion_tokens") or 0),
            duration_ms=(time.time() - start) * 1000,
            token_stats={
                "input_tokens": int(usage.get("prompt_tokens") or 0),
                "generated_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
                "cached_tokens": 0,
            },
        )


# ============================================================
# Processor — top-level swap target
# ============================================================


class VllmOmniProcessor(BaseProcessor):
    """Thin proxy: forwards work to vllm-omni; no in-process model load.

    Drop-in for UnifiedProcessor when `service.backend == "vllm_omni"`.
    """

    def __init__(
        self,
        model_path: str,
        pt_path: Optional[str] = None,
        device: str = "cuda",
        ref_audio_path: Optional[str] = None,
        duplex_config: Optional[Any] = None,
        preload_both_tts: bool = True,
        compile: bool = False,
        chat_vocoder: str = "token2wav",
        attn_implementation: str = "auto",
        api_base: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        # All transformer-loading kwargs are accepted for signature
        # parity with UnifiedProcessor and silently ignored — model
        # loading happens server-side in vllm-omni.
        self.pt_path = pt_path
        self.ref_audio_path = ref_audio_path
        self.duplex_config = duplex_config
        self.preload_both_tts = preload_both_tts
        self.compile = compile
        self.chat_vocoder = chat_vocoder
        self.attn_implementation = attn_implementation

        self.api_base = (
            api_base
            or os.environ.get("VLLM_OMNI_API_BASE")
            or _DEFAULT_API_BASE
        )
        self.model_id = (
            model_id
            or os.environ.get("VLLM_OMNI_MODEL")
            or _DEFAULT_MODEL_ID
        )

        self._chat_view: Optional[VllmOmniChatView] = None
        # BaseProcessor.__init__ calls _load_model — we override it
        # below to do a reachability probe instead of a real load.
        super().__init__(model_path=model_path, device=device)

    # ---- BaseProcessor abstract overrides ------------------------------

    @property
    def mode(self) -> ProcessorMode:
        # vllm-omni adapter does not have a mutable mode in Stage 1;
        # surface CHAT as the default. set_half_duplex_mode raises.
        return ProcessorMode.CHAT

    def _load_model(self) -> None:
        """Reachability probe — raise loudly if vllm-omni is not up.

        Also installs `_NoLocalModel` as `self.model` so the demo's
        diagnostic paths (`worker._log_device_map`) don't crash on a
        None-model attribute.
        """
        url = f"{self.api_base.rstrip('/')}/v1/models"
        try:
            with httpx.Client(timeout=5.0) as c:
                resp = c.get(url)
                resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"VllmOmniProcessor: vllm-omni backend not reachable at "
                f"{self.api_base} ({e})"
            ) from e
        # Replace the BaseProcessor-installed `self.model = None` with a
        # sentinel that satisfies introspection (`named_modules` etc.).
        self.model = _NoLocalModel(api_base=self.api_base, model_id=self.model_id)
        logger.info(
            f"VllmOmniProcessor connected: api_base={self.api_base}, "
            f"model={self.model_id}"
        )

    def _release_resources(self) -> None:
        # Nothing to release — no in-process model.
        return None

    # ---- UnifiedProcessor surface parity ------------------------------

    def set_chat_mode(self) -> VllmOmniChatView:
        if self._chat_view is None:
            self._chat_view = VllmOmniChatView(self)
        return self._chat_view

    def set_half_duplex_mode(self):  # noqa: D401
        raise NotImplementedError(
            "vllm-omni half-duplex adapter not implemented yet (issue #3 Stage 2)."
        )

    def set_duplex_mode(self):  # noqa: D401
        raise NotImplementedError(
            "vllm-omni full-duplex adapter is deferred — vllm-omni "
            "omni_connection.py is batch-only (Phase 11 lesson 17). "
            "Use the demo's HF transformers backend (`service.backend=transformers`) "
            "for full-duplex sessions."
        )

    @property
    def kv_cache_length(self) -> int:
        return 0


class _NoLocalModel:
    """Sentinel for `processor.model` when no local model is loaded.

    Exposes minimal attributes the worker uses for diagnostics.
    """

    def __init__(self, api_base: str, model_id: str):
        self.api_base = api_base
        self.model_id = model_id

    def named_modules(self):  # used by worker._log_device_map
        return []

    def __repr__(self) -> str:
        return f"<NoLocalModel api_base={self.api_base} model={self.model_id}>"
