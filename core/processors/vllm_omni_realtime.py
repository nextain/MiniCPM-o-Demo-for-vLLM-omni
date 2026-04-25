"""vllm-omni realtime backend processor (proxy to /v1/realtime).

Used when ``service.backend == "vllm_omni"``: no in-process model is loaded;
audio duplex calls are forwarded to an external vllm-omni server over a
WebSocket using the OpenAI Realtime API protocol.

Hardware rationale
==================
2× 24 GB GPUs cannot host both the demo's HF transformers model and
vllm-omni's PagedAttention model. ``backend='vllm_omni'`` makes the demo
skip its own model load entirely; vllm-omni holds the model, the demo
runs only the UI + worker glue + this proxy view.

Wire pattern
============
This view is a sync façade over an async ``/v1/realtime`` WebSocket. It
owns its own asyncio event loop on a dedicated thread; sync DuplexView
methods bridge via ``run_coroutine_threadsafe``.

Turn boundary: client-side RMS silence detector. The demo's audio_duplex
frontend sends 1-second PCM16 chunks blindly (no client-side VAD), and
vllm-omni's server_vad is declared-but-unimplemented (Phase 11 lesson 15).
So this view injects a silence detector mirroring naia-os
``shell/src/lib/voice/minicpm-o.ts``: speech RMS threshold 200 (Int16
scale), 1500 ms silence timer, 6000 ms max-buffer fallback. The demo UI's
``force_listen=True`` extends the silence timer (no commit yet).

Wire flow
=========
::

    prepare()  →  WS connect  →  session.created  →  session.update
                  (model, modalities=[text,audio], input_audio_format=pcm16,
                   output_audio_format=pcm16, instructions)
                  →  background recv task started

    prefill(waveform)  →  PCM16 base64  →  input_audio_buffer.append
                          + RMS-based silence/buffer accounting

    generate(force_listen=False):
       if !committed:
         if (silence ≥ 1500 ms or buffer ≥ 6000 ms) and !force_listen:
            → input_audio_buffer.commit + response.create
         else:
            → return is_listen=True (still waiting on user)

       drain server-event queue (audio_transcript.delta / audio.delta /
       response.done / error). Concatenate any PCM16 24kHz audio bytes into
       a float32 base64 blob (matches what the demo's audio-player.js
       expects from the in-process Phase 1 path).

       return DuplexGenerateResult(is_listen, text, audio_data,
                                   end_of_turn, current_time)

       on response.done → reset turn state for the next turn

    finalize() → no-op (server holds the state)
    stop()    → response.cancel
    cleanup() → close WS, stop loop, join thread

Multi-turn: each turn independent. ``omni_connection.py`` had its
multi-turn history disabled in vllm-omni ``fd273bdc`` (literal
``"[audio input]"`` placeholders confused the model). Until ASR-based
history lands upstream, every turn starts fresh — same trade-off as
naia-os.
"""

import asyncio
import base64
import json
import logging
import threading
from typing import Optional

import numpy as np
import websockets

from core.schemas.duplex import DuplexGenerateResult

logger = logging.getLogger(__name__)


# Constants matching naia-os shell/src/lib/voice/minicpm-o.ts
_SPEECH_RMS_THRESHOLD = 200       # Int16 PCM RMS threshold (0-32767 scale)
_SILENCE_TIMEOUT_MS = 1500        # silence duration before commit
_MAX_BUFFER_MS = 6000             # hard cap on accumulated audio per turn
_MAX_SILENCE_BUFFER_MS = 30000    # if no speech seen, clear server buffer at this point
_INPUT_SAMPLE_RATE = 16000        # demo emits 16 kHz mono
_OUTPUT_SAMPLE_RATE = 24000       # vllm-omni outputs 24 kHz mono PCM16

_LOOP_START_TIMEOUT_S = 5.0
_DEFAULT_RPC_TIMEOUT_S = 30.0
_SESSION_CREATED_TIMEOUT_S = 10.0


def _waveform_to_pcm16(waveform: np.ndarray) -> np.ndarray:
    """Coerce demo audio (float32 in [-1,1] or int16) to int16 PCM."""
    if waveform.dtype == np.int16:
        return waveform
    return (np.clip(waveform.astype(np.float32), -1.0, 1.0) * 32767).astype(np.int16)


def _rms_int16(pcm16: np.ndarray) -> float:
    if pcm16.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(pcm16.astype(np.float64) ** 2)))


class VllmOmniRealtimeDuplexView:
    """Duplex view proxying to vllm-omni ``/v1/realtime``.

    Interface-compatible with :class:`core.processors.unified.DuplexView`:
    ``prepare`` / ``prefill`` / ``generate`` / ``finalize`` / ``stop`` /
    ``cleanup`` / ``set_break`` / ``clear_break`` / ``is_break_set`` /
    ``is_stopped``.
    """

    def __init__(self, url: str, model: str):
        self.url = url
        self.model = model
        self.realtime_url = f"{url.rstrip('/')}/v1/realtime"

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_ready = threading.Event()

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._stopped = False

        # Turn state — accessed from the loop thread via _run()
        self._silence_ms = 0.0
        self._buffer_ms = 0.0
        self._has_pending_audio = False
        self._has_speech_seen = False    # at least one chunk crossed RMS threshold this turn
        self._committed = False
        self._response_active = False
        self._end_of_turn = False
        self._transcript = ""
        self._chunk_count = 0

        # Server → client event queue (created inside the loop)
        self._event_queue: Optional[asyncio.Queue] = None
        self._recv_task: Optional[asyncio.Task] = None

    # ─── async loop bootstrap ────────────────────────────────────────

    def _ensure_loop(self) -> None:
        if self._loop is not None and self._loop_thread is not None and self._loop_thread.is_alive():
            return

        def runner():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._event_queue = asyncio.Queue()
            self._loop = loop
            self._loop_ready.set()
            try:
                loop.run_forever()
            finally:
                loop.close()

        self._loop_ready.clear()
        self._loop_thread = threading.Thread(
            target=runner, name="vllm-omni-realtime-loop", daemon=True
        )
        self._loop_thread.start()
        if not self._loop_ready.wait(timeout=_LOOP_START_TIMEOUT_S):
            raise RuntimeError("event loop failed to start")

    def _run(self, coro, timeout: float = _DEFAULT_RPC_TIMEOUT_S):
        if self._loop is None or not self._loop.is_running():
            raise RuntimeError("event loop not running; call prepare() first")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # ─── async impl (runs in loop thread) ────────────────────────────

    async def _async_connect_and_configure(self, instructions: str) -> None:
        self._ws = await websockets.connect(
            self.realtime_url,
            max_size=64 * 1024 * 1024,
            open_timeout=10,
        )
        msg = await asyncio.wait_for(self._ws.recv(), timeout=_SESSION_CREATED_TIMEOUT_S)
        evt = json.loads(msg) if isinstance(msg, str) else {}
        if evt.get("type") != "session.created":
            raise RuntimeError(f"unexpected first event from /v1/realtime: {evt.get('type')}")

        await self._ws.send(json.dumps({
            "type": "session.update",
            "model": self.model,
            "session": {
                "modalities": ["text", "audio"],
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "instructions": instructions,
            },
        }))
        self._connected = True
        self._recv_task = asyncio.create_task(self._async_recv_loop())

    async def _async_recv_loop(self) -> None:
        try:
            async for raw in self._ws:
                if not isinstance(raw, str):
                    continue
                try:
                    evt = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                t = evt.get("type")
                if t == "response.audio_transcript.delta":
                    await self._event_queue.put({"kind": "transcript", "delta": evt.get("delta", "")})
                elif t == "response.audio.delta":
                    await self._event_queue.put({"kind": "audio", "delta": evt.get("delta", "")})
                elif t == "response.done":
                    await self._event_queue.put({"kind": "done"})
                elif t == "error":
                    await self._event_queue.put({"kind": "error", "message": evt.get("error", str(evt))})
                # other events (session.updated, response.created, audio.start, …) are informational
        except websockets.ConnectionClosed:
            await self._event_queue.put({"kind": "closed"})
        except Exception as e:
            logger.exception("vllm_omni_realtime: recv loop error: %s", e)
            await self._event_queue.put({"kind": "error", "message": str(e)})

    async def _async_send_event(self, payload: dict) -> None:
        if self._ws is None:
            raise RuntimeError("WS not connected")
        await self._ws.send(json.dumps(payload))

    async def _async_drain_queue(self) -> list:
        events = []
        while not self._event_queue.empty():
            try:
                events.append(self._event_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return events

    async def _async_close(self) -> None:
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._connected = False

    # ─── turn-state helpers ──────────────────────────────────────────

    def _reset_turn_state(self) -> None:
        self._silence_ms = 0.0
        self._buffer_ms = 0.0
        self._has_pending_audio = False
        self._has_speech_seen = False
        self._committed = False
        self._response_active = False
        self._end_of_turn = False
        self._transcript = ""
        self._chunk_count = 0

    async def _async_drain_queue_now(self) -> int:
        """Drop any queued events (used on stop/break to start the next turn clean)."""
        dropped = 0
        if self._event_queue is None:
            return 0
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
                dropped += 1
            except asyncio.QueueEmpty:
                break
        return dropped

    # ─── public sync interface ───────────────────────────────────────

    def prepare(
        self,
        system_prompt_text: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        prompt_wav_path: Optional[str] = None,
    ) -> str:
        """Open WS, do session.update, prime the recv loop."""
        self._ensure_loop()
        instructions = system_prompt_text or "You are a helpful conversational assistant."
        # ref_audio / prompt_wav: vllm-omni server uses its own ref handling — ignored here.
        self._run(self._async_connect_and_configure(instructions))
        self._reset_turn_state()
        self._stopped = False
        logger.info(
            f"vllm_omni_realtime: connected to {self.realtime_url} (model={self.model})"
        )
        return f"vllm_omni_session_{id(self)}"

    def prefill(
        self,
        audio_waveform: Optional[np.ndarray] = None,
        audio_path: Optional[str] = None,
        frame_list=None,
        max_slice_nums: int = 1,
    ) -> dict:
        """Append a chunk to input_audio_buffer + update silence/RMS timers."""
        if audio_path is not None and audio_waveform is None:
            import librosa
            audio_waveform, _ = librosa.load(audio_path, sr=_INPUT_SAMPLE_RATE, mono=True)
        if audio_waveform is None or len(audio_waveform) == 0:
            return {}
        if frame_list:
            logger.warning("vllm_omni_realtime: video/image frames not supported, ignoring frame_list")

        pcm16 = _waveform_to_pcm16(np.asarray(audio_waveform).reshape(-1))
        chunk_ms = (len(pcm16) / _INPUT_SAMPLE_RATE) * 1000.0

        rms = _rms_int16(pcm16)
        if rms >= _SPEECH_RMS_THRESHOLD:
            self._silence_ms = 0.0
            self._has_speech_seen = True
        else:
            self._silence_ms += chunk_ms
        self._buffer_ms += chunk_ms
        self._has_pending_audio = True

        b64 = base64.b64encode(pcm16.tobytes()).decode()
        self._run(self._async_send_event({
            "type": "input_audio_buffer.append",
            "audio": b64,
        }))

        # Silence-only buffer cap: if user never spoke and we accumulated a lot,
        # clear the server buffer so memory does not grow without bound. The demo
        # frontend keeps the mic open between turns and emits 1-second chunks
        # regardless of speech, so this path matters in practice.
        if not self._has_speech_seen and self._buffer_ms >= _MAX_SILENCE_BUFFER_MS:
            try:
                self._run(self._async_send_event({"type": "input_audio_buffer.clear"}))
            except Exception:
                pass
            self._silence_ms = 0.0
            self._buffer_ms = 0.0
            self._has_pending_audio = False
            logger.info(
                f"vllm_omni_realtime: cleared server buffer after "
                f"{_MAX_SILENCE_BUFFER_MS}ms of silence (no speech)"
            )
        return {}

    def generate(self, force_listen: bool = False) -> DuplexGenerateResult:
        """Decide commit/listen, drain server events, surface 1 result."""
        if self._stopped:
            return DuplexGenerateResult(is_listen=True, end_of_turn=True)

        # Pre-commit phase: are we ready to flush this turn's audio?
        if not self._committed:
            # Only commit when we have actually heard speech this turn — silence-only
            # buffer must not be sent (the demo frontend streams 1-second chunks even
            # while the user is silent, so without this guard we would send a silence
            # commit between turns and confuse the model with empty input).
            ready_to_commit = (
                self._has_pending_audio
                and self._has_speech_seen
                and not force_listen
                and (self._silence_ms >= _SILENCE_TIMEOUT_MS or self._buffer_ms >= _MAX_BUFFER_MS)
            )
            if not ready_to_commit:
                self._chunk_count += 1
                return DuplexGenerateResult(
                    is_listen=True,
                    end_of_turn=False,
                    current_time=self._chunk_count,
                )
            # Commit input + ask the server for a response
            self._run(self._async_send_event({"type": "input_audio_buffer.commit"}))
            self._run(self._async_send_event({"type": "response.create"}))
            self._committed = True
            self._response_active = True
            logger.info(
                f"vllm_omni_realtime: committed turn "
                f"(buffer={self._buffer_ms:.0f}ms, silence={self._silence_ms:.0f}ms)"
            )

        # Drain server events accumulated since last call
        events = self._run(self._async_drain_queue())

        audio_pcm16_chunks: list = []
        new_transcript_delta = ""
        for e in events:
            kind = e["kind"]
            if kind == "audio":
                pcm_bytes = base64.b64decode(e["delta"]) if e.get("delta") else b""
                if pcm_bytes:
                    audio_pcm16_chunks.append(pcm_bytes)
            elif kind == "transcript":
                d = e.get("delta", "")
                self._transcript += d
                new_transcript_delta += d
            elif kind == "done":
                self._response_active = False
                self._end_of_turn = True
            elif kind == "error":
                logger.error(f"vllm_omni_realtime: server error: {e.get('message')}")
                self._response_active = False
                self._end_of_turn = True
            elif kind == "closed":
                logger.warning("vllm_omni_realtime: WS closed unexpectedly")
                self._response_active = False
                self._end_of_turn = True

        # Convert PCM16 24kHz → float32 base64 (demo audio-player expects float32)
        audio_data: Optional[str] = None
        if audio_pcm16_chunks:
            pcm16_concat = b"".join(audio_pcm16_chunks)
            arr_int16 = np.frombuffer(pcm16_concat, dtype=np.int16)
            arr_float32 = arr_int16.astype(np.float32) / 32767.0
            audio_data = base64.b64encode(arr_float32.tobytes()).decode()

        self._chunk_count += 1
        result = DuplexGenerateResult(
            is_listen=False if (self._response_active or audio_data is not None) else True,
            text=new_transcript_delta,
            audio_data=audio_data,
            end_of_turn=self._end_of_turn,
            current_time=self._chunk_count,
        )

        if self._end_of_turn:
            self._reset_turn_state()
        return result

    def finalize(self) -> None:
        # /v1/realtime keeps state server-side; nothing to flush client-side.
        pass

    def stop(self) -> None:
        self._stopped = True
        if self._loop is not None and self._loop.is_running() and self._ws is not None:
            try:
                self._run(
                    self._async_send_event({"type": "response.cancel"}),
                    timeout=3.0,
                )
            except Exception:
                pass
            try:
                self._run(self._async_drain_queue_now(), timeout=2.0)
            except Exception:
                pass
        self._reset_turn_state()
        logger.info("vllm_omni_realtime: stopped")

    def cleanup(self) -> None:
        if self._loop is not None and self._loop.is_running():
            try:
                self._run(self._async_close(), timeout=5.0)
            except Exception:
                pass
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass
        if self._loop_thread is not None and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=3.0)
        self._loop = None
        self._loop_thread = None
        self._loop_ready.clear()
        self._event_queue = None
        logger.info("vllm_omni_realtime: cleanup done")

    def set_break(self) -> None:
        # Demo's break = interrupt current speak. Map to response.cancel + reset
        # turn so the next prefill starts a clean buffer.
        if self._response_active and self._loop is not None and self._loop.is_running():
            try:
                self._run(
                    self._async_send_event({"type": "response.cancel"}),
                    timeout=3.0,
                )
            except Exception:
                pass
            try:
                self._run(self._async_drain_queue_now(), timeout=2.0)
            except Exception:
                pass
        self._reset_turn_state()
        self._end_of_turn = True

    def clear_break(self) -> None:
        self._reset_turn_state()

    def is_break_set(self) -> bool:
        return self._end_of_turn and not self._response_active

    def is_stopped(self) -> bool:
        return self._stopped


class VllmOmniRealtimeProcessor:
    """Processor that proxies audio duplex to a vllm-omni server.

    Drop-in replacement for :class:`core.processors.unified.UnifiedProcessor`
    when ``service.backend == "vllm_omni"``. Loads no local model. Only
    ``set_duplex_mode()`` is supported — chat and half-duplex modes need
    ``backend='inproc'``.
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
