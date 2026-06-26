import asyncio
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from . import log_writer
from src.inference.shared.convfill_engine import ConvFillEngine, EngineSink, EngineError


class _WebSink(EngineSink):
    """Routes engine turn events into the session's WebSocket emits + JSONL
    logging. Every method forwards to the matching session callback, so all of
    the (intricate) logging logic stays in `ConvFillSession` untouched."""

    def __init__(self, session: "ConvFillSession"):
        self.s = session

    def on_thought(self, thought: str) -> None:
        self.s._on_thought(thought)

    def on_response_fragment(self, text: str) -> None:
        self.s._on_response_fragment(text)

    def on_rag_context(self, text: str) -> None:
        self.s._on_rag_context(text)

    def on_mcp_context(self, text: str) -> None:
        self.s._on_mcp_context(text)

    def on_turn_complete(self, final_response: str, final_thoughts: list) -> None:
        self.s._on_turn_complete(final_response, final_thoughts)

    def on_frontend_inference(
        self,
        ms: float,
        tokens: Optional[int] = None,
        first_sentence_ms: Optional[float] = None,
        first_token_ms: Optional[float] = None,
    ) -> None:
        self.s._on_frontend_inference(ms, tokens, first_sentence_ms, first_token_ms)

    def on_phrase_start(self, thought: str, gap_ms: float) -> None:
        self.s._on_phrase_start(thought, gap_ms)

    def stage_frontend_ms(self, ms: Optional[float]) -> None:
        self.s._staged_frontend_ms = ms

    def audio_in_flight(self) -> int:
        return self.s._get_audio_in_flight()

    def on_conversation_boundary(self) -> None:
        self.s._bump_conversation_id()

    def on_error(self, message: str) -> None:
        self.s._emit({"type": "error", "message": message})

    def on_reset(self) -> None:
        self.s._on_engine_reset()


class ConvFillSession:
    """Per-WebSocket-connection adapter over a `ConvFillEngine`.

    The engine owns inference state and turn execution; this class owns the web
    concerns: the asyncio outbound queue, marshalling worker-thread events onto
    it, per-turn JSONL logging, and translating client commands into engine
    calls. Worker threads run blocking turns; sink callbacks marshal events onto
    the asyncio queue for the WebSocket sender coroutine to drain.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, fragment_counter_start: int = 0):
        self.loop = loop
        self.outbound: asyncio.Queue = asyncio.Queue()
        self._fragment_index = fragment_counter_start
        self._fragment_lock = threading.Lock()
        self.turn_in_progress = False
        self._pending_mode: Optional[str] = None
        # Approximates "utterances queued/playing on the client": incremented
        # when a response_fragment is emitted, decremented on fragment_played
        # ack. Used by the engine to decide whether to keep emitting <sil>.
        self._audio_in_flight = 0
        self._audio_in_flight_lock = threading.Lock()

        # ---- Per-turn JSONL logging state ----
        # turn_id is 0-based within a conversation. _bump_conversation_id()
        # resets it back to -1 alongside each conversation_id bump, and
        # start_new_log() also resets it for a fresh log file.
        self._log_turn_id: int = -1
        self._turn_start_ts: Optional[float] = None
        self._current_user_text: str = ""
        # Accumulating turn-level record. Built by _begin_turn_logging, mutated
        # by _on_response_fragment / on_fragment_played_with_metrics /
        # _on_frontend_inference, flushed to JSONL when the turn fully completes.
        self._turn_record: Optional[dict] = None
        # Maps each row-terminal fragment_index → its slot in the per-phrase
        # lists of _turn_record.
        self._phrase_idx_by_fragment: dict = {}
        # Maps every emitted fragment_index → (phrase_idx, sub_position) so
        # TTS-metric acks can write into the nested per-phrase lists.
        self._subclause_slot: dict = {}
        # Flipped on by _on_turn_complete; consumed by the flush check in
        # on_fragment_played_with_metrics.
        self._turn_complete_pending: bool = False
        self._phrase_lock = threading.Lock()
        # Convfill only: set by _on_phrase_start (frontend thread), consumed by
        # every _on_response_fragment until _on_frontend_inference closes the cluster.
        self._current_thought_text: Optional[str] = None
        self._current_thought_fragments: list = []
        # Non-convfill paths set this (via the sink) before _on_response_fragment.
        self._staged_frontend_ms: Optional[float] = None
        # Per-sub-clause TTS data, keyed by fragment_index.
        self._pending_subclauses: dict = {}
        # Sub-clause fragment indices emitted since the previous terminal in the
        # current thought cluster.
        self._current_subclause_indices: list = []
        # Per-turn "Voiced first response" timing in ms; client sends it on the
        # first fragment_played ack and we stamp every row in the turn with it.
        self._voiced_first_response_ms: Optional[float] = None
        # Per-turn "Frontend response" timing in ms; stamped on the very first
        # fragment of the turn.
        self._frontend_response_ms: Optional[float] = None
        # Per-turn "Backend first response" timing in ms; stamped on the first
        # non-"<sil>" thought received.
        self._backend_first_response_ms: Optional[float] = None
        # Latched once per turn after backend_first_response_ms is written.
        self._backend_first_response_logged: bool = False
        # Conversation id: persists across turns within one WS connection. Bumps
        # on task switch, demo-mode switch, model switch, and Reset.
        self._conversation_id: int = 0

        # Build the engine with a sink that routes its events into WS emits +
        # JSONL logging. The engine owns all inference state from here on.
        self.engine = ConvFillEngine(sink=_WebSink(self))

        # Device placement: the engine computed per-host defaults; apply the
        # transport-side ones (Piper TTS, Whisper STT) to their services. These
        # services live in the web layer, so the engine only records the values.
        from .tts_service import set_device as _tts_set_device
        from .whisper_service import set_device as _whisper_set_device
        _tts_set_device(self.engine.device_settings["tts"])
        _whisper_set_device(self.engine.device_settings["whisper"])

    # ---- read-only delegations to the engine (consumed by main.py + logging) ----

    @property
    def demo_mode(self) -> str:
        return self.engine.demo_mode

    @property
    def active_mode(self) -> str:
        return self.engine.active_mode

    @property
    def available_modes(self) -> list:
        return self.engine.available_modes

    @property
    def active_frontend_model(self) -> Optional[str]:
        return self.engine.active_frontend_model

    @property
    def active_small_model(self) -> Optional[str]:
        return self.engine.active_small_model

    @property
    def active_backend_provider(self) -> str:
        return self.engine.active_backend_provider

    @property
    def active_backend_model(self) -> str:
        return self.engine.active_backend_model

    @property
    def frontend_models(self) -> list:
        return self.engine.frontend_models

    @property
    def backend_models(self) -> dict:
        return self.engine.backend_models

    @property
    def small_models(self) -> list:
        return self.engine.small_models

    @property
    def device_capabilities(self) -> dict:
        return self.engine.device_capabilities

    @property
    def device_settings(self) -> dict:
        return self.engine.device_settings

    @property
    def active_frontend_precision(self) -> str:
        return self.engine.active_frontend_precision

    @property
    def available_frontend_precisions(self) -> list:
        return self.engine.available_frontend_precisions

    @property
    def active_frontend_backend(self) -> str:
        return self.engine.active_frontend_backend

    def frontend_precision_event(self) -> dict:
        # Demo-mode-aware:
        #   * convfill mode honors the convfill config's `backend` field — when
        #     it's mlx the precision is locked to int8 (the convfill checkpoint
        #     was pre-quantized on disk and isn't runtime-swappable).
        #   * frontend_only loads SmallModelInference, which supports both MLX
        #     (int8) and HF (torch dtypes), so we offer the full list.
        if self.demo_mode == "convfill" and self.active_frontend_backend == "mlx":
            return {
                "type": "frontend_precision",
                "backend": "mlx",
                "precision": "int8",
                "available": ["int8"],
            }
        backend = "mlx" if self.active_frontend_precision == "int8" else "hf"
        return {
            "type": "frontend_precision",
            "backend": backend,
            "precision": self.active_frontend_precision,
            "available": self.available_frontend_precisions,
        }

    # ---- thread-safe event emitters (called from worker threads via the sink) ----

    def _emit(self, event: dict) -> None:
        self.loop.call_soon_threadsafe(self.outbound.put_nowait, event)

    def _on_thought(self, thought: str) -> None:
        if thought == "<sil>":
            self._emit({"type": "silence"})
        else:
            with self._phrase_lock:
                if (
                    self._backend_first_response_ms is None
                    and self._turn_start_ts is not None
                ):
                    self._backend_first_response_ms = (
                        time.perf_counter() - self._turn_start_ts
                    ) * 1000.0
            self._emit({"type": "thought", "text": thought})

    def _on_response_fragment(self, text: str) -> None:
        with self._fragment_lock:
            self._fragment_index += 1
            idx = self._fragment_index
        with self._audio_in_flight_lock:
            self._audio_in_flight += 1

        with self._phrase_lock:
            is_convfill = self.demo_mode == "convfill"
            # Stamp "time to first frontend token" on the very first fragment
            # of the turn, regardless of whether it's a sub-clause or a
            # terminal. Matches the client-side metric and ensures it never
            # exceeds voiced_first_response_ms.
            if (
                self._frontend_response_ms is None
                and self._turn_start_ts is not None
            ):
                self._frontend_response_ms = (
                    time.perf_counter() - self._turn_start_ts
                ) * 1000.0
                if self._turn_record is not None:
                    self._turn_record["frontend_response_ms"] = _round_or_none(
                        self._frontend_response_ms
                    )
            # Every emitted fragment is a sub-clause for logging purposes
            # (in non-convfill it's also the terminal, in convfill it's
            # either a clause sub-fragment or the terminal that closes a row).
            self._pending_subclauses[idx] = {
                "text": text,
                "tts_first_byte_ms": None,
                "tts_utterance_ms": None,
            }
            self._current_subclause_indices.append(idx)

            # In convfill we only close a phrase slot on the terminal-punctuation
            # fragment; non-convfill emits whole phrases per fragment so every
            # fragment closes its own slot.
            is_row_terminal = (not is_convfill) or _is_full_phrase(text)
            if is_row_terminal and self._turn_record is not None:
                if is_convfill:
                    # _on_phrase_start has already stashed the current thought
                    # text; a single thought can yield multiple clause fragments
                    # — they all share the same thought_text.
                    # _on_frontend_inference closes the cluster.
                    self._current_thought_fragments.append(idx)
                    frontend_ms: Optional[float] = None  # filled later by _on_frontend_inference
                    thought_text: Optional[str] = self._current_thought_text
                else:
                    # backend_only / frontend_only: caller staged the value.
                    frontend_ms = self._staged_frontend_ms
                    self._staged_frontend_ms = None
                    thought_text = None

                # Snapshot all sub-clause indices that belong to this slot
                # (every fragment since the previous terminal, including
                # this one) and reset for the next slot.
                subclause_indices = self._current_subclause_indices[:]
                self._current_subclause_indices = []

                # backend_first_response_ms is a once-per-turn metric: stamp it
                # only on the first phrase that comes from a real backend
                # thought (skip silence-filler slots and non-convfill slots
                # where there is no backend thought).
                if (
                    not self._backend_first_response_logged
                    and self._backend_first_response_ms is not None
                    and thought_text not in (None, "<sil>")
                ):
                    self._turn_record["backend_first_response_ms"] = _round_or_none(
                        self._backend_first_response_ms
                    )
                    self._backend_first_response_logged = True

                # Build the nested-list slots for TTS metrics, draining any
                # acks that already landed for sub-clauses of this row.
                phrase_idx = len(self._turn_record["thought"])
                tts_texts: list = []
                tts_first_token: list = []
                tts_utterance: list = []
                for sub_position, sub_idx in enumerate(subclause_indices):
                    sub_entry = self._pending_subclauses.pop(sub_idx, None)
                    if sub_entry is None:
                        tts_texts.append(None)
                        tts_first_token.append(None)
                        tts_utterance.append(None)
                    else:
                        tts_texts.append(sub_entry["text"])
                        tts_first_token.append(sub_entry["tts_first_byte_ms"])
                        tts_utterance.append(sub_entry["tts_utterance_ms"])
                    self._subclause_slot[sub_idx] = (phrase_idx, sub_position)

                # frontend_response is the full phrase = all sub-clause texts
                # joined. In convfill the model emits each clause fragment with
                # its trailing punctuation; non-convfill emits one fragment per
                # phrase so this just unwraps it.
                if is_convfill:
                    frontend_response: Optional[str] = " ".join(
                        t for t in tts_texts if t
                    ) or None
                else:
                    frontend_response = None

                # Append a fresh slot to every per-phrase list.
                self._turn_record["thought"].append(thought_text)
                self._turn_record["frontend_response"].append(frontend_response)
                self._turn_record["frontend_inference_ms"].append(_round_or_none(frontend_ms))
                self._turn_record["frontend_first_sentence_within_inference_ms"].append(None)
                self._turn_record["frontend_time_to_first_token"].append(None)
                self._turn_record["tts_phrases"].append(tts_texts)
                self._turn_record["tts_phrase_first_token_ms"].append(tts_first_token)
                self._turn_record["tts_utterance_ms"].append(tts_utterance)

                self._phrase_idx_by_fragment[idx] = phrase_idx

        self._emit({"type": "response_fragment", "text": text, "fragment_index": idx})

    def _get_audio_in_flight(self) -> int:
        with self._audio_in_flight_lock:
            return self._audio_in_flight

    def on_fragment_played(self, _idx: int) -> None:
        with self._audio_in_flight_lock:
            if self._audio_in_flight > 0:
                self._audio_in_flight -= 1

    def on_fragment_played_with_metrics(
        self,
        idx: int,
        tts_first_byte_ms: Optional[float],
        tts_utterance_ms: Optional[float],
        total_response_ms: Optional[float],
        voiced_first_response_ms: Optional[float],
    ) -> None:
        """Client-side ack carrying per-sub-clause TTS metrics. Writes the
        metrics into the right nested-list slot of the in-progress turn
        record, then — once the model has signaled turn_complete AND there are
        no more audio fragments in flight — flushes the turn to JSONL."""
        with self._audio_in_flight_lock:
            if self._audio_in_flight > 0:
                self._audio_in_flight -= 1
            in_flight_after = self._audio_in_flight

        record_to_write: Optional[dict] = None
        with self._phrase_lock:
            tts_first_byte_rounded = _round_or_none(tts_first_byte_ms)
            tts_utterance_rounded = _round_or_none(tts_utterance_ms)

            slot = self._subclause_slot.get(idx)
            if slot is not None and self._turn_record is not None:
                phrase_idx, sub_position = slot
                try:
                    if tts_first_byte_ms is not None:
                        self._turn_record["tts_phrase_first_token_ms"][phrase_idx][sub_position] = tts_first_byte_rounded
                    if tts_utterance_ms is not None:
                        self._turn_record["tts_utterance_ms"][phrase_idx][sub_position] = tts_utterance_rounded
                except (IndexError, KeyError):
                    pass
            else:
                # Ack arrived before the row-terminal allocated a slot; stash
                # in the pending buffer so the terminal handler picks it up.
                sub = self._pending_subclauses.get(idx)
                if sub is not None:
                    if tts_first_byte_ms is not None:
                        sub["tts_first_byte_ms"] = tts_first_byte_rounded
                    if tts_utterance_ms is not None:
                        sub["tts_utterance_ms"] = tts_utterance_rounded

            if voiced_first_response_ms is not None and self._voiced_first_response_ms is None:
                self._voiced_first_response_ms = _round_or_none(voiced_first_response_ms)
                if self._turn_record is not None:
                    self._turn_record["voiced_first_response_ms"] = self._voiced_first_response_ms

            # Row-terminal ack: refresh the turn-total wall-clock. Last
            # terminal wins, which gives the wall-clock from turn start to
            # end of audio playback for the final phrase.
            if idx in self._phrase_idx_by_fragment and self._turn_record is not None:
                if total_response_ms is not None:
                    self._turn_record["total_response_ms"] = _round_or_none(total_response_ms)

            # Flush when turn_complete has fired AND all audio is drained.
            if (
                self._turn_complete_pending
                and in_flight_after == 0
                and self._turn_record is not None
            ):
                self._turn_record["ts"] = datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%S.%fZ"
                )
                record_to_write = self._turn_record
                self._turn_record = None
                self._phrase_idx_by_fragment = {}
                self._subclause_slot = {}
                self._pending_subclauses.clear()
                self._turn_complete_pending = False

        if record_to_write is not None:
            try:
                log_writer.append(record_to_write)
            except Exception as exc:
                print(f"[log_writer] append failed: {exc!r}", flush=True)

    def _on_rag_context(self, text: str) -> None:
        self._emit({"type": "rag_context", "text": text})

    def _on_mcp_context(self, text: str) -> None:
        self._emit({"type": "mcp_context", "text": text})

    def _on_frontend_inference(
        self,
        ms: float,
        tokens: Optional[int] = None,
        first_sentence_ms: Optional[float] = None,
        first_token_ms: Optional[float] = None,
    ) -> None:
        # Convfill only: backfill frontend_inference_ms, first-sentence-within-
        # inference timing, and true first-token timing onto all phrase slots
        # of the just-completed thought, then close the cluster so the next
        # response_fragment starts fresh. (For the standalone paths the engine
        # calls this once at end-of-turn with an empty cluster — the loop is a
        # no-op and only the frontend_timing emit below matters.)
        with self._phrase_lock:
            ms_rounded = _round_or_none(ms)
            first_sentence_rounded = _round_or_none(first_sentence_ms)
            first_token_rounded = _round_or_none(first_token_ms)
            if self._turn_record is not None:
                for frag_idx in self._current_thought_fragments:
                    phrase_idx = self._phrase_idx_by_fragment.get(frag_idx)
                    if phrase_idx is None:
                        continue
                    try:
                        self._turn_record["frontend_inference_ms"][phrase_idx] = ms_rounded
                        self._turn_record["frontend_first_sentence_within_inference_ms"][phrase_idx] = first_sentence_rounded
                        self._turn_record["frontend_time_to_first_token"][phrase_idx] = first_token_rounded
                    except IndexError:
                        pass
            self._current_thought_fragments = []
            self._current_thought_text = None
        self._emit({"type": "frontend_timing", "ms": ms, "tokens": tokens})

    def _on_phrase_start(self, thought: str, backend_phrase_ms: float) -> None:
        """Frontend reports the thought it's about to infer on. The second
        argument (wall-clock gap) is accepted for callsite compatibility but no
        longer used now that backend timing is captured globally via _on_thought."""
        del backend_phrase_ms
        with self._phrase_lock:
            self._current_thought_text = thought

    def _on_turn_complete(self, final_response: str, final_thoughts: list) -> None:
        self._emit({
            "type": "turn_complete",
            "final_response": final_response,
            "thoughts": final_thoughts,
        })
        # Mark the turn ready to flush. If all TTS audio has already drained,
        # flush right now; otherwise the final on_fragment_played_with_metrics
        # call will pick it up. The in_flight read happens INSIDE phrase_lock
        # so that an ack thread can't race past us.
        record_to_write: Optional[dict] = None
        with self._phrase_lock:
            self._turn_complete_pending = True
            with self._audio_in_flight_lock:
                in_flight = self._audio_in_flight
            if in_flight == 0 and self._turn_record is not None:
                self._turn_record["ts"] = datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%S.%fZ"
                )
                record_to_write = self._turn_record
                self._turn_record = None
                self._phrase_idx_by_fragment = {}
                self._subclause_slot = {}
                self._pending_subclauses.clear()
                self._turn_complete_pending = False

        if record_to_write is not None:
            try:
                log_writer.append(record_to_write)
            except Exception as exc:
                print(f"[log_writer] append failed: {exc!r}", flush=True)

    def _bump_conversation_id(self) -> None:
        with self._phrase_lock:
            self._conversation_id += 1
            self._log_turn_id = -1

    def _on_engine_reset(self) -> None:
        """Clear the web-side per-turn state when the engine resets inference
        state. Mirrors what the old _reset_internal cleared beyond the engine's
        own queues/DSMs."""
        with self._audio_in_flight_lock:
            self._audio_in_flight = 0
        with self._phrase_lock:
            self._turn_record = None
            self._phrase_idx_by_fragment = {}
            self._subclause_slot = {}
            self._turn_complete_pending = False
            self._pending_subclauses.clear()
            self._current_subclause_indices = []
            self._current_thought_fragments = []
        self.turn_in_progress = False
        self._pending_mode = None

    def _begin_turn_logging(self, text: str) -> None:
        """Resets per-turn state and stamps turn_start. Call from worker
        thread or coroutine before producing any phrases."""
        with self._phrase_lock:
            self._log_turn_id += 1
            self._current_user_text = text
            self._turn_start_ts = time.perf_counter()
            self._pending_subclauses.clear()
            self._current_subclause_indices = []
            self._voiced_first_response_ms = None
            self._frontend_response_ms = None
            self._backend_first_response_ms = None
            self._backend_first_response_logged = False
            self._current_thought_fragments = []
            self._current_thought_text = None
            self._staged_frontend_ms = None
            self._phrase_idx_by_fragment = {}
            self._subclause_slot = {}
            self._turn_complete_pending = False
            self._turn_record = {
                "ts": None,
                "turn_id": self._log_turn_id,
                "conversation_id": self._conversation_id,
                "demo_mode": self.demo_mode,
                "task": self.active_mode,
                "frontend_model": (
                    self.active_frontend_model
                    if self.demo_mode == "convfill"
                    else (self.active_small_model if self.demo_mode == "frontend_only" else None)
                ),
                "backend_model": (
                    f"{self.active_backend_provider}/{self.active_backend_model}"
                    if self.demo_mode in ("convfill", "backend_only")
                    and self.active_backend_provider and self.active_backend_model
                    else None
                ),
                "user_utterance": text,
                "frontend_response_ms": None,
                "backend_first_response_ms": None,
                "voiced_first_response_ms": None,
                "total_response_ms": None,
                "thought": [],
                "frontend_response": [],
                "frontend_inference_ms": [],
                "frontend_first_sentence_within_inference_ms": [],
                "frontend_time_to_first_token": [],
                "tts_phrases": [],
                "tts_phrase_first_token_ms": [],
                "tts_utterance_ms": [],
            }

    # ---- public coroutines (called from the WS handler) ----

    async def handle_user_message(self, text: str) -> None:
        if self.turn_in_progress:
            self._emit({"type": "error", "message": "A turn is already in progress."})
            return

        err = self.engine.turn_precheck()
        if err:
            self._emit({"type": "error", "message": err})
            return

        self.turn_in_progress = True
        self._begin_turn_logging(text)
        await self.outbound.put({"type": "turn_start"})
        try:
            await self.loop.run_in_executor(None, self.engine.run_turn, text)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._emit({"type": "error", "message": f"Turn failed: {exc!r}"})
        finally:
            self.turn_in_progress = False
            if self._pending_mode is not None and self._pending_mode != self.engine.active_mode:
                self.engine.active_mode = self._pending_mode
                self._emit({"type": "mode_changed", "mode": self.engine.active_mode})
            self._pending_mode = None

    async def set_mode(self, mode: str) -> None:
        if self.turn_in_progress:
            self._pending_mode = mode
            self._emit({"type": "error", "message": "Mode change queued until current turn completes."})
            return
        try:
            await self.loop.run_in_executor(None, self.engine.set_mode, mode)
        except EngineError as exc:
            self._emit({"type": "error", "message": str(exc)})
            return
        self._emit({"type": "mode_changed", "mode": mode})

    async def set_demo_mode(self, demo_mode: str) -> None:
        if demo_mode not in ("convfill", "frontend_only", "backend_only"):
            self._emit({"type": "error", "message": f"Unknown demo mode: {demo_mode}"})
            return
        if demo_mode == self.demo_mode:
            self._emit({"type": "demo_mode_changed", "demo_mode": demo_mode})
            return
        if self.turn_in_progress:
            self._emit({"type": "error", "message": "Demo-mode change blocked: a turn is in progress."})
            return

        prev_mode = self.engine.active_mode
        try:
            await self.loop.run_in_executor(None, self.engine.set_demo_mode, demo_mode)
        except EngineError as exc:
            self._emit({"type": "error", "message": str(exc)})
            return

        # The engine may have bumped active_mode back to "normal" (leaving mcp
        # for frontend_only); reflect that to the client.
        if self.engine.active_mode != prev_mode:
            self._emit({"type": "mode_changed", "mode": self.engine.active_mode})

        self._emit({"type": "demo_mode_changed", "demo_mode": demo_mode})
        # convfill (potentially MLX/int8) and frontend_only (always HF) report
        # different backends; re-emit so the precision UI reflects the new mode.
        self._emit(self.frontend_precision_event())

    async def set_small_model(self, name: str) -> None:
        if name not in self.small_models:
            self._emit({"type": "error", "message": f"Unknown small model: {name}"})
            return
        if name == self.active_small_model:
            self._emit({"type": "small_model_changed", "name": name})
            return
        try:
            await self.loop.run_in_executor(None, self.engine.set_small_model, name)
        except EngineError as exc:
            self._emit({"type": "error", "message": str(exc)})
            return
        self._emit({"type": "small_model_changed", "name": name})

    async def reset(self) -> None:
        await self.loop.run_in_executor(None, self.engine.reset)
        self._bump_conversation_id()
        self._emit({"type": "reset_ack"})

    async def start_new_log(self) -> None:
        """Rotate to a fresh JSONL file and restart turn numbering at 0."""
        if self.turn_in_progress:
            self._emit({"type": "error", "message": "Cannot start a new log while a turn is in progress."})
            return
        path = await self.loop.run_in_executor(None, log_writer.start_new_log)
        with self._phrase_lock:
            self._log_turn_id = -1
            self._turn_record = None
            self._phrase_idx_by_fragment = {}
            self._subclause_slot = {}
            self._turn_complete_pending = False
            self._pending_subclauses.clear()
            self._current_subclause_indices = []
            self._current_thought_fragments = []
        self._emit({"type": "log_started", "path": path})

    async def set_device(self, component: str, device: str) -> None:
        if component not in self.device_capabilities:
            self._emit({"type": "error", "message": f"Unknown device component: {component}"})
            return
        allowed = self.device_capabilities[component]
        if device not in allowed:
            self._emit({"type": "error", "message": f"Device {device} not available for {component} (allowed: {allowed})"})
            return
        if self.turn_in_progress:
            self._emit({"type": "error", "message": "Device change blocked: a turn is in progress."})
            return
        if self.device_settings[component] == device:
            self._emit({"type": "device_changed", "component": component, "device": device})
            return

        try:
            await self.loop.run_in_executor(None, self.engine.set_device, component, device)
        except EngineError as exc:
            self._emit({"type": "error", "message": str(exc)})
            return

        # TTS/Whisper run in the web layer, not the engine; drive their services
        # here after the engine has recorded the new device setting.
        if component == "tts":
            from .tts_service import set_device as _tts_set_device, warmup as _tts_warmup
            _tts_set_device(device)
            # Re-warm the new device on a worker thread so the first user TTS
            # request doesn't pay session-creation latency.
            await self.loop.run_in_executor(None, _tts_warmup)
        elif component == "whisper":
            from .whisper_service import set_device as _whisper_set_device
            _whisper_set_device(device)
            # No warmup: first transcribe is user-driven and already async.

        self._emit({"type": "device_changed", "component": component, "device": device})

    async def set_frontend_model(self, name: str) -> None:
        if name not in self.frontend_models:
            self._emit({"type": "error", "message": f"Unknown frontend model: {name}"})
            return
        if name == self.active_frontend_model:
            self._emit({"type": "frontend_model_changed", "name": name})
            return
        try:
            await self.loop.run_in_executor(None, self.engine.set_frontend_model, name)
        except EngineError as exc:
            self._emit({"type": "error", "message": str(exc)})
            return
        self._emit({"type": "frontend_model_changed", "name": name})
        # The new model may use a different backend (mlx vs hf), which changes
        # what the precision UI should show. Re-emit so the client refreshes.
        self._emit(self.frontend_precision_event())

    async def set_backend_model(self, provider: str, name: str) -> None:
        if provider == self.active_backend_provider and name == self.active_backend_model:
            self._emit({"type": "backend_model_changed", "provider": provider, "name": name})
            return
        try:
            await self.loop.run_in_executor(None, self.engine.set_backend_model, provider, name)
        except EngineError as exc:
            self._emit({"type": "error", "message": str(exc)})
            return
        self._emit({"type": "backend_model_changed", "provider": provider, "name": name})

    async def set_precision(self, precision: str) -> None:
        if self.turn_in_progress:
            self._emit({"type": "error", "message": "Precision change blocked: a turn is in progress."})
            return
        if precision == self.active_frontend_precision:
            self._emit({"type": "precision_changed", "precision": precision})
            return
        try:
            await self.loop.run_in_executor(None, self.engine.set_precision, precision)
        except EngineError as exc:
            self._emit({"type": "error", "message": str(exc)})
            return
        self._emit({"type": "precision_changed", "precision": precision})


def _round_or_none(x: Optional[float]) -> Optional[int]:
    return None if x is None else int(round(x))


# Match strings ending in a *terminal* punctuation mark, optionally followed by
# closing quote / paren / bracket and trailing whitespace. Mirrors
# _SENTENCE_SPLIT_RE in src/inference/convfill_stack/convfill_frontend.py (the producer) and
# splitSentences() in App.tsx (the client bubble splitter) so server logs,
# client display, and frontend emission all agree on sentence boundaries.
_TERMINAL_END_RE = re.compile(r"[.!?][\"')\]]*\s*$")


def _is_full_phrase(text: str) -> bool:
    return bool(_TERMINAL_END_RE.search(text))
