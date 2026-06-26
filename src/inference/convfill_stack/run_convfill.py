# IMPORTANT: Install the package first with `pip install -e .` from the repo root

import os
import sys
import threading
import queue
import time

from src.inference.shared.dialogue_state_manager import DialogueStateManager
from src.inference.convfill_stack.convfill_backend_multi import ConvFillBackend
from src.inference.convfill_stack.convfill_frontend import ConvFillFrontend
from src.inference.convfill_stack.model_inference_functions_conversational import BackendInference
import json
from strip_markdown import strip_markdown
from src.utils.api_keys import get_api_key

class ConvFillConfig:
    def __init__(self, config_path, mode=None):
        with open(config_path, "r") as f:
            config = json.load(f)

        if "modes" in config:
            mode_config = config["modes"][mode]
            config = {**config, **mode_config, "mode": mode}

        self.frontend_model_config_path = config["frontend_model_config_path"]
        self.backend_prompt_template_file = config["backend_prompt_template_file"]
        self.backend_model_name = config["backend_model_name"]
        self.backend_model_mode = config["backend_model_mode"]
        self.num_history_turns = config["num_history_turns"]
        self.tts_mode = config["tts_mode"]
        self.tts_model_path = config.get("tts_model_path")
        self.task_specific_config = config.get("task_specific_config", None)
        self.mode = config["mode"]
        self.frontend_device = config.get("frontend_device", "cpu")
        self.frontend_dtype = config.get("frontend_dtype")

class ConvFillSystem:
    def __init__(self, convfill_config, dialogue_state_manager=None,
                 on_thought=None, on_response_fragment=None,
                 on_rag_context=None, on_mcp_context=None,
                 on_turn_complete=None,
                 on_frontend_inference=None,
                 on_phrase_start=None,
                 emit_tts=True,
                 audio_in_flight_fn=None):
        self.convfill_config = convfill_config

        self.thought_queue = queue.Queue()
        self.done_event = threading.Event()
        self.tts_queue = queue.Queue()
        self.tts_model_path = convfill_config.tts_model_path if hasattr(convfill_config, 'tts_model_path') else None

        # callbacks for web
        self.on_thought = on_thought
        self.on_response_fragment = on_response_fragment
        self.on_turn_complete = on_turn_complete
        self.on_frontend_inference = on_frontend_inference
        self.on_phrase_start = on_phrase_start
        self.emit_tts = emit_tts
        # count of utterances that have been emitted but not yet finished playing
        self.audio_in_flight_fn = audio_in_flight_fn if audio_in_flight_fn is not None \
            else (lambda: self.tts_queue.unfinished_tasks)

        # bridges the gap between pulling a thought off thought_queue and the corresponding audio fragment showing up in audio_in_flight_fn() so backend_fn can't see all-zeros while frontend inference is in flight
        self._frontend_inflight = 0
        self._frontend_inflight_lock = threading.Lock()

        # setup frontend, backend, dialogue state
        print("[LOG] Setting up Dialogue State Manager\n")
        if dialogue_state_manager is not None:
            self.dialogue_state_manager = dialogue_state_manager
        else:
            self.dialogue_state_manager = DialogueStateManager(num_history_turns=self.convfill_config.num_history_turns)

        print("[LOG] Setting up ConvFill Backend\n")
        self.api_key = get_api_key(self.convfill_config.backend_model_mode)
        self.convfill_backend = ConvFillBackend(
            dialogue_state_manager=self.dialogue_state_manager,
            prompt_template_path=self.convfill_config.backend_prompt_template_file,
            model_backend=BackendInference(
                api_key=self.api_key,
                model_name=self.convfill_config.backend_model_name,
                model_mode=self.convfill_config.backend_model_mode
            ),
            mode = self.convfill_config.mode,
            task_specific_config=self.convfill_config.task_specific_config,
            on_rag_context=on_rag_context,
            on_mcp_context=on_mcp_context,
        )

        print("[LOG] Setting up ConvFill Frontend\n")
        self.convfill_frontend = ConvFillFrontend(
            model_config_path=self.convfill_config.frontend_model_config_path,
            dialogue_state_manager=self.dialogue_state_manager,
            device=self.convfill_config.frontend_device,
            dtype=self.convfill_config.frontend_dtype)

    def backend_fn(self):
        q = queue.Queue()

        def producer():
            try:
                for chunk in self.convfill_backend.backend_infer():
                    q.put(chunk)
            finally:
                q.put(None)

        threading.Thread(target=producer, daemon=True).start()

        while True:
            try:
                chunk = q.get_nowait()
                if chunk is None:
                    return
                yield chunk
                continue
            except queue.Empty:
                pass

            # emit filler only when the listener would otherwise hear silence: no thought queued, no frontend inference in flight, and no audio currently playing.
            with self._frontend_inflight_lock:
                frontend_busy = self._frontend_inflight > 0
            if (self.thought_queue.qsize() == 0
                    and not frontend_busy
                    and self.audio_in_flight_fn() == 0):
                yield "<sil>"
                continue

            try:
                chunk = q.get(timeout=0.1)
                if chunk is None:
                    return
                yield chunk
            except queue.Empty:
                continue
        

    def run_backend(self):
        for thought in self.backend_fn():
            self.thought_queue.put(thought)
            if self.on_thought is not None:
                self.on_thought(thought)

        self.done_event.set()


    def run_frontend(self):
        t_prev_frontend_end = time.perf_counter()
        while not self.done_event.is_set() or not self.thought_queue.empty():
            try:
                thought = self.thought_queue.get(timeout=1)
                if thought == "<sil>":
                    with self.thought_queue.mutex:
                        has_real_pending = any(t != "<sil>" for t in self.thought_queue.queue)
                    if has_real_pending:
                        self.thought_queue.task_done()
                        continue

                phrase_start_ts = time.perf_counter()
                backend_gap_ms = (phrase_start_ts - t_prev_frontend_end) * 1000.0
                if self.on_phrase_start is not None:
                    self.on_phrase_start(thought, backend_gap_ms)
                with self._frontend_inflight_lock:
                    self._frontend_inflight += 1
                try:
                    for fragment in self.convfill_frontend.infer_and_update_state(thought):
                        if self.emit_tts:
                            cleaned = strip_markdown(fragment).strip()
                            if cleaned:
                                self.tts_queue.put(cleaned)
                        if self.on_response_fragment is not None:
                            self.on_response_fragment(fragment)
                    if self.on_frontend_inference is not None:
                        gen_ms = self.convfill_frontend.last_generate_ms
                        gen_tokens = self.convfill_frontend.last_generated_tokens
                        first_sentence_ms = self.convfill_frontend.last_first_sentence_ms
                        first_token_ms = self.convfill_frontend.last_first_token_ms
                        if gen_ms is not None:
                            self.on_frontend_inference(gen_ms, gen_tokens, first_sentence_ms, first_token_ms)
                finally:
                    with self._frontend_inflight_lock:
                        self._frontend_inflight -= 1
                    t_prev_frontend_end = time.perf_counter()
                self.thought_queue.task_done()
            except queue.Empty:
                continue
        user_turn, final_response, final_thoughts = self.convfill_frontend.get_final_response_and_reset_turn_state()
        if self.on_turn_complete is not None:
            self.on_turn_complete(final_response, final_thoughts)

    def run_turn(self, user_input):
        """Run a single turn programmatically (no stdin/stdout, no TTS thread)."""
        self.dialogue_state_manager.update_user_state(user_input)
        self.done_event.clear()

        backend_thread = threading.Thread(target=self.run_backend)
        backend_thread.start()

        frontend_thread = threading.Thread(target=self.run_frontend)
        frontend_thread.start()

        backend_thread.join()
        frontend_thread.join()
