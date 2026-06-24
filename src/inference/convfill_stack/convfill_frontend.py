from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import json
import logging
import threading
import time
from pathlib import Path
from src.inference.shared.turn_state_manager import TurnStateManager
from src.inference.shared.dialogue_state_manager import DialogueStateManager
import re
import mlx_lm

_SENTENCE_SPLIT_RE = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+")


def _flush_sentences(buf: str) -> tuple[list[str], str]:
    parts = _SENTENCE_SPLIT_RE.split(buf)
    if len(parts) == 1:
        return [], buf
    return [p for p in parts[:-1] if p.strip()], parts[-1]

# remove faulty Gemma error
logging.getLogger("transformers.utils.loading_report").setLevel(logging.ERROR)

# proceess-wide cache
_MODEL_CACHE: dict = {}
_MODEL_CACHE_LOCK = threading.Lock()

_DTYPE_NAMES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _resolve_dtype(device: str):
    # default to bfloat16 on accelerators
    if device in ("cuda", "mps"):
        return torch.bfloat16
    return torch.float32


class ConvFillFrontend:
    def __init__(self, model_config_path, checkpoint_dir, dialogue_state_manager, device: str = "cpu", dtype: str | None = None):
        self.model_config_path = model_config_path
        self.checkpoint_dir = checkpoint_dir
        self.run_name = self.extract_run_name()
        self.device = device
        self.dtype = dtype

        with open(model_config_path, "r") as f:
            model_config = json.load(f)

        self.model_config = model_config

        self.boundary_tokens = self.model_config["boundary_tokens"]
        self.special_tokens = self.model_config["special_tokens"]
        self.add_special_tokens = self.model_config["add_special_tokens"]
        self.model_name = self.model_config["model_name"]

        # backend defaults to mlx
        self.backend = self.model_config.get("backend", "mlx")
        self.checkpoint_suffix = self.model_config.get("checkpoint_suffix", "_mlx_q8")

        self.model, self.tokenizer = self.load_model()
        self.turn_state_manager = TurnStateManager(boundary_tokens=self.boundary_tokens)
        self.dialogue_state_manager = dialogue_state_manager
        self.last_generate_ms: float | None = None
        self.last_generated_tokens: int | None = None
        self.last_first_token_ms: float | None = None
        self.last_first_sentence_ms: float | None = None

    def get_boundary_tokens(self):
        return self.boundary_tokens

    def extract_run_name(self):
        path = Path(self.model_config_path)
        name = path.stem
        prefix = "convfill_"
        if name.startswith(prefix):
            extracted = name[len(prefix):]
            print("[MODEL SETUP] EXTRACTED RUN NAME:", extracted, flush=True)
            return extracted
            print("[MODEL SETUP] WARNING: RUN NAME DOES NOT START WITH EXPECTED PREFIX:", prefix, flush=True)
        return name

    def load_model(self):
        suffix = self.checkpoint_suffix if self.backend == "mlx" else ""
        model_checkpoint_path = f"{self.checkpoint_dir}/{self.run_name}{suffix}"

        config_dtype = self.model_config.get("torch_dtype")
        if self.dtype is not None:
            resolved_dtype = _DTYPE_NAMES[self.dtype]
        elif config_dtype is not None:
            resolved_dtype = _DTYPE_NAMES[config_dtype]
        else:
            resolved_dtype = _resolve_dtype(self.device)

        cache_key = (
            model_checkpoint_path,
            self.backend,
            bool(self.add_special_tokens),
            json.dumps(self.special_tokens, sort_keys=True) if self.special_tokens else "",
            self.device,
            str(resolved_dtype),
        )
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(cache_key)
            if cached is not None:
                model, tokenizer = cached
                print("[MODEL SETUP] REUSING CACHED MODEL", self.model_name, flush=True)
                return model, tokenizer

            if self.backend == "mlx":
                model, tokenizer = self._load_mlx(model_checkpoint_path)
            else:
                model, tokenizer = self._load_hf(model_checkpoint_path, resolved_dtype)

            _MODEL_CACHE[cache_key] = (model, tokenizer)
            return model, tokenizer

    def _load_hf(self, model_checkpoint_path, resolved_dtype):
        model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint_path,
            torch_dtype=resolved_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path)
        print("[MODEL SETUP] LOADED HF MODEL", self.model_name, "dtype:", resolved_dtype, flush=True)
        print("[MODEL SETUP] LOADED WEIGHTS FROM", model_checkpoint_path, flush=True)

        current_vocab = model.get_input_embeddings().weight.shape[0]

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
        print("[MODEL SETUP] PAD TOKEN ID:", tokenizer.pad_token_id, flush=True)

        if self.add_special_tokens:
            tokenizer.add_special_tokens(self.special_tokens)
        print("[MODEL SETUP] ADDED SPECIAL TOKENS:", self.special_tokens, flush=True)

        if len(tokenizer) != current_vocab:
            model.resize_token_embeddings(len(tokenizer))
            model.tie_weights()

        model.to(self.device)
        model.tie_weights()
        model.eval()
        print(f"[MODEL SETUP] MOVED MODEL TO DEVICE: {self.device}", flush=True)

        # warmup
        with torch.no_grad():
            warmup_ids = tokenizer("hello", return_tensors="pt")["input_ids"].to(self.device)
            model.generate(
                input_ids=warmup_ids,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        print("[MODEL SETUP] WARMED UP", self.model_name, flush=True)
        return model, tokenizer

    def _load_mlx(self, model_checkpoint_path):
        import mlx_lm
        model, tokenizer = mlx_lm.load(model_checkpoint_path)
        print("[MODEL SETUP] LOADED MLX MODEL", self.model_name, flush=True)
        print("[MODEL SETUP] LOADED WEIGHTS FROM", model_checkpoint_path, flush=True)

        if self.add_special_tokens:
            for tok in self.special_tokens.get("additional_special_tokens", []):
                ids = tokenizer.encode(tok, add_special_tokens=False)
                assert len(ids) == 1, (
                    f"Special token {tok!r} did not survive MLX conversion as a "
                    f"single token id (got {ids}). Re-run convert_to_mlx_q8 or "
                    f"check the source checkpoint's tokenizer."
                )

        # register the END boundary as an extra stop so generation halts
        end_tok = self.boundary_tokens["END"].strip()
        try:
            tokenizer.add_eos_token(end_tok)
        except Exception as e:
            print(f"[MODEL SETUP] WARNING: could not register extra EOS {end_tok!r}: {e}", flush=True)

        # warmup
        _ = mlx_lm.generate(model, tokenizer, prompt="hello", max_tokens=4, verbose=False)
        print("[MODEL SETUP] WARMED UP", self.model_name, flush=True)
        return model, tokenizer

    def reset_turn_state(self):
        self.turn_state_manager.reset()

    def _iter_hf_chunks(self, prompt):
        encoded = self.tokenizer(
            prompt, return_tensors="pt", padding=False, truncation=False
        )
        input_ids = encoded["input_ids"].to(self.device)
        pad_id = self.tokenizer.pad_token_id

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=False,
        )
        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=256,
            do_sample=False,
            top_p=None,
            top_k=None,
            pad_token_id=pad_id,
            streamer=streamer,
        )

        gen_error: list[BaseException] = []

        def _run_generate():
            try:
                with torch.no_grad():
                    self.model.generate(**gen_kwargs)
            except BaseException as exc:
                gen_error.append(exc)
            finally:
                try:
                    streamer.end()
                except Exception:
                    pass

        thread = threading.Thread(target=_run_generate, daemon=True)
        thread.start()

        try:
            count = 0
            for chunk in streamer:
                if chunk:
                    count += 1
                yield chunk, count
        finally:
            thread.join()
            if gen_error:
                raise gen_error[0]

    def _iter_mlx_chunks(self, prompt):
        n = 0
        for resp in mlx_lm.stream_generate(
            self.model, self.tokenizer, prompt=prompt, max_tokens=256
        ):
            n = resp.generation_tokens
            yield resp.text, n

    def run_convfill_inference(self, prompt):
        end_tok = self.boundary_tokens["END"].strip()

        if self.backend == "mlx":
            chunk_iter = self._iter_mlx_chunks(prompt)
        else:
            chunk_iter = self._iter_hf_chunks(prompt)

        t0 = time.perf_counter()
        self.last_first_token_ms = None
        self.last_first_sentence_ms = None

        buf = ""
        last_token_count = 0
        hit_end = False
        try:
            for chunk, token_count in chunk_iter:
                if not chunk:
                    continue
                last_token_count = token_count
                if self.last_first_token_ms is None:
                    self.last_first_token_ms = (time.perf_counter() - t0) * 1000.0
                buf += chunk

                if end_tok and end_tok in buf:
                    buf = buf.split(end_tok, 1)[0]
                    hit_end = True

                sentences, buf = _flush_sentences(buf)
                for s in sentences:
                    s = s.lstrip()
                    if s:
                        if self.last_first_sentence_ms is None:
                            self.last_first_sentence_ms = (time.perf_counter() - t0) * 1000.0
                        yield s

                if hit_end:
                    break
        finally:
            chunk_iter.close()

        self.last_generate_ms = (time.perf_counter() - t0) * 1000.0
        self.last_generated_tokens = last_token_count

        tail = buf.strip()
        if tail:
            if self.last_first_sentence_ms is None:
                self.last_first_sentence_ms = (time.perf_counter() - t0) * 1000.0
            yield tail

    def infer(self, current_user_input, current_thought, history=None):
        assert history is not None, "History must be provided for inference."
        print("\n[Inferring with thought]\n" + current_thought + "\n", flush=True)
        prompt = self.turn_state_manager.build_prompt(
            current_user_input=current_user_input,
            current_thought=current_thought,
            history=history,
        )
        yield from self.run_convfill_inference(prompt)


    def infer_and_update_state(self, current_thought):
        history = self.dialogue_state_manager.get_history()
        user_in = self.dialogue_state_manager.user_turns[-1]
        pieces: list[str] = []
        for fragment in self.infer(
            current_user_input=user_in,
            current_thought=current_thought,
            history=history,
        ):
            pieces.append(fragment)
            yield fragment
        full_response = " ".join(pieces).strip()
        if self.turn_state_manager.user_turn_empty():
            self.turn_state_manager.add_user_turn(user_in)
        self.turn_state_manager.add_response(full_response)
        self.turn_state_manager.add_thought(current_thought)

    def get_final_response_and_reset_turn_state(self):
        user_turn, final_response, final_thoughts = self.turn_state_manager.get_completed_response()
        self.dialogue_state_manager.update_response_state(final_response)
        self.dialogue_state_manager.update_thoughts_state(self.turn_state_manager.thoughts)
        self.reset_turn_state()
        return user_turn, final_response, final_thoughts
