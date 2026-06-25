import logging
import re
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


logging.getLogger("transformers.utils.loading_report").setLevel(logging.ERROR)


_MODEL_CACHE: dict = {}
_MODEL_CACHE_LOCK = threading.Lock()


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# inference_params keys that belong on tokenizer.apply_chat_template
# (e.g. Qwen3's `enable_thinking`) rather than model.generate.
_CHAT_TEMPLATE_KEYS = {"enable_thinking"}

# Forces small chat models to emit spoken-style prose. The downstream pipeline
# splits output on sentence punctuation and ships each chunk to TTS, so bullets
# / markdown / numbered lists would land as garbled audio.
SYSTEM_PROMPT = (
    "You are a conversational assistant whose responses will be read aloud by "
    "a text-to-speech system. Respond only in natural spoken English. Do not "
    "use bullet points, numbered lists, dashes, asterisks, markdown formatting, "
    "headers, code blocks, or emoji. Respond as if you are speaking out loud in "
    "complete sentences."
)


_DTYPE_NAMES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _resolve_dtype(device: str):
    # BF16 default on GPU: FP16 overflows on Gemma 270M activations on MPS and
    # collapses output to <pad>. BF16 keeps FP32's exponent range so it works
    # across Gemma and Qwen alike.
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.bfloat16
    return torch.float32


def _resolve_torch_dtype(device: str, dtype: Optional[str]):
    if dtype is not None:
        return _DTYPE_NAMES[dtype]
    return _resolve_dtype(device)


def _load(
    model_name: str,
    device: str,
    dtype: Optional[str] = None,
    backend: str = "hf",
):
    if backend == "mlx":
        # MLX manages dtype + device internally (Metal + quantized weights);
        # the cache key just needs the model_name.
        cache_key = ("mlx", model_name)
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(cache_key)
            if cached is not None:
                print(f"[small_model] REUSING CACHED MLX MODEL {model_name}", flush=True)
                return cached
            model, tokenizer = _load_mlx(model_name, model_name)
            _MODEL_CACHE[cache_key] = (model, tokenizer)
            return model, tokenizer

    resolved_dtype = _resolve_torch_dtype(device, dtype)
    cache_key = ("hf", model_name, device, str(resolved_dtype))
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            print(f"[small_model] REUSING CACHED MODEL {model_name} on {device}", flush=True)
            return cached

        print(f"[small_model] LOADING {model_name} from HF Hub…", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=resolved_dtype,
        )
        model.to(device)
        model.eval()
        print(f"[small_model] LOADED {model_name} on {device} dtype={resolved_dtype}", flush=True)

        with torch.no_grad():
            warmup_ids = tokenizer("hello", return_tensors="pt")["input_ids"].to(device)
            model.generate(
                input_ids=warmup_ids,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        print(f"[small_model] WARMED UP {model_name}", flush=True)

        _MODEL_CACHE[cache_key] = (model, tokenizer)
        return model, tokenizer


def _load_mlx(model_name: str, ckpt_path: str):
    import mlx_lm

    print(f"[small_model] LOADING MLX MODEL {model_name} from {ckpt_path}…", flush=True)
    model, tokenizer = mlx_lm.load(ckpt_path)
    print(f"[small_model] LOADED MLX MODEL {model_name}", flush=True)

    # First MLX call compiles the Metal kernels for prefill + decode; burn one
    # so the cached model serves the first user turn at steady-state latency.
    _ = mlx_lm.generate(model, tokenizer, prompt="hello", max_tokens=4, verbose=False)
    print(f"[small_model] WARMED UP {model_name}", flush=True)
    return model, tokenizer


def _format_prompt(
    tokenizer, messages: list[dict], template_kwargs: Optional[dict] = None
) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **(template_kwargs or {}),
        )
    lines = []
    for m in messages:
        role = m.get("role", "user").capitalize()
        lines.append(f"{role}: {m.get('content', '')}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _flush_sentences(buf: str) -> tuple[list[str], str]:
    """Pull complete sentences off the front of `buf`; return (sentences, remainder)."""
    parts = _SENTENCE_SPLIT_RE.split(buf)
    if len(parts) == 1:
        return [], buf
    return [p for p in parts[:-1] if p.strip()], parts[-1]


class SmallModelInference:
    """HuggingFace small-model chat inference loaded directly from the Hub.

    One instance per (model_name, device). Heavy load work happens lazily in
    `__init__`; callers must construct this on a worker thread, never on the
    asyncio event loop.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_new_tokens: int = 256,
        inference_params: Optional[dict] = None,
        dtype: Optional[str] = None,
        backend: str = "hf",
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.inference_params: dict = dict(inference_params or {})
        self.backend = backend
        self.model, self.tokenizer = _load(
            model_name, device, dtype, backend=backend
        )
        self.last_generate_ms: Optional[float] = None
        self.last_generated_tokens: Optional[int] = None

    def generate_chat(
        self,
        messages: list[dict],
        rag_context: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        chat_messages = [dict(m) for m in messages]
        if rag_context:
            for m in reversed(chat_messages):
                if m.get("role") == "user":
                    m["content"] = (
                        "Use the following context to answer the question.\n\n"
                        f"Context:\n{rag_context}\n\n"
                        f"Question: {m.get('content', '')}"
                    )
                    break

        print("CHAT MESSAGES:", chat_messages, flush=True)
        full_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *chat_messages,
        ]

        template_kwargs = {
            k: v for k, v in self.inference_params.items() if k in _CHAT_TEMPLATE_KEYS
        }
        extra_generate_kwargs = {
            k: v for k, v in self.inference_params.items() if k not in _CHAT_TEMPLATE_KEYS
        }

        prompt = _format_prompt(self.tokenizer, full_messages, template_kwargs)
        max_tokens = max_new_tokens or self.max_new_tokens

        t0 = time.perf_counter()
        buf = ""
        token_count = 0
        diagnose_ctx: Optional[tuple] = None

        if self.backend == "mlx":
            chunk_iter = self._iter_mlx_chunks(prompt, max_tokens)
        else:
            chunk_iter, diagnose_ctx = self._iter_hf_chunks(
                prompt, max_tokens, extra_generate_kwargs
            )

        try:
            for chunk, count in chunk_iter:
                if not chunk:
                    continue
                token_count = count
                buf += chunk
                sentences, buf = _flush_sentences(buf)
                for s in sentences:
                    yield s
        finally:
            chunk_iter.close()

        self.last_generate_ms = (time.perf_counter() - t0) * 1000.0
        self.last_generated_tokens = token_count

        tail = buf.strip()
        if tail:
            yield tail

        if token_count == 0 and diagnose_ctx is not None:
            gen_kwargs, input_ids = diagnose_ctx
            self._diagnose_empty_output(gen_kwargs, input_ids)

    def _iter_hf_chunks(self, prompt: str, max_tokens: int, extra_generate_kwargs: dict):
        used_chat_template = bool(getattr(self.tokenizer, "chat_template", None))
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=not used_chat_template,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            streamer=streamer,
        )
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        gen_kwargs.update(extra_generate_kwargs)

        gen_error: list[BaseException] = []

        def _run_generate():
            try:
                self.model.generate(**gen_kwargs)
            except BaseException as exc:
                gen_error.append(exc)
                try:
                    streamer.end()
                except Exception:
                    pass

        thread = threading.Thread(target=_run_generate)
        thread.start()

        def _gen():
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

        return _gen(), (gen_kwargs, input_ids)

    def _iter_mlx_chunks(self, prompt: str, max_tokens: int):
        import mlx_lm
        for resp in mlx_lm.stream_generate(
            self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens
        ):
            yield resp.text, resp.generation_tokens

    def _diagnose_empty_output(self, gen_kwargs: dict, input_ids) -> None:
        """Re-run generation without the streamer and dump raw token IDs so we
        can see *why* nothing made it through (common cause on MPS: fp16 NaN
        logits → argmax collapses to a special token like <pad> / <eos>)."""
        try:
            debug_kwargs = {k: v for k, v in gen_kwargs.items() if k != "streamer"}
            with torch.no_grad():
                out = self.model.generate(**debug_kwargs)
            prompt_len = input_ids.shape[1]
            new_ids = out[0, prompt_len:].tolist()
            raw_text = self.tokenizer.decode(new_ids, skip_special_tokens=False)
            print(
                f"[small_model][diagnose] model={self.model_name} dtype={self.model.dtype} "
                f"device={self.device} generated_ids={new_ids[:32]} "
                f"(total={len(new_ids)}) raw_decoded={raw_text[:300]!r}",
                flush=True,
            )
        except Exception as exc:
            print(f"[small_model][diagnose] failed: {exc!r}", flush=True)

    def generate(
        self, prompt: str, max_new_tokens: Optional[int] = None
    ) -> Iterator[str]:
        yield from self.generate_chat(
            [{"role": "user", "content": prompt}],
            max_new_tokens=max_new_tokens,
        )
