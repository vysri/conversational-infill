import os
import re
import time
from typing import Iterator, Optional

from jinja2 import Environment, FileSystemLoader

from src.inference.single_model_stack.model_inference_functions import BackendInference
from src.utils.api_keys import get_api_key


_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
_BACKEND_PROMPTS_DIR = os.path.join(_REPO_ROOT, "configs", "backend_only_prompts")
_TEMPLATE_BY_MODE = {
    "normal": "backend_model_prompt_template.txt",
    "rag": "backend_model_prompt_template_rag.txt",
    "mcp": "backend_model_prompt_template_mcp.txt",
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _flush_sentences(buf: str) -> tuple[list[str], str]:
    parts = _SENTENCE_SPLIT_RE.split(buf)
    if len(parts) == 1:
        return [], buf
    return [p for p in parts[:-1] if p.strip()], parts[-1]


class LargeModelInference:
    def __init__(
        self,
        provider: str,
        model_name: str,
        sub_mode: str = "normal",
    ):
        if sub_mode not in _TEMPLATE_BY_MODE:
            raise ValueError(f"Unsupported sub_mode: {sub_mode}")
        self.provider = provider
        self.model_name = model_name
        self.sub_mode = sub_mode

        self.api_key = get_api_key(provider)

        self._backend = BackendInference(
            api_key=self.api_key,
            model_name=model_name,
            model_mode=provider,
        )

        self._template_path = os.path.join(_BACKEND_PROMPTS_DIR, _TEMPLATE_BY_MODE[sub_mode])
        env = Environment(loader=FileSystemLoader(_BACKEND_PROMPTS_DIR))
        self._template = env.get_template(_TEMPLATE_BY_MODE[sub_mode])

        self.last_generate_ms: Optional[float] = None
        self.last_generated_tokens: Optional[int] = None

    def _build_prompt(self, transcript: str, rag_context: Optional[str]) -> str:
        return self._template.render(
            transcript=transcript,
            rag_context=rag_context or "",
        )

    def generate(
        self,
        transcript: str,
        *,
        rag_context: Optional[str] = None,
        mcp_tools=None,
        dispatch_tool=None,
        on_tool_call=None,
    ) -> Iterator[str]:
        prompt = self._build_prompt(transcript, rag_context)

        t0 = time.perf_counter()
        chunk_count = 0
        buf = ""

        if self.sub_mode == "mcp" and mcp_tools:
            for kind, payload in self._backend.infer_text_with_tools(
                prompt, mcp_tools, dispatch_tool
            ):
                if kind == "tool_call":
                    if on_tool_call is not None:
                        name = payload.get("name", "?")
                        args = payload.get("input", {}) or {}
                        arg_keys = ", ".join(sorted(args.keys())) if args else ""
                        label = f"Called {name}({arg_keys})" if arg_keys else f"Called {name}()"
                        on_tool_call(label)
                elif kind == "text":
                    chunk_count += 1
                    buf += payload
                    sentences, buf = _flush_sentences(buf)
                    for s in sentences:
                        yield s
        else:
            for chunk in self._backend.infer_text(prompt):
                chunk_count += 1
                buf += chunk
                sentences, buf = _flush_sentences(buf)
                for s in sentences:
                    yield s

        tail = buf.strip()
        if tail:
            yield tail

        self.last_generate_ms = (time.perf_counter() - t0) * 1000.0
        self.last_generated_tokens = chunk_count
