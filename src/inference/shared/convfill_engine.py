"""ConvFill engine shared by the terminal and web demos.

- three demo modes (convfill / frontend_only / backend_only)
- three task modes (normal / rag / mcp)
-  model/device/precision selection, and turn execution

Note: Install the package first with `pip install -e .` from the repo root
"""

import glob
import json
import os
import queue
import sys
import threading
import time
from typing import Optional

from strip_markdown import strip_markdown

from src.utils.api_keys import get_api_key, has_api_key
from src.inference.convfill_stack.run_convfill import ConvFillConfig, ConvFillSystem
from src.inference.shared.dialogue_state_manager import DialogueStateManager
from src.inference.shared.dialogue_state_manager_standalone import DialogueStateManagerStandalone
from src.inference.single_model_stack.small_model_only import SmallModelInference
from src.inference.single_model_stack.large_model_only import LargeModelInference

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_INFERENCE_DIR = os.path.join(_REPO_ROOT, "src", "inference")


NORMAL_CONFIG_PATH = os.path.join(_REPO_ROOT, "configs", "demo_mode", "convfill_config.json")
RAG_CONFIG_PATH = os.path.join(_REPO_ROOT, "configs", "demo_mode", "convfill_config_rag.json")
MCP_CONFIG_PATH = os.path.join(_REPO_ROOT, "configs", "demo_mode", "convfill_config_mcp.json")

CONFIGS_DIR = os.path.join(_REPO_ROOT, "configs", "convfill_frontend_configs")
import re as _re
_FRONTEND_MODEL_RE = _re.compile(r"^convfill_(.+)_nd\.json$")

_BACKEND_MODELS_DIR = os.path.join(_REPO_ROOT, "configs", "backend_model_configs")
_BACKEND_PROVIDERS = ("claude", "openai", "gemini")


class EngineError(Exception):
    """Invalid engine request (bad model name, missing API key, mode conflict)."""


class EngineSink:

    def on_thought(self, thought: str) -> None: ...
    def on_response_fragment(self, text: str) -> None: ...
    def on_rag_context(self, text: str) -> None: ...
    def on_mcp_context(self, text: str) -> None: ...
    def on_turn_complete(self, final_response: str, final_thoughts: list) -> None: ...
    def on_frontend_inference(
        self,
        ms: float,
        tokens: Optional[int] = None,
        first_sentence_ms: Optional[float] = None,
        first_token_ms: Optional[float] = None,
    ) -> None: ...
    def on_phrase_start(self, thought: str, gap_ms: float) -> None: ...

    def stage_frontend_ms(self, ms: Optional[float]) -> None:
        """standalone (frontend_only / backend_only) paths report each fragment's wall-clock gap just before on_response_fragment"""

    def audio_in_flight(self) -> int:
        """utterances emitted but not yet finished playing"""
        return 0

    def on_conversation_boundary(self) -> None:
        """fired when a switch (task/demo mode, model, reset) starts a new conversation."""

    def on_error(self, message: str) -> None:
        """non-fatal, in-turn error (e.g. the model produced no output). """

    def on_reset(self) -> None:
        """ adapter can clear per-turn state (logging record, audio counters)."""


# ----- module-level discovery ----


def _discover_backend_models() -> dict:
    """- Discover available backend providers and their model names.

    - Walks configs/backend_model_configs/<provider>/model_names.json. - Skips providers whose model_names.json is missing or whose API key is not set 
    """
    out: dict = {}
    for provider in _BACKEND_PROVIDERS:
        models_json = os.path.join(_BACKEND_MODELS_DIR, provider, "model_names.json")
        if not os.path.isfile(models_json) or not has_api_key(provider):
            continue
        try:
            with open(models_json, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                names = data.get("model_names") or data.get("models") or []
            else:
                names = data
            if isinstance(names, list) and names:
                out[provider] = [str(n) for n in names]
        except (OSError, json.JSONDecodeError):
            continue
    return out


def _torch_caps() -> tuple:
    """Return (cuda_available, mps_available). Torch import is deferred so the module stays lightweight if the caller never asks for capabilities."""
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        return cuda, mps
    except Exception:
        return False, False


def _device_default_precision(device: str) -> str:
    return "int8"


def _device_default_hf_precision(device: str) -> str:
    if device in ("cuda", "mps"):
        return "bfloat16"
    return "float32"


def compute_device_capabilities() -> dict:
    cuda, mps = _torch_caps()

    def torch_devices() -> list:
        opts = ["cpu"]
        if mps:
            opts.append("mps")
        if cuda:
            opts.append("cuda")
        return opts

    def onnx_devices() -> list:
        opts = ["cpu"]
        if cuda:
            opts.append("cuda")
        return opts

    capabilities = {
        "frontend": torch_devices(),
        "reranker": torch_devices(),
        "tts": onnx_devices(),
        "whisper": onnx_devices(),
    }
    defaults = {
        "frontend": "mps" if mps else "cpu",
        "reranker": "mps" if mps else "cpu",
        "tts": "cuda" if cuda else "cpu",
        "whisper": "cuda" if cuda else "cpu",
    }
    return {"capabilities": capabilities, "defaults": defaults}


def _discover_frontend_models() -> list:
    names: list = []
    for path in glob.glob(os.path.join(CONFIGS_DIR, "convfill_*_nd.json")):
        m = _FRONTEND_MODEL_RE.match(os.path.basename(path))
        if m:
            names.append(m.group(1))
    return sorted(names)


def _discover_small_models() -> tuple:
    names: set = set()
    params: dict = {}
    for path in glob.glob(os.path.join(CONFIGS_DIR, "convfill_*_nd.json")):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        mn = data.get("model_name")
        if not isinstance(mn, str) or not mn:
            continue
        names.add(mn)
        ip = data.get("inference_params")
        if isinstance(ip, dict) and ip:
            params[mn] = dict(ip)
    return sorted(names), params


def _frontend_model_path(name: str) -> str:
    return os.path.join(CONFIGS_DIR, f"convfill_{name}_nd.json")


def _frontend_model_name_from_path(path: str) -> Optional[str]:
    m = _FRONTEND_MODEL_RE.match(os.path.basename(path))
    return m.group(1) if m else None


def _read_frontend_backend(config_path: str) -> str:
    try:
        with open(config_path, "r") as f:
            return json.load(f).get("backend", "mlx")
    except (OSError, json.JSONDecodeError):
        return "mlx"


def _drain(q: "queue.Queue") -> None:
    try:
        while True:
            q.get_nowait()
            try:
                q.task_done()
            except ValueError:
                pass
    except queue.Empty:
        return


class ConvFillEngine:
    """Owns inference state + turn execution for all demo/task modes.

    Synchronous and transport-neutral. Concurrency (serializing turns, queueing
    mode changes during a turn) is the adapter's job — the engine assumes the
    caller does not invoke a mutator while `run_turn` is executing.
    """

    def __init__(self, sink: Optional[EngineSink] = None):
        self.sink = sink or EngineSink()

        self.active_mode: str = "normal"
        self.demo_mode: str = "convfill"

        self.dialogue_state_manager = DialogueStateManager(num_history_turns=1)
        self.standalone_dsm = DialogueStateManagerStandalone()

        self._configs = {
            "normal": ConvFillConfig(NORMAL_CONFIG_PATH),
            "rag": ConvFillConfig(RAG_CONFIG_PATH),
            "mcp": ConvFillConfig(MCP_CONFIG_PATH),
        }
        self._systems: dict = {}

        caps = compute_device_capabilities()
        self.device_capabilities: dict = caps["capabilities"]
        self.device_settings: dict = dict(caps["defaults"])

        self.active_frontend_precision: str = _device_default_precision(self.device_settings["frontend"])

        self.available_frontend_precisions: list = ["int8", "bfloat16", "float16", "float32"]

        hf_dtype_fallback = (
            self.active_frontend_precision
            if self.active_frontend_precision != "int8"
            else _device_default_hf_precision(self.device_settings["frontend"])
        )
        for cfg in self._configs.values():
            cfg.frontend_device = self.device_settings["frontend"]
            cfg.reranker_device = self.device_settings["reranker"]
            cfg.frontend_dtype = hf_dtype_fallback

        self.frontend_models: list = _discover_frontend_models()
        any_path = self._configs["normal"].frontend_model_config_path
        derived = _frontend_model_name_from_path(any_path)
        self.active_frontend_model: Optional[str] = derived if derived in self.frontend_models else (
            self.frontend_models[0] if self.frontend_models else None
        )

        self.backend_models: dict = _discover_backend_models()
        self.active_backend_provider: str = self._configs["normal"].backend_model_mode
        self.active_backend_model: str = self._configs["normal"].backend_model_name

        # Frontend-only demo-mode state. Lazily built; the user only pays HF Hub
        # download cost once they actually switch demo modes.
        self.small_models, self.small_model_params = _discover_small_models()
        self.active_small_model: Optional[str] = (
            self.small_models[0] if self.small_models else None
        )
        self._small_inference = None
        self._small_rag = None
        # Backend-only demo-mode state. Rebuilt whenever provider, model, or
        # sub-mode (normal/rag/mcp) changes — each sub-mode binds a different
        # Jinja template.
        self._large_model_inference = None

        # Eagerly build the initial (normal) system so the first turn is fast.
        self._get_system("normal")

    # ---- system / inference builders ----

    def _get_system(self, mode: str):
        if mode not in self._systems:
            print(f"[engine] building ConvFillSystem for mode={mode}…", flush=True)
            self._systems[mode] = ConvFillSystem(
                self._configs[mode],
                dialogue_state_manager=self.dialogue_state_manager,
                on_thought=self.sink.on_thought,
                on_response_fragment=self.sink.on_response_fragment,
                on_rag_context=self.sink.on_rag_context,
                on_mcp_context=self.sink.on_mcp_context,
                on_turn_complete=self.sink.on_turn_complete,
                on_frontend_inference=self.sink.on_frontend_inference,
                on_phrase_start=self.sink.on_phrase_start,
                emit_tts=False,
                audio_in_flight_fn=self.sink.audio_in_flight,
            )
        return self._systems[mode]

    @property
    def active_system(self):
        return self._get_system(self.active_mode)

    @property
    def active_frontend_backend(self) -> str:
        return _read_frontend_backend(self._configs["normal"].frontend_model_config_path)

    def _build_small_inference(self):
        params = self.small_model_params.get(self.active_small_model, {})
        # Gemma3 + fp16 on MPS produces NaNs and collapses to immediate EOS, so we pin Gemma to CPU regardless of the frontend device setting.
        device = self.device_settings["frontend"]
        if "gemma" in (self.active_small_model or "").lower():
            device = "cpu"

        # MLX int8 path: load pre-converted quantized weights from
        # frontend_model_int8/. dtype/device are not meaningful on this path.
        if self.active_frontend_precision == "int8":
            backend = "mlx"
            dtype_for_call: Optional[str] = None
        else:
            backend = "hf"
            dtype_for_call = self.active_frontend_precision

        if self._small_inference is None \
                or self._small_inference.model_name != self.active_small_model \
                or getattr(self._small_inference, "backend", "hf") != backend \
                or self._small_inference.device != device \
                or self._small_inference.inference_params != params \
                or self._small_inference.dtype != dtype_for_call:
            self._small_inference = SmallModelInference(
                self.active_small_model,
                device=device,
                inference_params=params,
                dtype=dtype_for_call,
                backend=backend,
            )
        return self._small_inference

    def _build_small_rag(self):
        if self._small_rag is not None:
            return self._small_rag
        from src.inference.rag.retreive import RunRAG
        with open(RAG_CONFIG_PATH, "r") as f:
            rag_cfg = json.load(f).get("task_specific_config", {})
        self._small_rag = RunRAG(
            index_path=rag_cfg["rag_index"],
            chunks_path=rag_cfg["rag_chunks"],
            embedding_model=rag_cfg.get("embedding_model", "text-embedding-3-large"),
            reranker_model=rag_cfg.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            device=self.device_settings["reranker"],
        )
        return self._small_rag

    def _build_large_model_inference(self):
        provider = self.active_backend_provider
        model_name = self.active_backend_model
        sub_mode = self.active_mode
        if self._large_model_inference is None \
                or self._large_model_inference.provider != provider \
                or self._large_model_inference.model_name != model_name \
                or self._large_model_inference.sub_mode != sub_mode:
            self._large_model_inference = LargeModelInference(
                provider=provider,
                model_name=model_name,
                sub_mode=sub_mode,
            )
        return self._large_model_inference

    # ---- turn execution ----

    def turn_precheck(self) -> Optional[str]:
        """Return an error message if the current demo/task/model combo can't run
        a turn, else None. Adapters call this before run_turn and surface the
        message however they report errors."""
        if self.demo_mode == "backend_only":
            if not self.active_backend_provider or not self.active_backend_model:
                return "No backend model selected for backend_only demo mode."
        elif self.demo_mode == "frontend_only":
            if self.active_mode == "mcp":
                return "MCP is not available in frontend_only demo mode."
            if not self.active_small_model:
                return "No small model selected for frontend_only demo mode."
        return None

    def run_turn(self, text: str) -> None:
        """Run one full turn, blocking until it completes. Dispatches on demo
        mode; events flow out through the sink. Call turn_precheck() first."""
        if self.demo_mode == "backend_only":
            self._run_backend_only_turn(text)
        elif self.demo_mode == "frontend_only":
            self._run_small_model_turn(text)
        else:
            self.active_system.run_turn(text)

    def _run_backend_only_turn(self, text: str) -> None:
        """Synchronous backend-only turn (no local frontend model)."""
        self.standalone_dsm.update_user_turn(text)

        rag_context = None
        if self.active_mode == "rag":
            rag = self._build_small_rag()
            rag_context = rag.rag_infer(text)
            if rag_context:
                self.sink.on_rag_context(rag_context)

        mcp_tools = None
        dispatch_tool = None
        if self.active_mode == "mcp":
            mcp_system = self._get_system("mcp")
            backend = mcp_system.convfill_backend
            mcp_tools = getattr(backend, "mcp_tools", None) or []
            mcp_hub = getattr(backend, "mcp_hub", None)
            if mcp_hub is not None:
                dispatch_tool = mcp_hub.call_tool

        inference = self._build_large_model_inference()
        transcript = self.standalone_dsm.get_transcript()

        t0 = time.perf_counter()
        collected: list = []
        for fragment in inference.generate(
            transcript,
            rag_context=rag_context,
            mcp_tools=mcp_tools,
            dispatch_tool=dispatch_tool,
            on_tool_call=self.sink.on_mcp_context,
        ):
            if not fragment:
                continue
            # frontend_inference_ms stays None since no local frontend model runs in backend_only mode.
            self.sink.stage_frontend_ms(None)
            collected.append(fragment)
            self.sink.on_response_fragment(fragment)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        tokens = inference.last_generated_tokens
        self.sink.on_frontend_inference(elapsed_ms, tokens)

        final_response = " ".join(collected).strip()
        if not final_response:
            msg = (
                f"Backend model {self.active_backend_provider}/{self.active_backend_model} "
                f"produced no output."
            )
            print(f"[engine:backend_only] {msg}", flush=True)
            self.sink.on_error(msg)
        self.standalone_dsm.update_response(final_response)
        self.sink.on_turn_complete(final_response, [])

    def _run_small_model_turn(self, text: str) -> None:
        """Synchronous frontend-only turn (small local model, no backend thoughts)."""
        self.standalone_dsm.update_user_turn(text)

        rag_context = None
        if self.active_mode == "rag":
            rag = self._build_small_rag()
            rag_context = rag.rag_infer(text)
            if rag_context:
                self.sink.on_rag_context(rag_context)

        inference = self._build_small_inference()
        messages = self.standalone_dsm.get_messages()

        t0 = time.perf_counter()
        prev_t = t0
        collected: list = []
        for fragment in inference.generate_chat(messages, rag_context=rag_context):
            if not fragment:
                continue
            # strip md for small models cause they don't follow instructions
            cleaned = strip_markdown(fragment).strip()
            if not cleaned:
                continue
            now = time.perf_counter()
            self.sink.stage_frontend_ms((now - prev_t) * 1000.0)
            collected.append(cleaned)
            self.sink.on_response_fragment(cleaned)
            prev_t = now

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        tokens = inference.last_generated_tokens
        self.sink.on_frontend_inference(elapsed_ms, tokens)

        final_response = " ".join(collected).strip()
        if not final_response:
            msg = (
                f"Model {self.active_small_model} produced no output "
                f"(generated {tokens or 0} tokens in {elapsed_ms:.0f}ms)."
            )
            print(f"[engine:frontend_only] {msg}", flush=True)
            self.sink.on_error(msg)
        self.standalone_dsm.update_response(final_response)
        self.sink.on_turn_complete(final_response, [])

    # ---- state mutators ----

    def reset(self) -> None:
        """Abort any in-flight work, clear queues, and reset conversation state
        across all built systems. Fires sink.on_reset() so the adapter can clear
        its own per-turn state too."""
        built = [s for s in self._systems.values() if s is not None]
        for sys_ in built:
            sys_.done_event.set()
        time.sleep(0.05)
        for sys_ in built:
            sys_.convfill_frontend.reset_turn_state()
            _drain(sys_.thought_queue)
            _drain(sys_.tts_queue)
            sys_.done_event.clear()
        self.dialogue_state_manager.reset()
        self.standalone_dsm.reset()
        self.sink.on_reset()

    def set_mode(self, mode: str) -> None:
        if mode not in ("normal", "rag", "mcp"):
            raise EngineError(f"Unknown mode: {mode}")
        if mode == "mcp" and self.demo_mode == "frontend_only":
            raise EngineError("MCP is not available in frontend_only demo mode.")
        if mode == self.active_mode:
            return
        if self.demo_mode == "convfill" and mode not in self._systems:
            self._get_system(mode)
        self.active_mode = mode
        self._large_model_inference = None
        self.sink.on_conversation_boundary()

    def set_demo_mode(self, demo_mode: str) -> None:
        if demo_mode not in ("convfill", "frontend_only", "backend_only"):
            raise EngineError(f"Unknown demo mode: {demo_mode}")
        if demo_mode == self.demo_mode:
            return
        self.reset()
        self.demo_mode = demo_mode
        self.sink.on_conversation_boundary()

        if demo_mode == "frontend_only" and self.active_mode == "mcp":
            self.active_mode = "normal"

    def set_small_model(self, name: str) -> None:
        if name not in self.small_models:
            raise EngineError(f"Unknown small model: {name}")
        if name == self.active_small_model:
            return
        self.reset()
        self.active_small_model = name
        self._small_inference = None
        if self.demo_mode == "frontend_only":
            self._build_small_inference()
        self.sink.on_conversation_boundary()

    def set_frontend_model(self, name: str) -> None:
        if name not in self.frontend_models:
            raise EngineError(f"Unknown frontend model: {name}")
        if name == self.active_frontend_model:
            return
        self.reset()

        new_path = _frontend_model_path(name)
        for cfg in self._configs.values():
            cfg.frontend_model_config_path = new_path
        self._systems = {}
        self.active_frontend_model = name

        self._get_system(self.active_mode)
        self.sink.on_conversation_boundary()

    def set_backend_model(self, provider: str, name: str) -> None:
        if provider not in self.backend_models:
            raise EngineError(f"Unknown backend provider: {provider}")
        if name not in self.backend_models[provider]:
            raise EngineError(f"Unknown backend model: {provider}/{name}")
        if not has_api_key(provider):
            raise EngineError(f"Missing API key for {provider}; add it to the .env file at the repo root")
        if provider == self.active_backend_provider and name == self.active_backend_model:
            return
        self.reset()
        for cfg in self._configs.values():
            cfg.backend_model_mode = provider
            cfg.backend_model_name = name
        self._systems = {}
        self.active_backend_provider = provider
        self.active_backend_model = name
        self._large_model_inference = None
        if self.demo_mode == "convfill":
            self._get_system(self.active_mode)
        self.sink.on_conversation_boundary()

    def set_precision(self, precision: str) -> None:
        if precision not in self.available_frontend_precisions:
            raise EngineError(f"Unknown precision: {precision}")
        if precision == self.active_frontend_precision:
            return
        self.reset()
        self.active_frontend_precision = precision

        if precision != "int8":
            for cfg in self._configs.values():
                cfg.frontend_dtype = precision
        self._systems = {}
        self._small_inference = None
        if self.demo_mode == "convfill":
            self._get_system(self.active_mode)
        elif self.demo_mode == "frontend_only" and self.active_small_model:
            self._build_small_inference()

    def set_device(self, component: str, device: str) -> None:
        """Record a device choice and, for the two PyTorch components
        (frontend / reranker), rebuild affected state. TTS/Whisper are
        transport-side services the engine doesn't own — it only records the
        setting so the adapter can apply it."""
        if component not in self.device_capabilities:
            raise EngineError(f"Unknown device component: {component}")
        allowed = self.device_capabilities[component]
        if device not in allowed:
            raise EngineError(f"Device {device} not available for {component} (allowed: {allowed})")
        if self.device_settings[component] == device:
            return
        self.device_settings[component] = device
        if component == "frontend":
            self.reset()
            for cfg in self._configs.values():
                cfg.frontend_device = device
                cfg.frontend_dtype = self.active_frontend_precision
            self._systems = {}
            self._small_inference = None
            if self.demo_mode == "convfill":
                self._get_system(self.active_mode)
            elif self.active_small_model:
                self._build_small_inference()
        elif component == "reranker":
            self.reset()
            for cfg in self._configs.values():
                cfg.reranker_device = device
            self._systems = {}
            self._small_rag = None
            if self.demo_mode == "convfill":
                self._get_system(self.active_mode)