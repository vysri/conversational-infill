"""TTS streaming for the web backend.

Yields raw int16 little-endian PCM chunks for a given string. The HTTP layer
(main.py) wraps this in a StreamingResponse so the browser can begin playback
before the full sentence is rendered. Two engines are wired up; flip
TTS_ENGINE below to switch.
"""

import glob
import os
import subprocess
import threading
from typing import Iterator, Literal, Optional

from piper.voice import PiperVoice
from strip_markdown import strip_markdown


_SAY_SAMPLE_RATE = 22050
# `say -v` voice name (e.g. "Samantha"). Empty uses the macOS system voice.
_SAY_VOICE = ""

_state = {"mode": None, "model_path": None}


def configure(mode: Literal["piper", "say"]) -> None:
    if mode not in ("piper", "say"):
        raise ValueError(f"Unknown tts_mode: {mode!r}")
    model_path = None
    if mode == "piper":
        model_dir = os.path.join(os.path.dirname(__file__), "../../src/tts/voices")
        model_dir = os.path.abspath(model_dir)
        if not os.path.isdir(model_dir):
            raise ValueError(f"TTS model directory not found: {model_dir!r}")
        try:
            onnx_files = glob.glob(os.path.join(model_dir, "*.onnx"))
        except PermissionError as e:
            raise ValueError(f"Permission denied reading TTS model directory: {model_dir!r}") from e
        if len(onnx_files) == 0:
            raise ValueError(f"No .onnx files found in {model_dir!r}")
        if len(onnx_files) > 1:
            raise ValueError(f"Expected exactly one .onnx file in {model_dir!r}, found {len(onnx_files)}: {onnx_files}")
        model_path = onnx_files[0]
    _state["mode"] = mode
    _state["model_path"] = model_path
    with _voice_lock:
        _voices.clear()


_voice_lock = threading.Lock()
# Cache voices per device. Piper only meaningfully supports cpu and cuda
# (no Metal/MPS backend in piper's ONNX Runtime wrapper).
_voices: dict[str, PiperVoice] = {}
_active_device: str = "cpu"


def _load_voice(device: str) -> PiperVoice:
    if _state["model_path"] is None:
        raise RuntimeError("TTS not configured; call configure() first")
    use_cuda = device == "cuda"
    try:
        return PiperVoice.load(_state["model_path"], use_cuda=use_cuda)
    except TypeError:
        # Older piper builds don't accept use_cuda; fall back to CPU load.
        return PiperVoice.load(_state["model_path"])


def _get_voice() -> PiperVoice:
    with _voice_lock:
        v = _voices.get(_active_device)
        if v is None:
            v = _load_voice(_active_device)
            _voices[_active_device] = v
        return v


def set_device(device: str) -> None:
    """Switch the active Piper device. Loads lazily on next synth call."""
    global _active_device
    with _voice_lock:
        _active_device = device


def get_active_device() -> str:
    return _active_device


def get_sample_rate() -> int:
    if _state["mode"] is None:
        raise RuntimeError("TTS not configured; call configure() first")
    if _state["mode"] == "say":
        return _SAY_SAMPLE_RATE
    return _get_voice().config.sample_rate


def _stream_say_pcm(text: str) -> Iterator[bytes]:
    # Plays via `say` on the server's CoreAudio output; nothing is streamed
    # to the browser. Runs synchronously so consecutive fragments don't
    # overlap on the server's speakers.
    cmd = ["say"]
    if _SAY_VOICE:
        cmd.extend(["-v", _SAY_VOICE])
    cmd.append(text)
    subprocess.run(cmd, check=False)
    yield from ()


def stream_pcm(text: str) -> Iterator[bytes]:
    if _state["mode"] is None:
        raise RuntimeError("TTS not configured; call configure() first")
    text = strip_markdown(text).strip()
    if not text:
        return
    if _state["mode"] == "say":
        yield from _stream_say_pcm(text)
        return
    voice = _get_voice()
    for chunk in voice.synthesize(text):
        yield chunk.audio_int16_bytes


def warmup() -> None:
    # Piper's first synthesize call pays ONNX session creation + first-inference
    # cost (~0.5-1.5s). Burn it here at server start (or after a device flip) so
    # the first user-facing TTS request doesn't.
    if _state["mode"] is None:
        raise RuntimeError("TTS not configured; call configure() first")
    if _state["mode"] == "say":
        print(">>> [TTS] using macOS `say` (fire-and-forget, server speakers)", flush=True)
        return
    voice = _get_voice()
    for _ in voice.synthesize("hello"):
        pass
    print(f">>> [TTS] piper warmed up on device={_active_device}", flush=True)
