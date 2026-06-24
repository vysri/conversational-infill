import os
import tempfile
import threading

from faster_whisper import WhisperModel


WHISPER_MODEL_SIZE = os.environ.get("CONVFILL_WHISPER_MODEL", "base")

_lock = threading.Lock()
_active_device: str = os.environ.get("CONVFILL_WHISPER_DEVICE", "cpu")
# Cache models per (device, compute) so flipping device is cheap once warmed.
_models: dict[tuple[str, str], WhisperModel] = {}


def _compute_for(device: str) -> str:
    # Env override wins if set explicitly; otherwise pick a sane default per device.
    env = os.environ.get("CONVFILL_WHISPER_COMPUTE")
    if env:
        return env
    return "float16" if device == "cuda" else "int8"


def get_model() -> WhisperModel:
    with _lock:
        compute = _compute_for(_active_device)
        key = (_active_device, compute)
        m = _models.get(key)
        if m is None:
            print(f"[Whisper] loading model={WHISPER_MODEL_SIZE} device={_active_device} compute={compute}", flush=True)
            m = WhisperModel(WHISPER_MODEL_SIZE, device=_active_device, compute_type=compute)
            _models[key] = m
        return m


def set_device(device: str) -> None:
    """Switch the active Whisper device. Loads lazily on next transcribe call."""
    global _active_device
    with _lock:
        _active_device = device


def get_active_device() -> str:
    return _active_device


def transcribe_bytes(audio_bytes: bytes, suffix: str = ".webm") -> str:
    model = get_model()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        segments, _ = model.transcribe(tmp_path, beam_size=1, vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
