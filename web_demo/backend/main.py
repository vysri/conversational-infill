import asyncio
import json
import os

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .session import ConvFillSession
from .tts_service import configure as tts_configure, get_sample_rate, stream_pcm, warmup as tts_warmup
from .whisper_service import transcribe_bytes
from src.inference.convfill_stack.run_convfill import ConvFillConfig


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")


app = FastAPI(title="ConvFill Web")


@app.on_event("startup")
async def _startup() -> None:
    config_path = os.environ.get("CONVFILL_CONFIG", os.path.join(_REPO_ROOT, "configs", "demo_mode", "convfill_full_config.json"))
    cfg = ConvFillConfig(config_path, mode="normal")
    tts_configure(cfg.tts_mode)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, tts_warmup)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Sample-Rate", "X-Channels"],
)


@app.get("/api/tts")
async def tts(text: str) -> StreamingResponse:
    sr = get_sample_rate()
    headers = {
        "X-Sample-Rate": str(sr),
        "X-Channels": "1",
        "Cache-Control": "no-store",
    }
    return StreamingResponse(stream_pcm(text), media_type="application/octet-stream", headers=headers)


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)) -> JSONResponse:
    audio_bytes = await file.read()
    suffix = "." + (file.filename.rsplit(".", 1)[-1] if file.filename and "." in file.filename else "webm")
    text = await asyncio.get_event_loop().run_in_executor(None, transcribe_bytes, audio_bytes, suffix)
    return JSONResponse({"text": text})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    print("[ws] client connected; loading session (this may take 10-30s on first turn)…", flush=True)
    loop = asyncio.get_event_loop()
    try:
        session = await loop.run_in_executor(None, ConvFillSession, loop)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        try:
            await ws.send_text(json.dumps({"type": "error", "message": f"Session init failed: {exc!r}"}))
        finally:
            await ws.close()
        return
    print("[ws] session ready", flush=True)
    try:
        await ws.send_text(json.dumps({"type": "ready"}))
        await ws.send_text(json.dumps({
            "type": "frontend_models",
            "names": session.frontend_models,
            "active": session.active_frontend_model,
        }))
        await ws.send_text(json.dumps(session.frontend_precision_event()))
        _UI_DEVICE_COMPONENTS = ("frontend",)
        await ws.send_text(json.dumps({
            "type": "device_capabilities",
            "capabilities": {k: session.device_capabilities[k] for k in _UI_DEVICE_COMPONENTS if k in session.device_capabilities},
            "active": {k: session.device_settings[k] for k in _UI_DEVICE_COMPONENTS if k in session.device_settings},
        }))
        await ws.send_text(json.dumps({
            "type": "backend_models",
            "providers": session.backend_models,
            "active_provider": session.active_backend_provider,
            "active_name": session.active_backend_model,
        }))
        await ws.send_text(json.dumps({
            "type": "modes",
            "names": session.available_modes,
            "active": session.active_mode,
        }))
        await ws.send_text(json.dumps({
            "type": "demo_mode",
            "mode": session.demo_mode,
        }))
        await ws.send_text(json.dumps({
            "type": "small_models",
            "names": session.small_models,
            "active": session.active_small_model,
        }))
    except Exception:
        # Client closed during the (slow) session init.
        return

    async def sender() -> None:
        try:
            while True:
                event = await session.outbound.get()
                try:
                    await ws.send_text(json.dumps(event))
                except Exception:
                    # Client closed mid-send (ClientDisconnected, ConnectionClosed*,
                    # WebSocketDisconnect, RuntimeError on closed transport, etc.).
                    # Treat any send failure as "connection gone" and shut the sender down.
                    return
        except asyncio.CancelledError:
            raise

    sender_task = asyncio.create_task(sender())

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await session.outbound.put({"type": "error", "message": "Invalid JSON."})
                continue
            mtype = msg.get("type")
            if mtype == "user_message":
                asyncio.create_task(session.handle_user_message(msg.get("text", "")))
            elif mtype == "set_mode":
                await session.set_mode(msg.get("mode", "normal"))
            elif mtype == "set_frontend_model":
                asyncio.create_task(session.set_frontend_model(msg.get("name", "")))
            elif mtype == "set_backend_model":
                asyncio.create_task(session.set_backend_model(msg.get("provider", ""), msg.get("name", "")))
            elif mtype == "set_device":
                asyncio.create_task(session.set_device(msg.get("component", ""), msg.get("device", "")))
            elif mtype == "set_demo_mode":
                asyncio.create_task(session.set_demo_mode(msg.get("mode", "convfill")))
            elif mtype == "set_small_model":
                asyncio.create_task(session.set_small_model(msg.get("name", "")))
            elif mtype == "set_precision":
                asyncio.create_task(session.set_precision(msg.get("precision", "")))
            elif mtype == "reset":
                await session.reset()
            elif mtype == "fragment_played":
                idx = int(msg.get("fragment_index", 0))
                tts_ms = msg.get("tts_first_byte_ms")
                utterance_ms = msg.get("tts_utterance_ms")
                total_ms = msg.get("total_response_ms")
                voiced_first_ms = msg.get("voiced_first_response_ms")
                if (
                    tts_ms is not None
                    or utterance_ms is not None
                    or total_ms is not None
                    or voiced_first_ms is not None
                ):
                    session.on_fragment_played_with_metrics(
                        idx,
                        tts_ms,
                        utterance_ms,
                        total_ms,
                        voiced_first_ms,
                    )
                else:
                    session.on_fragment_played(idx)
            elif mtype == "start_new_log":
                asyncio.create_task(session.start_new_log())
            elif mtype == "ping":
                await session.outbound.put({"type": "pong"})
            else:
                await session.outbound.put({"type": "error", "message": f"Unknown type: {mtype}"})
    except WebSocketDisconnect:
        pass
    finally:
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass


# Serve the built frontend (after `cd frontend && npm run build`).
# In dev, redirect bare visits to the Vite dev server at :5173.
if os.path.isdir(_FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
else:
    @app.get("/")
    async def _root_redirect() -> RedirectResponse:
        return RedirectResponse(url="http://127.0.0.1:5173")
