import { useCallback, useEffect, useRef, useState } from "react";
import ChatPanel from "./components/ChatPanel";
import ThoughtsPanel from "./components/ThoughtsPanel";
import { useTTS } from "./hooks/useTTS";
import { useWebSocket } from "./hooks/useWebSocket";
import type {
  ChatMessage,
  DemoMode,
  Device,
  DeviceCapabilities,
  DeviceComponent,
  DeviceSettings,
  Mode,
  ServerEvent,
  TurnBlock,
} from "./types";

const DEVICE_COMPONENTS: { key: DeviceComponent; label: string }[] = [
  { key: "frontend", label: "HF Frontend" },
  { key: "reranker", label: "RAG Reranker" },
];

const MODE_LABELS: Record<string, string> = {
  normal: "Normal",
  rag: "RAG",
  mcp: "MCP",
};

const DEMO_MODE_LABELS: Record<DemoMode, string> = {
  convfill: "ConvFill (frontend + backend)",
  frontend_only: "Frontend only",
  backend_only: "Backend only",
};

const DEVICE_LABELS: Record<Device, string> = {
  cpu: "CPU",
  mps: "MPS (Apple GPU)",
  cuda: "CUDA (NVIDIA GPU)",
};

export default function App() {
  const [mode, setMode] = useState<Mode>("normal");
  const [pendingMode, setPendingMode] = useState<Mode | null>(null);
  const [availableModes, setAvailableModes] = useState<string[]>(["normal"]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [turns, setTurns] = useState<TurnBlock[]>([]);
  const [turnInProgress, setTurnInProgress] = useState(false);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showThoughts, setShowThoughts] = useState(true);
  const [showTiming, setShowTiming] = useState(true);
  const [showAssistantBubbles, setShowAssistantBubbles] = useState(false);
  const [frontendModels, setFrontendModels] = useState<string[]>([]);
  const [activeFrontendModel, setActiveFrontendModel] = useState<string | null>(null);
  const [pendingFrontendModel, setPendingFrontendModel] = useState<string | null>(null);
  const [backendProviders, setBackendProviders] = useState<Record<string, string[]>>({});
  const [activeBackendProvider, setActiveBackendProvider] = useState<string | null>(null);
  const [pendingBackendProvider, setPendingBackendProvider] = useState<string | null>(null);
  const [activeBackendModel, setActiveBackendModel] = useState<string | null>(null);
  const [pendingBackendModel, setPendingBackendModel] = useState<string | null>(null);
  const [deviceCapabilities, setDeviceCapabilities] = useState<DeviceCapabilities | null>(null);
  const [activeDevices, setActiveDevices] = useState<DeviceSettings | null>(null);
  const [pendingDevices, setPendingDevices] = useState<Partial<Record<DeviceComponent, Device>>>({});
  const [demoMode, setDemoMode] = useState<DemoMode>("convfill");
  const [pendingDemoMode, setPendingDemoMode] = useState<DemoMode | null>(null);
  const [smallModels, setSmallModels] = useState<string[]>([]);
  const [activeSmallModel, setActiveSmallModel] = useState<string | null>(null);
  const [pendingSmallModel, setPendingSmallModel] = useState<string | null>(null);
  const [precision, setPrecision] = useState<string | null>(null);
  const [availablePrecisions, setAvailablePrecisions] = useState<string[]>([]);
  const [pendingPrecision, setPendingPrecision] = useState<string | null>(null);
  const [frontendBackend, setFrontendBackend] = useState<"mlx" | "hf" | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const turnCounterRef = useRef(0);
  const latestSpeakRef = useRef<Promise<void>>(Promise.resolve());
  const sendRef = useRef<((msg: object) => void) | null>(null);

  const [metrics, setMetrics] = useState<{
    frontendFirstMs: number | null;
    backendFirstMs: number | null;
    voicedFirstMs: number | null;
    frontendAvgMs: number | null;
    frontendAvgTokPerSec: number | null;
    ttsAvgMs: number | null;
    totalResponseMs: number | null;
  }>({
    frontendFirstMs: null,
    backendFirstMs: null,
    voicedFirstMs: null,
    frontendAvgMs: null,
    frontendAvgTokPerSec: null,
    ttsAvgMs: null,
    totalResponseMs: null,
  });
  const turnStartTsRef = useRef<number | null>(null);
  const firstBackendThoughtSeenRef = useRef(false);
  const frontendInferenceMsRef = useRef<number[]>([]);
  const frontendInferenceTokensRef = useRef<number[]>([]);
  const ttsLatenciesRef = useRef<number[]>([]);
  // Perf-clock timestamp of the first TTS playback start in the current turn.
  // Used both for the "Voiced first response" metric in the timing bar and for
  // the voiced_first_response_ms field on the first fragment_played ack.
  const ttsFirstPlaybackStartTsRef = useRef<number | null>(null);
  // Per-fragment TTS first-byte latency, keyed by fragment_index. Drained as
  // each phrase finishes playback and the metric is sent back to the server
  // for JSONL logging.
  const ttsLatencyByFragmentRef = useRef<Map<number, number>>(new Map());
  // Per-fragment playback start/end (perf-clock ms), used to compute
  // tts_utterance_ms (audio playback duration) per sub-clause.
  const ttsPlaybackStartByFragmentRef = useRef<Map<number, number>>(new Map());
  const ttsPlaybackEndByFragmentRef = useRef<Map<number, number>>(new Map());
  const voicedFirstSentRef = useRef<boolean>(false);

  const handleTTSLatency = useCallback((ms: number, fragmentIndex: number) => {
    ttsLatenciesRef.current.push(ms);
    ttsLatencyByFragmentRef.current.set(fragmentIndex, ms);
    const ls = ttsLatenciesRef.current;
    const avg = ls.reduce((a, b) => a + b, 0) / ls.length;
    setMetrics((m) => ({ ...m, ttsAvgMs: Math.round(avg) }));
  }, []);
  const handleTTSPlaybackStart = useCallback((perfTimeMs: number, fragmentIndex: number) => {
    ttsPlaybackStartByFragmentRef.current.set(fragmentIndex, perfTimeMs);
    if (ttsFirstPlaybackStartTsRef.current === null) {
      ttsFirstPlaybackStartTsRef.current = perfTimeMs;
      if (turnStartTsRef.current !== null) {
        setMetrics((m) => ({
          ...m,
          voicedFirstMs:
            m.voicedFirstMs ?? Math.round(perfTimeMs - turnStartTsRef.current!),
        }));
      }
    }
  }, []);
  const handleTTSPlaybackEnd = useCallback((perfTimeMs: number, fragmentIndex: number) => {
    ttsPlaybackEndByFragmentRef.current.set(fragmentIndex, perfTimeMs);
  }, []);
  const { speak, cancel: cancelTTS } = useTTS(
    handleTTSLatency,
    handleTTSPlaybackStart,
    handleTTSPlaybackEnd,
  );

  const clearAllUiState = useCallback(() => {
    setMessages([]);
    setTurns([]);
    setTurnInProgress(false);
    // NOTE: do NOT reset turnCounterRef — keep it monotonic so any in-flight
    // speak() promises from a discarded turn can't match a future turnId.
    cancelTTS();
    turnStartTsRef.current = null;
    firstBackendThoughtSeenRef.current = false;
    frontendInferenceMsRef.current = [];
    frontendInferenceTokensRef.current = [];
    ttsLatenciesRef.current = [];
    ttsFirstPlaybackStartTsRef.current = null;
    ttsLatencyByFragmentRef.current.clear();
    ttsPlaybackStartByFragmentRef.current.clear();
    ttsPlaybackEndByFragmentRef.current.clear();
    voicedFirstSentRef.current = false;
    setMetrics({
      frontendFirstMs: null,
      backendFirstMs: null,
      voicedFirstMs: null,
      frontendAvgMs: null,
      frontendAvgTokPerSec: null,
      ttsAvgMs: null,
      totalResponseMs: null,
    });
  }, [cancelTTS]);

  const handleEvent = useCallback(
    (e: ServerEvent) => {
      switch (e.type) {
        case "turn_start": {
          turnCounterRef.current += 1;
          const id = turnCounterRef.current;
          setTurns((t) => [...t, { id, items: [], complete: false }]);
          setMessages((m) => [
            ...m,
            { role: "assistant", text: "", pending: true, turnId: id },
          ]);
          setTurnInProgress(true);
          break;
        }
        case "rag_context":
          setTurns((t) => appendToActiveTurn(t, { kind: "rag_context", text: e.text }));
          break;
        case "mcp_context":
          setTurns((t) => appendToActiveTurn(t, { kind: "mcp_context", text: e.text }));
          break;
        case "thought":
          if (!firstBackendThoughtSeenRef.current && turnStartTsRef.current !== null) {
            firstBackendThoughtSeenRef.current = true;
            setMetrics((m) => ({
              ...m,
              backendFirstMs: Math.round(performance.now() - turnStartTsRef.current!),
            }));
          }
          setTurns((t) => appendToActiveTurn(t, { kind: "thought", text: e.text }));
          break;
        case "silence":
          setTurns((t) => appendToActiveTurn(t, { kind: "silence" }));
          break;
        case "frontend_timing": {
          const ms = e.ms;
          const arr = frontendInferenceMsRef.current;
          arr.push(ms);
          if (e.tokens !== null) frontendInferenceTokensRef.current.push(e.tokens);
          const avg = arr.reduce((a, b) => a + b, 0) / arr.length;
          const totalMs = arr.reduce((a, b) => a + b, 0);
          const totalTok = frontendInferenceTokensRef.current.reduce((a, b) => a + b, 0);
          const tps = totalTok > 0 && totalMs > 0 ? totalTok / (totalMs / 1000) : null;
          setMetrics((m) => ({
            ...m,
            frontendAvgMs: Math.round(avg),
            frontendAvgTokPerSec: tps !== null ? Math.round(tps) : null,
          }));
          break;
        }
        case "response_fragment": {
          const turnId = turnCounterRef.current;
          const fragText = e.text;
          const fragIdx = e.fragment_index;
          if (turnStartTsRef.current !== null) {
            const elapsed = Math.round(performance.now() - turnStartTsRef.current);
            setMetrics((m) => ({
              ...m,
              frontendFirstMs: m.frontendFirstMs ?? elapsed,
            }));
          }
          // Speak first; reveal the bubble updates only once this utterance's
          // audio actually finishes playing. Late-arriving promises whose turnId
          // no longer matches any pending bubble are no-ops thanks to the idx<0
          // guard in the updater (and turnCounterRef is monotonic across resets).
          const speakPromise = speak(fragText, fragIdx);
          latestSpeakRef.current = speakPromise;
          // Ack playback completion so backend stops emitting <sil> fillers
          // while this utterance is still in flight. Piggyback per-phrase
          // TTS first-byte and total response time for JSONL logging.
          void speakPromise.then(() => {
            const ttsMs = ttsLatencyByFragmentRef.current.get(fragIdx);
            ttsLatencyByFragmentRef.current.delete(fragIdx);
            const playbackStart = ttsPlaybackStartByFragmentRef.current.get(fragIdx);
            const playbackEnd = ttsPlaybackEndByFragmentRef.current.get(fragIdx);
            ttsPlaybackStartByFragmentRef.current.delete(fragIdx);
            ttsPlaybackEndByFragmentRef.current.delete(fragIdx);
            const utteranceMs =
              playbackStart !== undefined && playbackEnd !== undefined
                ? Math.max(0, playbackEnd - playbackStart)
                : null;
            const totalMs =
              turnStartTsRef.current !== null
                ? performance.now() - turnStartTsRef.current
                : null;
            let voicedFirstMs: number | null = null;
            if (
              !voicedFirstSentRef.current
              && ttsFirstPlaybackStartTsRef.current !== null
              && turnStartTsRef.current !== null
            ) {
              voicedFirstSentRef.current = true;
              voicedFirstMs =
                ttsFirstPlaybackStartTsRef.current - turnStartTsRef.current;
            }
            sendRef.current?.({
              type: "fragment_played",
              fragment_index: fragIdx,
              tts_first_byte_ms: ttsMs ?? null,
              tts_utterance_ms: utteranceMs,
              total_response_ms: totalMs,
              voiced_first_response_ms: voicedFirstMs,
            });
          });
          void latestSpeakRef.current.then(() => {
            setMessages((m) => {
              const copy = [...m];
              let idx = -1;
              for (let i = copy.length - 1; i >= 0; i--) {
                if (copy[i].role === "assistant" && copy[i].pending && copy[i].turnId === turnId) {
                  idx = i;
                  break;
                }
              }
              if (idx < 0) return copy;
              const combined = (copy[idx].text + " " + fragText).trim();
              const parts = splitSentences(combined);
              if (parts.length === 0) {
                copy[idx] = { ...copy[idx], text: "" };
                return copy;
              }
              const lastIsComplete = /[.!?]["')\]]*$/.test(parts[parts.length - 1]);
              const finalized: ChatMessage[] = [];
              let pendingText = "";
              if (lastIsComplete) {
                for (const p of parts) finalized.push({ role: "assistant", text: p, turnId });
              } else {
                for (let i = 0; i < parts.length - 1; i++) {
                  finalized.push({ role: "assistant", text: parts[i], turnId });
                }
                pendingText = parts[parts.length - 1];
              }
              copy.splice(idx, 1, ...finalized, {
                role: "assistant",
                text: pendingText,
                pending: true,
                turnId,
              });
              return copy;
            });
          });
          break;
        }
        case "turn_complete": {
          const turnId = turnCounterRef.current;
          const finalResponse = e.final_response;
          // Wait for the last queued utterance to finish playing before
          // rebuilding from the canonical final_response — this keeps the
          // bubbles in sync with the audio rather than snapping forward.
          void latestSpeakRef.current.then(() => {
            if (ttsFirstPlaybackStartTsRef.current !== null) {
              const total = performance.now() - ttsFirstPlaybackStartTsRef.current;
              setMetrics((m) => ({ ...m, totalResponseMs: Math.round(total) }));
            }
            setMessages((m) => {
              if (!m.some((msg) => msg.role === "assistant" && msg.turnId === turnId)) {
                return m;
              }
              const copy = m.filter((msg) => !(msg.role === "assistant" && msg.turnId === turnId));
              const sentences = splitSentences(finalResponse);
              const bubbles: ChatMessage[] = sentences.length > 0
                ? sentences.map((s) => ({ role: "assistant", text: s, turnId }))
                : [{ role: "assistant", text: finalResponse, turnId }];
              return [...copy, ...bubbles];
            });
            setTurns((t) => {
              const copy = [...t];
              const lastIdx = copy.findIndex((x) => x.id === turnId);
              if (lastIdx >= 0) {
                copy[lastIdx] = { ...copy[lastIdx], complete: true };
              }
              return copy;
            });
            setTurnInProgress(false);
          });
          break;
        }
        case "mode_changed":
          setMode(e.mode);
          setPendingMode(null);
          break;
        case "modes":
          setAvailableModes(e.names);
          setMode(e.active);
          break;
        case "reset_ack":
          clearAllUiState();
          break;
        case "frontend_models":
          setFrontendModels(e.names);
          setActiveFrontendModel(e.active);
          break;
        case "frontend_model_changed":
          setActiveFrontendModel(e.name);
          setPendingFrontendModel(null);
          clearAllUiState();
          break;
        case "backend_models":
          setBackendProviders(e.providers);
          setActiveBackendProvider(e.active_provider);
          setActiveBackendModel(e.active_name);
          break;
        case "backend_model_changed":
          setActiveBackendProvider(e.provider);
          setActiveBackendModel(e.name);
          setPendingBackendProvider(null);
          setPendingBackendModel(null);
          clearAllUiState();
          break;
        case "device_capabilities":
          setDeviceCapabilities(e.capabilities);
          setActiveDevices(e.active);
          break;
        case "demo_mode":
          setDemoMode(e.mode);
          break;
        case "demo_mode_changed":
          setDemoMode(e.demo_mode);
          setPendingDemoMode(null);
          clearAllUiState();
          break;
        case "small_models":
          setSmallModels(e.names);
          setActiveSmallModel(e.active);
          break;
        case "small_model_changed":
          setActiveSmallModel(e.name);
          setPendingSmallModel(null);
          clearAllUiState();
          break;
        case "frontend_precision":
          setPrecision(e.precision);
          setAvailablePrecisions(e.available);
          setFrontendBackend(e.backend);
          break;
        case "precision_changed":
          setPrecision(e.precision);
          setPendingPrecision(null);
          clearAllUiState();
          break;
        case "device_changed":
          setActiveDevices((prev) =>
            prev ? { ...prev, [e.component]: e.device } : prev
          );
          setPendingDevices((prev) => {
            const next = { ...prev };
            delete next[e.component];
            return next;
          });
          break;
        case "error":
          setError(e.message);
          setTimeout(() => setError(null), 4000);
          setPendingBackendProvider((p) => (p !== null ? null : p));
          setPendingBackendModel((p) => (p !== null ? null : p));
          break;
        case "ready":
          setReady(true);
          break;
        case "log_started":
          console.info(`[log] new JSONL file: ${e.path}`);
          break;
        case "pong":
          break;
      }
    },
    [speak, cancelTTS, clearAllUiState]
  );

  const { send, connected } = useWebSocket(handleEvent);
  sendRef.current = send;

  useEffect(() => {
    if (!connected) setReady(false);
  }, [connected]);

  const sendUserMessage = (text: string) => {
    setMessages((m) => [...m, { role: "user", text }]);
    turnStartTsRef.current = performance.now();
    firstBackendThoughtSeenRef.current = false;
    frontendInferenceMsRef.current = [];
    frontendInferenceTokensRef.current = [];
    ttsLatenciesRef.current = [];
    ttsFirstPlaybackStartTsRef.current = null;
    ttsLatencyByFragmentRef.current.clear();
    ttsPlaybackStartByFragmentRef.current.clear();
    ttsPlaybackEndByFragmentRef.current.clear();
    voicedFirstSentRef.current = false;
    setMetrics({
      frontendFirstMs: null,
      backendFirstMs: null,
      voicedFirstMs: null,
      frontendAvgMs: null,
      frontendAvgTokPerSec: null,
      ttsAvgMs: null,
      totalResponseMs: null,
    });
    send({ type: "user_message", text });
  };

  const onModeChange = (next: string) => {
    if (next === mode || pendingMode === next) return;
    setPendingMode(next);
    send({ type: "set_mode", mode: next });
  };

  const onReset = () => {
    send({ type: "reset" });
  };

  const onFrontendModelChange = (name: string) => {
    if (!name || name === activeFrontendModel || pendingFrontendModel !== null) return;
    setPendingFrontendModel(name);
    clearAllUiState();
    send({ type: "set_frontend_model", name });
  };

  const onDemoModeChange = (next: DemoMode) => {
    if (next === demoMode || pendingDemoMode !== null) return;
    setPendingDemoMode(next);
    clearAllUiState();
    send({ type: "set_demo_mode", mode: next });
  };

  const onSmallModelChange = (name: string) => {
    if (!name || name === activeSmallModel || pendingSmallModel !== null) return;
    setPendingSmallModel(name);
    clearAllUiState();
    send({ type: "set_small_model", name });
  };

  const onPrecisionChange = (next: string) => {
    if (!next || next === precision || pendingPrecision !== null) return;
    setPendingPrecision(next);
    clearAllUiState();
    send({ type: "set_precision", precision: next });
  };

  const sendBackendModelChange = (provider: string, name: string) => {
    if (
      provider === activeBackendProvider &&
      name === activeBackendModel &&
      pendingBackendProvider === null &&
      pendingBackendModel === null
    ) {
      return;
    }
    setPendingBackendProvider(provider);
    setPendingBackendModel(name);
    clearAllUiState();
    send({ type: "set_backend_model", provider, name });
  };

  const onBackendProviderChange = (provider: string) => {
    if (!provider || pendingBackendProvider !== null || pendingBackendModel !== null) return;
    const models = backendProviders[provider] ?? [];
    if (models.length === 0) return;
    const first = models[0];
    sendBackendModelChange(provider, first);
  };

  const onBackendModelChange = (name: string) => {
    if (!name) return;
    const provider = pendingBackendProvider ?? activeBackendProvider;
    if (!provider) return;
    if (pendingBackendProvider !== null || pendingBackendModel !== null) return;
    sendBackendModelChange(provider, name);
  };

  const onDeviceChange = (component: DeviceComponent, device: Device) => {
    if (!activeDevices || activeDevices[component] === device) return;
    if (pendingDevices[component] !== undefined) return;
    setPendingDevices((prev) => ({ ...prev, [component]: device }));
    send({ type: "set_device", component, device });
  };

  useEffect(() => {
    if (!drawerOpen) return;
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === "Escape") setDrawerOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [drawerOpen]);

  return (
    <div className="app">
      <header>
        <button
          type="button"
          className="hamburger"
          onClick={() => setDrawerOpen((v) => !v)}
          aria-label="Open menu"
          aria-expanded={drawerOpen}
        >
          ☰
        </button>
        <h1>ConvFill</h1>
        <div className="controls">
          <span className={`status ${!connected ? "down" : ready ? "ok" : "loading"}`}>
            {!connected ? "disconnected" : ready ? "connected" : "loading"}
          </span>
          <div className="mode-toggle">
            {availableModes.map((m) => (
              <label key={m}>
                <input
                  type="radio"
                  checked={mode === m}
                  disabled={
                    turnInProgress || !ready || (m === "mcp" && demoMode === "frontend_only")
                  }
                  onChange={() => onModeChange(m)}
                />
                {MODE_LABELS[m] ?? m}
              </label>
            ))}
          </div>
          <label className="bubble-toggle">
            <input
              type="checkbox"
              checked={showAssistantBubbles}
              onChange={(ev) => setShowAssistantBubbles(ev.target.checked)}
            />
            Show responses
          </label>
          <button type="button" onClick={onReset} disabled={!ready}>
            Reset
          </button>
        </div>
      </header>
      {drawerOpen && (
        <div
          className="side-drawer-overlay"
          onClick={() => setDrawerOpen(false)}
          aria-hidden="true"
        />
      )}
      <aside
        className={`side-drawer${drawerOpen ? " open" : ""}`}
        aria-hidden={!drawerOpen}
      >
        <div className="side-drawer-header">
          <span>Menu</span>
          <button
            type="button"
            className="side-drawer-close"
            onClick={() => setDrawerOpen(false)}
            aria-label="Close menu"
          >
            ×
          </button>
        </div>
        <div className="side-drawer-section">
          <label className="side-drawer-label" htmlFor="demo-mode-select">
            Demo mode
          </label>
          <select
            id="demo-mode-select"
            className="frontend-model-select"
            value={pendingDemoMode ?? demoMode}
            disabled={!ready || turnInProgress || pendingDemoMode !== null}
            onChange={(ev) => onDemoModeChange(ev.target.value as DemoMode)}
          >
            {(Object.keys(DEMO_MODE_LABELS) as DemoMode[]).map((dm) => (
              <option key={dm} value={dm}>
                {DEMO_MODE_LABELS[dm]}
              </option>
            ))}
          </select>
        </div>
        {demoMode === "frontend_only" && (
          <div className="side-drawer-section">
            <label className="side-drawer-label" htmlFor="frontend-precision-select">
              Frontend precision
            </label>
            <div className="device-row-label">
              Backend: <strong>{frontendBackend ? frontendBackend.toUpperCase() : "—"}</strong>
            </div>
            <div className="device-row-label">
              Currently loaded in <strong>{precision ?? "—"}</strong>
            </div>
            {availablePrecisions.length > 1 && (
              <select
                id="frontend-precision-select"
                className="frontend-model-select"
                value={pendingPrecision ?? precision ?? ""}
                disabled={
                  !ready ||
                  turnInProgress ||
                  pendingPrecision !== null ||
                  availablePrecisions.length === 0
                }
                onChange={(ev) => onPrecisionChange(ev.target.value)}
              >
                {precision === null && <option value="">—</option>}
                {availablePrecisions.map((p) => (
                  <option key={p} value={p}>
                    {p === "int8" ? "int8 (MLX)" : p === "bf16" ? "bf16 (HuggingFace)" : p}
                  </option>
                ))}
              </select>
            )}
          </div>
        )}
        {demoMode === "convfill" && (
          <div className="side-drawer-section">
            <label className="side-drawer-label" htmlFor="frontend-model-select">
              Frontend model
            </label>
            <select
              id="frontend-model-select"
              className="frontend-model-select"
              value={activeFrontendModel ?? ""}
              disabled={
                !ready ||
                pendingFrontendModel !== null ||
                frontendModels.length === 0
              }
              onChange={(ev) => onFrontendModelChange(ev.target.value)}
            >
              {activeFrontendModel === null && <option value="">—</option>}
              {frontendModels.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </div>
        )}
        {(demoMode === "convfill" || demoMode === "backend_only") && (
          <div className="side-drawer-section">
            <div className="side-drawer-label">Backend model</div>
            {(() => {
              const providerValue = pendingBackendProvider ?? activeBackendProvider ?? "";
              const modelValue = pendingBackendModel ?? activeBackendModel ?? "";
              const providers = Object.keys(backendProviders);
              const models = providerValue ? (backendProviders[providerValue] ?? []) : [];
              const pendingActive =
                pendingBackendProvider !== null || pendingBackendModel !== null;
              const providerDisabled =
                !ready || pendingActive || providers.length === 0;
              const modelDisabled =
                !ready || pendingActive || !providerValue || models.length === 0;
              return (
                <>
                  <div className="device-row">
                    <label className="device-row-label" htmlFor="backend-provider-select">
                      Provider
                    </label>
                    <select
                      id="backend-provider-select"
                      className="frontend-model-select"
                      value={providerValue}
                      disabled={providerDisabled}
                      onChange={(ev) => onBackendProviderChange(ev.target.value)}
                    >
                      {providerValue === "" && <option value="">—</option>}
                      {providers.map((p) => (
                        <option key={p} value={p}>
                          {p}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="device-row">
                    <label className="device-row-label" htmlFor="backend-model-select">
                      Model
                    </label>
                    <select
                      id="backend-model-select"
                      className="frontend-model-select"
                      value={modelValue}
                      disabled={modelDisabled}
                      onChange={(ev) => onBackendModelChange(ev.target.value)}
                    >
                      {modelValue === "" && <option value="">—</option>}
                      {models.map((m) => (
                        <option key={m} value={m}>
                          {m}
                        </option>
                      ))}
                    </select>
                  </div>
                </>
              );
            })()}
          </div>
        )}
        {demoMode === "frontend_only" && (
          <div className="side-drawer-section">
            <label className="side-drawer-label" htmlFor="small-model-select">
              Small model
            </label>
            <select
              id="small-model-select"
              className="frontend-model-select"
              value={activeSmallModel ?? ""}
              disabled={
                !ready ||
                pendingSmallModel !== null ||
                smallModels.length === 0
              }
              onChange={(ev) => onSmallModelChange(ev.target.value)}
            >
              {activeSmallModel === null && <option value="">—</option>}
              {smallModels.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </div>
        )}
        {deviceCapabilities && activeDevices && (
          <div className="side-drawer-section">
            <div className="side-drawer-label">Devices</div>
            {DEVICE_COMPONENTS.map(({ key, label }) => {
              if (demoMode === "convfill" && key === "frontend") return null;
              const opts = deviceCapabilities[key] ?? [];
              const isPending = pendingDevices[key] !== undefined;
              const disabled =
                !ready ||
                turnInProgress ||
                isPending ||
                opts.length <= 1;
              const value = isPending
                ? (pendingDevices[key] as Device)
                : activeDevices[key];
              return (
                <div className="device-row" key={key}>
                  <label
                    className="device-row-label"
                    htmlFor={`device-select-${key}`}
                  >
                    {label}
                  </label>
                  <select
                    id={`device-select-${key}`}
                    className="frontend-model-select"
                    value={value}
                    disabled={disabled}
                    onChange={(ev) =>
                      onDeviceChange(key, ev.target.value as Device)
                    }
                  >
                    {opts.map((dev) => (
                      <option key={dev} value={dev}>
                        {DEVICE_LABELS[dev]}
                      </option>
                    ))}
                  </select>
                </div>
              );
            })}
          </div>
        )}
      </aside>
      {showTiming ? (
      <div className="metrics-bar" role="status" aria-live="polite">
        <div className="metric">
          <span className="metric-label">Frontend first response</span>
          <span className="metric-value">
            {metrics.frontendFirstMs !== null ? `${metrics.frontendFirstMs} ms` : "—"}
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">Backend first response</span>
          <span className="metric-value">
            {metrics.backendFirstMs !== null ? `${metrics.backendFirstMs} ms` : "—"}
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">Voiced first response</span>
          <span className="metric-value">
            {metrics.voicedFirstMs !== null ? `${metrics.voicedFirstMs} ms` : "—"}
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">Frontend avg inference</span>
          <span className="metric-value">
            {metrics.frontendAvgMs !== null ? `${metrics.frontendAvgMs} ms` : "—"}
            {metrics.frontendAvgTokPerSec !== null
              ? ` · ${metrics.frontendAvgTokPerSec} tok/s`
              : ""}
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">TTS avg latency</span>
          <span className="metric-value">
            {metrics.ttsAvgMs !== null ? `${metrics.ttsAvgMs} ms` : "—"}
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">Total response</span>
          <span className="metric-value">
            {metrics.totalResponseMs !== null ? `${metrics.totalResponseMs} ms` : "—"}
          </span>
        </div>
        <button
          type="button"
          className="panel-collapse"
          onClick={() => setShowTiming(false)}
          aria-label="Collapse timing panel"
          title="Collapse timing panel"
        >
          ▾
        </button>
      </div>
      ) : (
        <button
          type="button"
          className="metrics-bar collapsed"
          onClick={() => setShowTiming(true)}
          aria-label="Expand timing panel"
          title="Expand timing panel"
        >
          <span className="metric-label">Timing</span>
          <span className="panel-collapse" aria-hidden="true">▸</span>
        </button>
      )}
      {error && <div className="error-bar">{error}</div>}
      {(pendingDemoMode !== null || pendingSmallModel !== null) && (
        <div
          className="loading-overlay"
          role="alertdialog"
          aria-modal="true"
          aria-live="assertive"
          aria-label="Loading"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="loading-overlay-card">
            <div className="loading-overlay-spinner" aria-hidden="true" />
            <div className="loading-overlay-title">
              {pendingDemoMode === "frontend_only"
                ? "Switching to Frontend-only demo mode"
                : pendingDemoMode === "backend_only"
                ? "Switching to Backend-only demo mode"
                : pendingDemoMode === "convfill"
                ? "Switching to ConvFill demo mode"
                : pendingSmallModel !== null
                ? `Loading small model: ${pendingSmallModel}`
                : "Loading…"}
            </div>
            <div className="loading-overlay-detail">
              {pendingDemoMode === "frontend_only"
                ? "Resetting session state. Pick a small model from the dropdown to begin — weights download on first selection."
                : pendingDemoMode === "backend_only"
                ? "Resetting session state. Responses will be generated by the selected backend model (Claude / OpenAI / Gemini)."
                : pendingDemoMode === "convfill"
                ? "Restoring ConvFill frontend + backend systems…"
                : pendingSmallModel !== null
                ? "Downloading from HuggingFace Hub if not already cached. Weights land in ~/.cache/huggingface/hub. Please wait…"
                : "Please wait."}
            </div>
          </div>
        </div>
      )}
      <main className={showThoughts ? "" : "no-thoughts"}>
        {showThoughts ? (
          <ThoughtsPanel turns={turns} onCollapse={() => setShowThoughts(false)} />
        ) : (
          <button
            type="button"
            className="thoughts-rail"
            onClick={() => setShowThoughts(true)}
            aria-label="Expand thoughts panel"
            title="Expand thoughts panel"
          >
            <span className="rail-arrow" aria-hidden="true">▸</span>
            <span className="rail-label">Thoughts</span>
          </button>
        )}
        <ChatPanel
          messages={messages}
          showAssistantBubbles={showAssistantBubbles}
          onSend={sendUserMessage}
          disabled={
            turnInProgress ||
            !connected ||
            !ready ||
            pendingMode !== null ||
            pendingFrontendModel !== null ||
            pendingBackendProvider !== null ||
            pendingBackendModel !== null ||
            pendingDemoMode !== null ||
            pendingSmallModel !== null
          }
          placeholder={
            !connected
              ? "Disconnected"
              : !ready
              ? "Loading models…"
              : pendingDemoMode !== null
              ? `Switching demo mode to ${DEMO_MODE_LABELS[pendingDemoMode]}…`
              : pendingSmallModel !== null
              ? `Loading small model ${pendingSmallModel}…`
              : pendingMode !== null
              ? `Switching to ${pendingMode} mode…`
              : pendingFrontendModel !== null
              ? `Switching frontend model to ${pendingFrontendModel}…`
              : pendingBackendProvider !== null || pendingBackendModel !== null
              ? `Switching backend model to ${pendingBackendProvider ?? ""}/${pendingBackendModel ?? ""}…`
              : undefined
          }
        />
      </main>
      {(connected && !ready) ||
      pendingMode !== null ||
      pendingFrontendModel !== null ||
      pendingBackendProvider !== null ||
      pendingBackendModel !== null ? (
        <div className="loading-overlay" role="status" aria-live="polite">
          <div className="loading-card">
            <span className="spinner spinner-lg" aria-hidden="true" />
            <div className="loading-title">
              {!ready
                ? "Loading models"
                : pendingFrontendModel !== null
                ? `Switching frontend model to ${pendingFrontendModel}`
                : pendingBackendProvider !== null || pendingBackendModel !== null
                ? `Switching backend model to ${pendingBackendProvider ?? ""}/${pendingBackendModel ?? ""}`
                : `Switching to ${pendingMode} mode`}
            </div>
            <div className="loading-sub">
              {!ready
                ? "This can take 10–30 seconds on first connect."
                : pendingFrontendModel !== null
                ? "Loading the selected frontend model — please wait."
                : pendingBackendProvider !== null || pendingBackendModel !== null
                ? "Reconfiguring the backend model — please wait."
                : "Loading the model for this mode — please wait."}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function appendToActiveTurn(turns: TurnBlock[], item: TurnBlock["items"][number]): TurnBlock[] {
  if (turns.length === 0) return turns;
  const copy = [...turns];
  const last = copy[copy.length - 1];
  copy[copy.length - 1] = { ...last, items: [...last.items, item] };
  return copy;
}

function splitSentences(text: string): string[] {
  // Require two lowercase letters before the sentence-ending punctuation so
  // abbreviations like "Ph.D.", "U.S.", "Mr.", "e.g." don't trigger a split.
  // Real sentence endings ("...answer.", "...here!") still match because the
  // preceding word ends in lowercase letters.
  return text
    .split(/(?<=[a-z][a-z][.!?]["')\]]*)\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}
