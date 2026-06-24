import { useCallback, useEffect, useRef } from "react";

/**
 * Streams Piper TTS audio from the backend (raw int16 LE PCM) and schedules
 * playback back-to-back via the Web Audio API. Successive `speak()` calls
 * queue: each new utterance starts when the previously-queued audio ends.
 */
export function useTTS(
  onLatency?: (ms: number, fragmentIndex: number) => void,
  onPlaybackStart?: (perfTimeMs: number, fragmentIndex: number) => void,
  onPlaybackEnd?: (perfTimeMs: number, fragmentIndex: number) => void,
) {
  const ctxRef = useRef<AudioContext | null>(null);
  const nextStartRef = useRef<number>(0);
  const activeSourcesRef = useRef<AudioBufferSourceNode[]>([]);
  const abortersRef = useRef<AbortController[]>([]);
  const pendingWaitsRef = useRef<Array<{ timerId: number; resolve: () => void }>>([]);
  // Serializes the *scheduling* phase of speak() calls (the fetch + read +
  // schedule-on-AudioContext loop). The next speak()'s scheduling waits for
  // this to resolve — but not for playback to finish — so fragment N+1's
  // fetch overlaps with fragment N's playback. Audio stays contiguous because
  // both fragments schedule against the same monotonic nextStartRef.
  const speakChainRef = useRef<Promise<void>>(Promise.resolve());
  // Counts spoken utterances so we can insert a longer "breath" pause every
  // few phrases. Reset on cancel() so each new turn starts fresh.
  const utteranceCountRef = useRef<number>(0);

  const getCtx = (): AudioContext => {
    if (!ctxRef.current) {
      const Ctor: typeof AudioContext =
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext ?? window.AudioContext;
      ctxRef.current = new Ctor();
    }
    if (ctxRef.current.state === "suspended") {
      void ctxRef.current.resume();
    }
    return ctxRef.current;
  };

  useEffect(() => {
    return () => {
      abortersRef.current.forEach((a) => a.abort());
      abortersRef.current = [];
      activeSourcesRef.current.forEach((s) => {
        try {
          s.stop();
        } catch {
          /* noop */
        }
      });
      activeSourcesRef.current = [];
      pendingWaitsRef.current.forEach(({ timerId, resolve }) => {
        window.clearTimeout(timerId);
        resolve();
      });
      pendingWaitsRef.current = [];
      ctxRef.current?.close().catch(() => undefined);
      ctxRef.current = null;
    };
  }, []);

  const doSpeak = useCallback(async (
    text: string,
    fragmentIndex: number,
    onSchedulingComplete?: () => void,
  ): Promise<void> => {
    const trimmed = text.trim();
    if (!trimmed) {
      onSchedulingComplete?.();
      return;
    }
    console.log(`[tts ${performance.now().toFixed(0)}ms] speak: ${JSON.stringify(trimmed)}`);
    let schedulingSignaled = false;
    const signalScheduled = () => {
      if (!schedulingSignaled) {
        schedulingSignaled = true;
        onSchedulingComplete?.();
      }
    };

    const ctrl = new AbortController();
    abortersRef.current.push(ctrl);

    const tttsStart = performance.now();
    let firstByteReported = false;

    try {
      const resp = await fetch(`/api/tts?text=${encodeURIComponent(trimmed)}`, {
        signal: ctrl.signal,
      });
      if (!resp.ok || !resp.body) {
        console.error("piper TTS request failed", resp.status);
        return;
      }
      const sr = parseInt(resp.headers.get("X-Sample-Rate") ?? "22050", 10);
      const ctx = getCtx();
      if (nextStartRef.current < ctx.currentTime) {
        nextStartRef.current = ctx.currentTime;
      }

      const INTER_UTTERANCE_GAP_S = 0.25;
      const reader = resp.body.getReader();
      let carry: Uint8Array | null = null;
      let firstChunkScheduled = false;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        if (!value || value.byteLength === 0) continue;
        if (!firstByteReported) {
          firstByteReported = true;
          onLatency?.(performance.now() - tttsStart, fragmentIndex);
        }

        let bytes: Uint8Array = value;
        if (carry) {
          const merged = new Uint8Array(carry.byteLength + bytes.byteLength);
          merged.set(carry, 0);
          merged.set(bytes, carry.byteLength);
          bytes = merged;
          carry = null;
        }
        const useLen = bytes.byteLength - (bytes.byteLength % 2);
        if (useLen < bytes.byteLength) {
          carry = bytes.slice(useLen);
        }
        if (useLen === 0) continue;

        // Copy into a fresh aligned ArrayBuffer (chunk byteOffset alignment is
        // not guaranteed, and Int16Array needs 2-byte alignment).
        const ab = new ArrayBuffer(useLen);
        new Uint8Array(ab).set(bytes.subarray(0, useLen));
        const int16 = new Int16Array(ab);
        const float32 = new Float32Array(int16.length);
        for (let i = 0; i < int16.length; i++) {
          float32[i] = int16[i] / 32768;
        }

        const buf = ctx.createBuffer(1, float32.length, sr);
        buf.copyToChannel(float32, 0);
        const src = ctx.createBufferSource();
        src.buffer = buf;
        src.connect(ctx.destination);
        const startAt = Math.max(nextStartRef.current, ctx.currentTime);
        src.start(startAt);
        if (!firstChunkScheduled) {
          firstChunkScheduled = true;
          const delayMs = Math.max(0, (startAt - ctx.currentTime) * 1000);
          onPlaybackStart?.(performance.now() + delayMs, fragmentIndex);
        }
        nextStartRef.current = startAt + buf.duration;
        activeSourcesRef.current.push(src);
        src.onended = () => {
          activeSourcesRef.current = activeSourcesRef.current.filter((s) => s !== src);
        };
      }
      // Report this utterance's end-of-audio on the perf clock *before* the
      // inter-utterance gap is added, so listeners can compute true playback
      // duration without the trailing breath.
      if (firstChunkScheduled) {
        const ctxNow = getCtx().currentTime;
        const endDelayMs = Math.max(0, (nextStartRef.current - ctxNow) * 1000);
        onPlaybackEnd?.(performance.now() + endDelayMs, fragmentIndex);
      }
      // Pad a short gap after this utterance so successive speak() calls
      // don't slam phrases together — gives audio room to breathe. Every
      // few phrases, take a longer "breath" pause to sound more natural.
      if (firstChunkScheduled) {
        utteranceCountRef.current += 1;
        const isBreathBoundary = utteranceCountRef.current % 3 === 0;
        nextStartRef.current += isBreathBoundary ? 0.9 : INTER_UTTERANCE_GAP_S;
      }
    } catch (err) {
      if ((err as { name?: string }).name !== "AbortError") {
        console.error("piper TTS stream failed", err);
      }
      return;
    } finally {
      abortersRef.current = abortersRef.current.filter((a) => a !== ctrl);
      // Release the schedule chain so the next speak()'s fetch can begin
      // overlapping with this utterance's playback.
      signalScheduled();
    }

    // Wait until this utterance's audio finishes playing on the AudioContext
    // clock. Successive speak() calls advance nextStartRef monotonically, so
    // each call's endTime is captured after its chunks are scheduled and
    // resolves in FIFO order with playback.
    const ctx = ctxRef.current;
    if (!ctx) return;
    const endTime = nextStartRef.current;
    const remainingMs = (endTime - ctx.currentTime) * 1000;
    if (remainingMs <= 0) return;
    await new Promise<void>((resolve) => {
      const entry = { timerId: 0, resolve };
      entry.timerId = window.setTimeout(() => {
        pendingWaitsRef.current = pendingWaitsRef.current.filter((e) => e !== entry);
        resolve();
      }, remainingMs);
      pendingWaitsRef.current.push(entry);
    });
  }, []);

  const speak = useCallback((text: string, fragmentIndex: number): Promise<void> => {
    // Capture prior speak's scheduling-complete promise. Our scheduling
    // doesn't start until that resolves; our playback runs in parallel with
    // the next speak()'s fetch+schedule.
    const prevScheduled = speakChainRef.current;
    let releaseChain: () => void = () => {};
    const ourScheduled = new Promise<void>((r) => { releaseChain = r; });
    speakChainRef.current = ourScheduled;

    return (async () => {
      try {
        await prevScheduled;
        await doSpeak(text, fragmentIndex, releaseChain);
      } finally {
        releaseChain();
      }
    })();
  }, [doSpeak]);

  const cancel = useCallback(() => {
    abortersRef.current.forEach((a) => a.abort());
    abortersRef.current = [];
    activeSourcesRef.current.forEach((s) => {
      try {
        s.stop();
      } catch {
        /* noop */
      }
    });
    activeSourcesRef.current = [];
    pendingWaitsRef.current.forEach(({ timerId, resolve }) => {
      window.clearTimeout(timerId);
      resolve();
    });
    pendingWaitsRef.current = [];
    speakChainRef.current = Promise.resolve();
    utteranceCountRef.current = 0;
    if (ctxRef.current) {
      nextStartRef.current = ctxRef.current.currentTime;
    }
  }, []);

  return { speak, cancel };
}
