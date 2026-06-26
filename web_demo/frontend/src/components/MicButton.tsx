import { useEffect, useRef, useState } from "react";

type Props = {
  onTranscribed: (text: string) => void;
  disabled?: boolean;
};

export default function MicButton({ onTranscribed, disabled }: Props) {
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const recordingRef = useRef(false);
  const spaceHeldRef = useRef(false);

  const start = async () => {
    if (recording || disabled) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mr = new MediaRecorder(stream);
      chunksRef.current = [];
      mr.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
      };
      mr.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: mr.mimeType || "audio/webm" });
        const ext = (mr.mimeType || "audio/webm").includes("ogg") ? "ogg" : "webm";
        setBusy(true);
        try {
          const fd = new FormData();
          fd.append("file", blob, `recording.${ext}`);
          const res = await fetch("/api/transcribe", { method: "POST", body: fd });
          const data = await res.json();
          if (data.text) onTranscribed(data.text);
        } catch (err) {
          console.error("transcribe failed", err);
        } finally {
          setBusy(false);
          streamRef.current?.getTracks().forEach((t) => t.stop());
          streamRef.current = null;
        }
      };
      mr.start();
      recorderRef.current = mr;
      recordingRef.current = true;
      setRecording(true);
    } catch (err) {
      console.error("mic error", err);
    }
  };

  const stop = () => {
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
    recordingRef.current = false;
    setRecording(false);
  };

  useEffect(() => {
    const isEditable = (el: EventTarget | null) => {
      if (!(el instanceof HTMLElement)) return false;
      const tag = el.tagName;
      return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable;
    };

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code !== "Space" || e.repeat) return;
      if (isEditable(e.target)) return;
      if (disabled || busy) return;
      e.preventDefault();
      if (spaceHeldRef.current) return;
      spaceHeldRef.current = true;
      start();
    };

    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code !== "Space") return;
      if (!spaceHeldRef.current) return;
      e.preventDefault();
      spaceHeldRef.current = false;
      if (recordingRef.current) stop();
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [disabled, busy]);

  const onMouseDown = () => {
    if (disabled || busy) return;
    start();
  };

  const onMouseUp = () => {
    if (recordingRef.current) stop();
  };

  return (
    <button
      type="button"
      className={`mic ${recording ? "recording" : ""}`}
      disabled={disabled || busy}
      title="Hold spacebar or click to record"
      aria-live="polite"
      onMouseDown={onMouseDown}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseUp}
    >
      {busy ? "..." : recording ? "● rec" : "🎤 hold space"}
    </button>
  );
}
