import { useEffect, useRef, useState } from "react";
import type { ChatMessage } from "../types";
import MicButton from "./MicButton";

type Props = {
  messages: ChatMessage[];
  onSend: (text: string) => void;
  disabled?: boolean;
  placeholder?: string;
  showAssistantBubbles?: boolean;
};

export default function ChatPanel({
  messages,
  onSend,
  disabled,
  placeholder,
  showAssistantBubbles = true,
}: Props) {
  const [draft, setDraft] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const submit = () => {
    const text = draft.trim();
    if (!text || disabled) return;
    onSend(text);
    setDraft("");
  };

  return (
    <div className="chat-panel">
      <h2>Chat</h2>
      <div className="messages">
        {messages.map((m, i) => {
          if (m.role === "assistant" && !showAssistantBubbles) return null;
          return (
            <div key={i} className={`bubble ${m.role}`}>
              {m.text}
              {m.pending && <span className="cursor">▍</span>}
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>
      <div className="composer">
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          placeholder={placeholder ?? (disabled ? "Waiting..." : "Type or hold mic to talk")}
          disabled={disabled}
          rows={2}
        />
        <button type="button" onClick={submit} disabled={disabled || !draft.trim()}>
          Send
        </button>
        <MicButton
          disabled={disabled}
          onTranscribed={(text) => {
            const cleaned = text.trim();
            if (cleaned) onSend(cleaned);
          }}
        />
      </div>
    </div>
  );
}
