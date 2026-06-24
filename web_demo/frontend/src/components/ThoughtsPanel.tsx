import { useEffect, useRef } from "react";
import type { TurnBlock } from "../types";

type Props = {
  turns: TurnBlock[];
  onCollapse?: () => void;
};

export default function ThoughtsPanel({ turns, onCollapse }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  return (
    <div className="thoughts-panel">
      <div className="panel-header">
        <h2>Thoughts</h2>
        {onCollapse && (
          <button
            type="button"
            className="panel-collapse"
            onClick={onCollapse}
            aria-label="Collapse thoughts panel"
            title="Collapse thoughts panel"
          >
            ◂
          </button>
        )}
      </div>
      {turns.length === 0 && <p className="placeholder">Thoughts will appear here as the assistant reasons.</p>}
      {turns.map((turn) => (
        <div key={turn.id} className={`turn-block ${turn.complete ? "complete" : "active"}`}>
          {turn.items.map((item, i) => {
            if (item.kind === "rag_context") {
              return (
                <div key={i} className="thought thought-rag">
                  <span className="tag">context</span>
                  <span className="text">{item.text}</span>
                </div>
              );
            }
            if (item.kind === "mcp_context") {
              return (
                <div key={i} className="thought thought-mcp">
                  <span className="tag">MCP</span>
                  <span className="text">{item.text}</span>
                </div>
              );
            }
            if (item.kind === "silence") {
              return (
                <div key={i} className="thought thought-silence">
                  <span className="text">… (silence)</span>
                </div>
              );
            }
            return (
              <div key={i} className="thought thought-plain">
                <span className="text">{item.text}</span>
              </div>
            );
          })}
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
