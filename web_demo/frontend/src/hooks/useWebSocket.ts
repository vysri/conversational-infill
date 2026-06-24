import { useEffect, useRef, useState } from "react";
import type { ServerEvent } from "../types";

type Sender = (msg: object) => void;

export function useWebSocket(onEvent: (e: ServerEvent) => void): { send: Sender; connected: boolean } {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const handlerRef = useRef(onEvent);
  handlerRef.current = onEvent;

  useEffect(() => {
    const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);
    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data) as ServerEvent;
        handlerRef.current(data);
      } catch (err) {
        console.error("Bad WS message", err);
      }
    };

    return () => {
      ws.close();
    };
  }, []);

  const send: Sender = (msg) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  };

  return { send, connected };
}
