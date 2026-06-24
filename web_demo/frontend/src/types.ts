export type Mode = "normal" | "rag" | "mcp";

export type DemoMode = "convfill" | "frontend_only" | "backend_only";

export type DeviceComponent = "frontend" | "reranker";
export type Device = "cpu" | "mps" | "cuda";

export type DeviceCapabilities = Record<DeviceComponent, Device[]>;
export type DeviceSettings = Record<DeviceComponent, Device>;

export type ServerEvent =
  | { type: "turn_start" }
  | { type: "rag_context"; text: string }
  | { type: "mcp_context"; text: string }
  | { type: "thought"; text: string }
  | { type: "silence" }
  | { type: "response_fragment"; text: string; fragment_index: number }
  | { type: "frontend_timing"; ms: number; tokens: number | null }
  | { type: "turn_complete"; final_response: string; thoughts: string[] }
  | { type: "mode_changed"; mode: Mode }
  | { type: "reset_ack" }
  | { type: "error"; message: string }
  | { type: "pong" }
  | { type: "ready" }
  | { type: "frontend_models"; names: string[]; active: string | null }
  | { type: "frontend_model_changed"; name: string }
  | { type: "device_capabilities"; capabilities: DeviceCapabilities; active: DeviceSettings }
  | { type: "device_changed"; component: DeviceComponent; device: Device }
  | { type: "backend_models"; providers: Record<string, string[]>; active_provider: string; active_name: string }
  | { type: "backend_model_changed"; provider: string; name: string }
  | { type: "demo_mode"; mode: DemoMode }
  | { type: "demo_mode_changed"; demo_mode: DemoMode }
  | { type: "small_models"; names: string[]; active: string | null }
  | { type: "small_model_changed"; name: string }
  | { type: "frontend_precision"; backend: "mlx" | "hf"; precision: string; available: string[] }
  | { type: "precision_changed"; precision: string }
  | { type: "log_started"; path: string };

export type ChatMessage = {
  role: "user" | "assistant";
  text: string;
  pending?: boolean;
  turnId?: number;
};

export type ThoughtItem =
  | { kind: "thought"; text: string }
  | { kind: "rag_context"; text: string }
  | { kind: "mcp_context"; text: string }
  | { kind: "silence" };

export type TurnBlock = {
  id: number;
  items: ThoughtItem[];
  complete: boolean;
};
