"""provider-neutral MCP client wrapper used by ConvFillBackend in mcp mode.
"""

import asyncio
import json
import logging
import os
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SECRETS_FILE = _REPO_ROOT / "scripts" / "secrets.env"
_LOG_DIR = _REPO_ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_log = logging.getLogger("mcp_client")
if not _log.handlers:
    _log.setLevel(logging.DEBUG)
    _h = logging.FileHandler(_LOG_DIR / "mcp_client.log")
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _log.addHandler(_h)
    _log.propagate = False


def _dump(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        return repr(obj)


def _dbg(direction: str, msg: str) -> None:
    print(f"[mcp {direction}] {msg}", flush=True)


def _load_secrets() -> Dict[str, str]:
    """Parse simple KEY=VALUE lines from scripts/secrets.env."""
    out: Dict[str, str] = {}
    if not _SECRETS_FILE.exists():
        return out
    for raw_line in _SECRETS_FILE.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if (val.startswith("'") and val.endswith("'")) or (
            val.startswith('"') and val.endswith('"')
        ):
            val = val[1:-1]
        out[key] = val
    return out


_SECRETS = _load_secrets()


_TEXT_PRIMITIVES = {"string", "number", "integer", "boolean"}
_MAX_DESC_CHARS = 200
_MAX_PROP_DESC_CHARS = 120


def _normalize_type(t: Any) -> Optional[str]:
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        return non_null[0] if non_null else None
    return t if isinstance(t, str) else None


def _is_text_only_prop(prop_schema: Dict[str, Any]) -> bool:
    if not isinstance(prop_schema, dict):
        return False
    t = _normalize_type(prop_schema.get("type"))
    if t in _TEXT_PRIMITIVES:
        return True
    if t == "array":
        it = _normalize_type((prop_schema.get("items") or {}).get("type"))
        return it in _TEXT_PRIMITIVES
    return False


def _is_text_only_schema(input_schema: Dict[str, Any]) -> bool:
    props = (input_schema or {}).get("properties") or {}
    return all(_is_text_only_prop(v) for v in props.values())


def _shrink_schema(input_schema: Dict[str, Any]) -> Dict[str, Any]:
    props_in = (input_schema or {}).get("properties") or {}
    props_out: Dict[str, Any] = {}
    for name, p in props_in.items():
        if not isinstance(p, dict):
            continue
        t = _normalize_type(p.get("type"))
        entry: Dict[str, Any] = {}
        if t:
            entry["type"] = t
        if t == "array":
            it = _normalize_type((p.get("items") or {}).get("type"))
            if it:
                entry["items"] = {"type": it}
        desc = (p.get("description") or "").strip()
        if desc:
            entry["description"] = desc[:_MAX_PROP_DESC_CHARS]
        props_out[name] = entry
    out: Dict[str, Any] = {"type": "object", "properties": props_out}
    req = input_schema.get("required") if isinstance(input_schema, dict) else None
    if isinstance(req, list) and req:
        out["required"] = req
    return out


@dataclass
class MCPServerSpec:
    name: str
    transport: str
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict) 
    url: Optional[str] = None
    headers_env: Optional[str] = None 
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MCPServerSpec":
        return cls(
            name=d["name"],
            transport=d["transport"],
            command=d.get("command"),
            args=d.get("args", []),
            env=d.get("env", {}),
            url=d.get("url"),
            headers_env=d.get("headers_env"),
        )


class MCPToolHub:
    """Holds one MCP session per configured server. Sync-safe facade for
    threaded callers; all real I/O happens on `self._loop` in `self._thread`.
    """

    def __init__(self, server_specs: List[MCPServerSpec]):
        self.server_specs = server_specs
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._sessions: Dict[str, Any] = {}
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._connect_all()

    # ---- background loop ----

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    # ---- server connection ----

    def _connect_all(self) -> None:
        try:
            from mcp import ClientSession  # noqa: F401
        except ImportError:
            print("[mcp] WARNING: `mcp` package not installed; MCP mode will have zero tools.", flush=True)
            return

        for spec in self.server_specs:
            try:
                tools = self._submit(self._connect_one(spec))
                for raw_name, t in tools.items():
                    qname = f"{spec.name}__{raw_name}"
                    self._tools[qname] = {
                        "server": spec.name,
                        "raw_name": raw_name,
                        "description": t.get("description", ""),
                        "input_schema": t.get("input_schema", {"type": "object", "properties": {}}),
                    }
                print(f"[mcp] connected {spec.name} ({len(tools)} tools)", flush=True)
            except Exception as exc:
                print(f"[mcp] WARNING: failed to start {spec.name!r}: {exc!r}", flush=True)
                traceback.print_exc()

    async def _connect_one(self, spec: MCPServerSpec) -> Dict[str, Dict[str, Any]]:
        from contextlib import AsyncExitStack
        from mcp import ClientSession

        stack = AsyncExitStack()
        await stack.__aenter__()

        try:
            if spec.transport == "stdio":
                print(">>> RUNNING MCP ON stdio with command:", spec.command, spec.args, flush=True)
                from mcp import StdioServerParameters
                from mcp.client.stdio import stdio_client

                if not spec.command:
                    raise ValueError("stdio MCP server requires `command`")
                child_env = dict(os.environ)
                for inner_name, secret_key in spec.env.items():
                    val = _SECRETS.get(secret_key)
                    if val is None:
                        raise RuntimeError(
                            f"missing secret {secret_key!r} in {_SECRETS_FILE} for server {spec.name}"
                        )
                    child_env[inner_name] = val
                params = StdioServerParameters(command=spec.command, args=spec.args, env=child_env)
                read, write = await stack.enter_async_context(stdio_client(params))
            elif spec.transport == "http":
                print(" >>> RUNNING MCP ON HTTP with command:", spec.command, spec.args, flush=True)
                if not spec.url:
                    raise ValueError("http MCP server requires `url`")
                headers: Dict[str, str] = {}
                if spec.headers_env:
                    raw = _SECRETS.get(spec.headers_env)
                    if not raw:
                        raise RuntimeError(
                            f"missing secret {spec.headers_env!r} in {_SECRETS_FILE} for server {spec.name}"
                        )
                    headers["Authorization"] = raw
                try:
                    from mcp.client.streamable_http import streamablehttp_client
                    ctx = streamablehttp_client(spec.url, headers=headers)
                    read, write, _ = await stack.enter_async_context(ctx)
                except ImportError:
                    from mcp.client.sse import sse_client
                    read, write = await stack.enter_async_context(sse_client(spec.url, headers=headers))
            else:
                raise ValueError(f"unknown transport {spec.transport!r}")

            session = await stack.enter_async_context(ClientSession(read, write))
            _log.info("SEND %s initialize", spec.name)
            _dbg("SEND", f"{spec.name} initialize")
            init_result = await session.initialize()
            init_dump = _dump(getattr(init_result, "model_dump", lambda: init_result)())
            _log.info("RECV %s initialize -> %s", spec.name, init_dump)
            _dbg("RECV", f"{spec.name} initialize -> {init_dump}")
            _log.info("SEND %s list_tools", spec.name)
            _dbg("SEND", f"{spec.name} list_tools")
            tool_list = await session.list_tools()
            tool_names = [getattr(t, "name", str(t)) for t in tool_list.tools]
            _log.info("RECV %s list_tools -> %s", spec.name, _dump(tool_names))
            _dbg("RECV", f"{spec.name} list_tools -> {_dump(tool_names)}")
        except Exception:
            await stack.__aexit__(None, None, None)
            raise

        self._sessions[spec.name] = (session, stack)

        out: Dict[str, Dict[str, Any]] = {}
        dropped: List[str] = []
        for t in tool_list.tools:
            schema = getattr(t, "inputSchema", None) or {"type": "object", "properties": {}}
            if not _is_text_only_schema(schema):
                dropped.append(t.name)
                continue
            desc = (getattr(t, "description", "") or "").strip()[:_MAX_DESC_CHARS]
            out[t.name] = {
                "description": desc,
                "input_schema": _shrink_schema(schema),
            }
        if dropped:
            print(f"[mcp] {spec.name}: dropped {len(dropped)} non-text tools: {dropped}", flush=True)
            _log.info("%s dropped non-text tools: %s", spec.name, dropped)
        return out

    # ---- public ----

    def list_tools(self) -> List[Dict[str, Any]]:
        """Provider-neutral tool descriptors: [{name, description, input_schema}, ...]."""
        return [
            {
                "name": qname,
                "description": meta["description"],
                "input_schema": meta["input_schema"],
            }
            for qname, meta in self._tools.items()
        ]

    def call_tool(self, qualified_name: str, arguments: Dict[str, Any]) -> str:
        args_dump = _dump(arguments)
        _log.info("REQ call_tool %s arguments=%s", qualified_name, args_dump)
        _dbg("REQ", f"call_tool {qualified_name} arguments={args_dump}")
        meta = self._tools.get(qualified_name)
        if meta is None:
            msg = f"ERROR: unknown tool {qualified_name!r}"
            _log.warning("RESP %s -> %s", qualified_name, msg)
            _dbg("RESP", f"{qualified_name} -> {msg}")
            return msg
        server_name = meta["server"]
        raw_name = meta["raw_name"]
        entry = self._sessions.get(server_name)
        if entry is None:
            msg = f"ERROR: server {server_name!r} not connected"
            _log.warning("RESP %s -> %s", qualified_name, msg)
            _dbg("RESP", f"{qualified_name} -> {msg}")
            return msg
        session, _stack = entry
        try:
            out = self._submit(self._call(session, raw_name, arguments))
            _log.info("RESP %s -> %s", qualified_name, _dump(out))
            _dbg("RESP", f"{qualified_name} -> {_dump(out)}")
            return out
        except Exception as exc:
            msg = f"ERROR calling {qualified_name}: {exc!r}"
            _log.exception("RESP %s -> %s", qualified_name, msg)
            _dbg("RESP", f"{qualified_name} -> {msg}")
            return msg

    async def _call(self, session, raw_name: str, arguments: Dict[str, Any]) -> str:
        args_dump = _dump(arguments)
        _log.info("SEND call_tool %s arguments=%s", raw_name, args_dump)
        _dbg("SEND", f"call_tool {raw_name} arguments={args_dump}")
        try:
            result = await session.call_tool(raw_name, arguments=arguments)
        except Exception as exc:
            _log.exception("RECV call_tool %s raised %r", raw_name, exc)
            _dbg("RECV", f"call_tool {raw_name} raised {exc!r}")
            raise
        parts: List[str] = []
        for c in getattr(result, "content", []) or []:
            text = getattr(c, "text", None)
            if text:
                parts.append(text)
            else:
                parts.append(json.dumps(getattr(c, "model_dump", lambda: {"type": "unknown"})()))
        is_error = bool(getattr(result, "isError", False))
        joined = "\n".join(parts) if parts else ""
        _log.info(
            "RECV call_tool %s isError=%s content=%s",
            raw_name,
            is_error,
            _dump(joined),
        )
        _dbg("RECV", f"call_tool {raw_name} isError={is_error} content={_dump(joined)}")
        if is_error:
            return "ERROR: " + (joined or "tool call failed")
        return joined

    def close(self) -> None:
        async def _shutdown():
            for name, (_session, stack) in list(self._sessions.items()):
                try:
                    await stack.__aexit__(None, None, None)
                except Exception:
                    pass
            self._sessions.clear()
        try:
            self._submit(_shutdown())
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
