import json
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import google.generativeai as genai
import anthropic


def _llm_send(provider: str, model: str, payload: dict) -> None:
    """Print every payload sent to an LLM. Dumps the full thing — no truncation."""
    try:
        body = json.dumps(payload, default=str, ensure_ascii=False)
    except Exception:
        body = repr(payload)
    print(f"[llm SEND][{provider}] model={model} {body}", flush=True)


class BackendInference:
    def __init__(self, api_key, model_name, model_mode):
        self.api_key = api_key
        self.model_name = model_name
        self.model_mode = model_mode

    def infer_text(self, prompt):
        if self.model_mode == "gemini":
            gemini = GeminiInference(self.api_key, self.model_name)
            yield from gemini.generate_text_stream(prompt)
        elif self.model_mode == "claude":
            claude = ClaudeInference(self.api_key, self.model_name)
            yield from claude.generate_text_stream(prompt)
        elif self.model_mode == "openai":
            openai_inf = OpenAIInference(self.api_key, self.model_name)
            yield from openai_inf.generate_text_stream(prompt)
        else:
            raise ValueError(f"Unsupported model mode: {self.model_mode}")

    def infer_text_with_tools(self, prompt, tools, dispatch_tool, max_iterations=4):
        if self.model_mode == "claude":
            claude = ClaudeInference(self.api_key, self.model_name)
            yield from claude.generate_text_with_tools(prompt, tools, dispatch_tool, max_iterations)
        elif self.model_mode == "gemini":
            gemini = GeminiInference(self.api_key, self.model_name)
            yield from gemini.generate_text_with_tools(prompt, tools, dispatch_tool, max_iterations)
        elif self.model_mode == "openai":
            openai_inf = OpenAIInference(self.api_key, self.model_name)
            yield from openai_inf.generate_text_with_tools(prompt, tools, dispatch_tool, max_iterations)
        else:
            raise ValueError(f"Unsupported model mode for tool use: {self.model_mode}")


class GeminiInference:
    def __init__(self, api_key, model_name="gemini-2.5-flash-lite-preview-09-2025"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name

    def generate_text_stream(self, prompt, max_tokens=1024):
        _llm_send("gemini", self.model_name, {"max_output_tokens": max_tokens, "contents": prompt})
        response = self.model.generate_content(
            contents=prompt,
            generation_config={"max_output_tokens": max_tokens},
            stream=True,
        )
        for chunk in response:
            text = chunk.text or ""
            if text:
                yield text

    def generate_text_with_tools(self, prompt, tools, dispatch_tool, max_iterations=4, max_tokens=1024):
        if not tools:
            for chunk in self.generate_text_stream(prompt, max_tokens=max_tokens):
                yield ("text", chunk)
            return

        function_decls = []
        for t in tools:
            function_decls.append(genai.protos.FunctionDeclaration(
                name=t["name"],
                description=t.get("description", "")[:1024],
                parameters=_to_gemini_schema(t.get("input_schema", {})),
            ))
        gemini_tools = [genai.protos.Tool(function_declarations=function_decls)]

        chat = self.model.start_chat(history=[])
        message = prompt

        for _ in range(max_iterations):
            _llm_send("gemini", self.model_name, {
                "max_output_tokens": max_tokens,
                "tools": [t["name"] for t in tools],
                "message": str(message),
            })
            response = chat.send_message(
                message,
                tools=gemini_tools,
                generation_config={"max_output_tokens": max_tokens},
            )
            candidate = response.candidates[0]
            parts = candidate.content.parts

            function_calls = [p.function_call for p in parts if getattr(p, "function_call", None) and p.function_call.name]
            if not function_calls:
                text = "".join(getattr(p, "text", "") or "" for p in parts)
                if text:
                    yield ("text", text)
                return

            function_responses = []
            for fc in function_calls:
                args = dict(fc.args) if fc.args else {}
                yield ("tool_call", {"name": fc.name, "input": args})
                result_text = dispatch_tool(fc.name, args)
                function_responses.append(genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=fc.name,
                        response={"result": result_text},
                    )
                ))
            message = genai.protos.Content(role="function", parts=function_responses)

        yield ("text", "I was unable to finish using my tools.")


class ClaudeInference:
    def __init__(self, api_key, model_name="claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name

    def generate_text_stream(self, prompt, max_tokens=1024):
        messages = [{"role": "user", "content": prompt}]
        _llm_send("claude", self.model_name, {"max_tokens": max_tokens, "messages": messages})
        with self.client.messages.stream(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                if text:
                    yield text

    def generate_text_with_tools(self, prompt, tools, dispatch_tool, max_iterations=4, max_tokens=1024):
        if not tools:
            for chunk in self.generate_text_stream(prompt, max_tokens=max_tokens):
                yield ("text", chunk)
            return

        claude_tools = [
            {
                "name": t["name"],
                "description": t.get("description", "")[:1024],
                "input_schema": t.get("input_schema", {"type": "object", "properties": {}}),
            }
            for t in tools
        ]

        messages = [{"role": "user", "content": prompt}]

        for _ in range(max_iterations):
            _llm_send("claude", self.model_name, {
                "max_tokens": max_tokens,
                "tools": claude_tools,
                "messages": messages,
            })
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                tools=claude_tools,
                messages=messages,
            )
            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        args = block.input or {}
                        yield ("tool_call", {"name": block.name, "input": args})
                        result_text = dispatch_tool(block.name, args)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                        })
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                continue

            text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
            if text:
                yield ("text", text)
            return

        yield ("text", "I was unable to finish using my tools.")


class OpenAIInference:
    def __init__(self, api_key, model_name="gpt-4o-mini"):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate_text_stream(self, prompt, max_tokens=1024):
        messages = [{"role": "user", "content": prompt}]
        _llm_send("openai", self.model_name, {"max_completion_tokens": max_tokens, "messages": messages})
        stream = self.client.chat.completions.create(
            model=self.model_name,
            max_completion_tokens=max_tokens,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None) or ""
            if text:
                yield text

    def generate_text_with_tools(self, prompt, tools, dispatch_tool, max_iterations=4, max_tokens=1024):
        if not tools:
            for chunk in self.generate_text_stream(prompt, max_tokens=max_tokens):
                yield ("text", chunk)
            return

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", "")[:1024],
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]

        messages = [{"role": "user", "content": prompt}]

        for _ in range(max_iterations):
            _llm_send("openai", self.model_name, {
                "max_completion_tokens": max_tokens,
                "messages": messages,
                "tools": openai_tools,
            })
            response = self.client.chat.completions.create(
                model=self.model_name,
                max_completion_tokens=max_tokens,
                messages=messages,
                tools=openai_tools,
            )
            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in tool_calls
                    ],
                })
                for tc in tool_calls:
                    import json as _json
                    try:
                        args = _json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except _json.JSONDecodeError:
                        args = {}
                    yield ("tool_call", {"name": tc.function.name, "input": args})
                    result_text = dispatch_tool(tc.function.name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    })
                continue

            text = msg.content or ""
            if text:
                yield ("text", text)
            return

        yield ("text", "I was unable to finish using my tools.")


def _to_gemini_schema(schema):
    """Best-effort conversion of a JSON schema dict into a Gemini Schema proto.

    Gemini accepts a restricted subset (object/string/number/integer/boolean/array).
    Unknown types are downgraded to string.
    """
    if not isinstance(schema, dict):
        return genai.protos.Schema(type=genai.protos.Type.OBJECT)
    t = (schema.get("type") or "object").lower()
    type_map = {
        "object": genai.protos.Type.OBJECT,
        "string": genai.protos.Type.STRING,
        "number": genai.protos.Type.NUMBER,
        "integer": genai.protos.Type.INTEGER,
        "boolean": genai.protos.Type.BOOLEAN,
        "array": genai.protos.Type.ARRAY,
    }
    proto = genai.protos.Schema(type=type_map.get(t, genai.protos.Type.STRING))
    if schema.get("description"):
        proto.description = schema["description"][:1024]
    if t == "object":
        for k, v in (schema.get("properties") or {}).items():
            proto.properties[k].CopyFrom(_to_gemini_schema(v))
        for req in schema.get("required", []) or []:
            proto.required.append(req)
    elif t == "array":
        items = schema.get("items") or {"type": "string"}
        proto.items.CopyFrom(_to_gemini_schema(items))
    return proto
