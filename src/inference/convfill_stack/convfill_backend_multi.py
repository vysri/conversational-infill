import torch
import json
import re
import string
from jinja2 import Environment, FileSystemLoader
from src.inference.rag.retreive import RunRAG
from src.inference.shared.mcp_client import MCPServerSpec, MCPToolHub
import os


class ConvFillBackend:
    def __init__(self, dialogue_state_manager, prompt_template_path, model_backend, mode="normal",
                 task_specific_config=None, on_rag_context=None, on_mcp_context=None,
                 reranker_device: str = "cpu"):
        self.dialogue_state_manager = dialogue_state_manager
        self.model_backend = model_backend
        # Mode can be one of "normal", "rag", or "mcp"
        assert mode in ["normal", "rag", "mcp"], f"Unsupported mode: {mode}"
        self.mode = mode
        self.prompt_template_path = prompt_template_path
        self.on_rag_context = on_rag_context
        self.on_mcp_context = on_mcp_context
        if self.mode == "rag":
            assert task_specific_config is not None, "task_specific_config must be provided for RAG mode"
            self.rag_index = task_specific_config["rag_index"]
            self.rag_chunks = task_specific_config["rag_chunks"]
            self.reranker_model = task_specific_config["reranker_model"]
            self.embedding_model = task_specific_config["embedding_model"]

            self.retriever = RunRAG(self.rag_index, self.rag_chunks, embedding_model=self.embedding_model, reranker_model=self.reranker_model, device=reranker_device)
        elif self.mode == "mcp":
            assert task_specific_config is not None, "task_specific_config must be provided for MCP mode"
            server_specs = [MCPServerSpec.from_dict(d) for d in task_specific_config["mcp_servers"]]
            self.max_tool_iterations = int(task_specific_config.get("max_tool_iterations", 4))
            self.mcp_hub = MCPToolHub(server_specs)
            self.mcp_tools = self.mcp_hub.list_tools()
            print(f"[mcp] {len(self.mcp_tools)} tools available across {len(server_specs)} servers", flush=True)

    def emit_silence_token(self):
        yield "<sil>"

    def load_prompt_env(self):
        template_dir = os.path.dirname(self.prompt_template_path)
        template_file = os.path.basename(self.prompt_template_path)
        env = Environment(loader=FileSystemLoader(template_dir))
        return env, template_file

    def build_prompt_simple(self, transcript):
        env, template_file = self.load_prompt_env()
        template = env.get_template(template_file)
        output = template.render({"conversation": transcript})
        return output

    def build_prompt_rag(self, transcript, retrieved_context):
        env, template_file = self.load_prompt_env()
        template = env.get_template(template_file)
        output = template.render({
            "conversation": transcript,
            "rag_context": retrieved_context
        })
        return output

    def build_prompt_mcp(self, transcript):
        env, template_file = self.load_prompt_env()
        template = env.get_template(template_file)
        return template.render({"conversation": transcript})

    def backend_infer_simple(self):
        prompt = self.build_prompt_simple(
            self.dialogue_state_manager.get_transcript()
        )
        yield from self.model_backend.infer(prompt)

    def backend_infer_rag(self):
        print("Inferring with RAG-enhanced context retrieval...", flush=True)
        query = self.dialogue_state_manager.user_turns[-1]
        transcript = self.dialogue_state_manager.get_transcript()
        retrieved_context = self.retriever.rag_infer(query)
        if self.on_rag_context is not None:
            self.on_rag_context(retrieved_context)
        prompt = self.build_prompt_rag(
            transcript,
            retrieved_context=retrieved_context
        )
        yield from self.model_backend.infer(prompt)

    def backend_infer_mcp(self):
        print("Inferring with MCP tool access...", flush=True)
        transcript = self.dialogue_state_manager.get_transcript()
        prompt = self.build_prompt_mcp(transcript)
        events = self.model_backend.infer_with_tools(
            prompt,
            tools=self.mcp_tools,
            dispatch_tool=self.mcp_hub.call_tool,
            max_iterations=self.max_tool_iterations,
        )
        for kind, payload in events:
            if kind == "tool_call":
                if self.on_mcp_context is not None:
                    name = payload.get("name", "?")
                    args = payload.get("input", {}) or {}
                    arg_keys = ", ".join(sorted(args.keys())) if args else ""
                    label = f"Called {name}({arg_keys})" if arg_keys else f"Called {name}()"
                    self.on_mcp_context(label)
            elif kind == "sentence":
                yield payload

    def backend_infer(self):
        if self.mode == "normal":
            yield from self.backend_infer_simple()
        elif self.mode == "rag":
            yield from self.backend_infer_rag()
        elif self.mode == "mcp":
            yield from self.backend_infer_mcp()
