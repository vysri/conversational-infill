import threading


class DialogueStateManagerStandalone:

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._user_turns: list[str] = []
        self._responses: list[str] = []

    def update_user_turn(self, text: str) -> None:
        with self._lock:
            if len(self._user_turns) == len(self._responses):
                self._user_turns.append(text)

    def update_response(self, text: str) -> None:
        with self._lock:
            assert len(self._user_turns) == len(self._responses) + 1, \
                "update_response called without a preceding update_user_turn"
            self._responses.append(text)

    def get_messages(self) -> list[dict]:
        with self._lock:
            msgs: list[dict] = []
            for i, u in enumerate(self._user_turns):
                msgs.append({"role": "user", "content": u})
                if i < len(self._responses):
                    msgs.append({"role": "assistant", "content": self._responses[i]})
            return msgs

    def get_transcript(self) -> str:
        with self._lock:
            lines: list[str] = []
            for i, u in enumerate(self._user_turns):
                lines.append(f"User: {u}")
                if i < len(self._responses):
                    lines.append(f"Assistant: {self._responses[i]}")
            return "\n".join(lines)

    def reset(self) -> None:
        with self._lock:
            self._user_turns = []
            self._responses = []
