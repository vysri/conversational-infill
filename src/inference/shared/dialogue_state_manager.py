import torch
import json
from pathlib import Path
from src.inference.shared.turn_state_manager import TurnStateManager
import re
import threading

class DialogueStateManager:
    def __init__(self, num_history_turns=0):
        self.transcript = []
        self.user_turns = []
        # This should be a list of lists, where each inner list is the assistant's thoughts for a turn.
        self.thoughts = [] 
        self.responses = []
        self.num_history_turns = num_history_turns
        self.lock = threading.Lock()

    # Guarded from multiple adds
    def update_user_state(self, user_turn):
            if len(self.user_turns) == len(self.responses):
                self.transcript.append(f"User: {user_turn}")
                self.user_turns.append(user_turn)

    def update_response_state(self, assistant_response):
            self.transcript.append(f"You: {assistant_response}")
            self.responses.append(assistant_response)
            assert(len(self.user_turns) == len(self.responses))

    # Must happen from the turn state manager in the correct format 
    # assistant_thoughts should be a list of strings
    def update_thoughts_state(self, assistant_thoughts):
            non_sil_thoughts = [thought for thought in assistant_thoughts if thought != "<sil>"]
            self.thoughts.append(non_sil_thoughts)
            assert(len(self.thoughts) == len(self.user_turns))

    def get_transcript(self) -> str:
            assert(len(self.thoughts) == len(self.responses))
            lines = []
            for i in range(len(self.responses)):
                # User turn
                lines.append(f"User: {self.user_turns[i]}")
                # Assistant thoughts as bullet points
                lines.append("You:")
                # print("THOUGHTS FOR TURN", self.thoughts)
                if len(self.thoughts) > 0:
                    for thought in self.thoughts[i]:
                        lines.append(f"{thought}")
            lines.append(f"User: {self.user_turns[-1]}")
            return "\n".join(lines)

    def get_history(self):
            # Get the history without the latest user turn and response, since those are not confirmed until the next turn
            confirmed_history_len = min(len(self.user_turns), len(self.responses))
            confirmed_user_history = self.user_turns[0:confirmed_history_len]
            confirmed_response_history = self.responses[0:confirmed_history_len]
            assert(len(confirmed_user_history) == len(confirmed_response_history))

            history = {"user_turns_history": [], "responses_history": []}
            num_user_turns = len(confirmed_user_history)
            start_idx = max(0, num_user_turns - self.num_history_turns)
            assert(len(confirmed_user_history) == len(confirmed_response_history))
            for i in range(start_idx, num_user_turns):
                history["user_turns_history"].append(f"{confirmed_user_history[i]}")
                history["responses_history"].append(f"{confirmed_response_history[i]}")
            return history

    def reset(self):
        with self.lock:
            self.transcript = []
            self.user_turns = []
            self.responses = []
            self.thoughts = []
