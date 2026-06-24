from torch.utils.data import Dataset
import json
import torch


class StreamingTurnDataset(Dataset):
    def __init__(self, turns, boundary_tokens, tokenizer, mode="last"):
        self.examples = []
        self.boundary_tokens = boundary_tokens
        self.tokenizer = tokenizer
        self.mode = mode
        assert self.mode in ["full", "last"], "Invalid mode. Choose 'full_history' or 'last'."

        for i, turn in enumerate(turns):
            turn = json.loads(turn)
            user = turn.get("user", "")
            previous_user_turn = turn.get("previous_user_turn", "")
            previous_responder_turn = turn.get("previous_responder_turn", "")
            last_previous_responder_phrase = turn.get("last_previous_responder_phrase", "")
            
            if not user:
                raise ValueError(f"Turn {i} is missing 'user' field or it is empty.")

            if "grouped_responses" in turn:
                for step in turn["grouped_responses"]:
                    self.examples.append({
                        "user": user,
                        "previous_responses": step["previous_responses"],
                        "current_thought": step["current_thought"],
                        "current_response": step["current_response"],
                        "conv_history": {
                            "previous_user_turn": previous_user_turn,
                            "previous_responder_turn": previous_responder_turn,
                            "last_previous_responder_phrase": last_previous_responder_phrase
                        }
                    })

        ##############################################################
        # Tokenize once
        ##############################################################
        self.tokenized_cache = []
        tokens = self.boundary_tokens

        for i, ex in enumerate(self.examples):

            # ---------------------------------------------------------
            # 1. Build FULL STRING (NEW STRUCTURE)
            # ---------------------------------------------------------
            full_input = self.format_example(i)

            # ---------------------------------------------------------
            # 2. Define last_span
            # ---------------------------------------------------------
            last_span = ex["current_response"] + tokens["END"]

            # ---------------------------------------------------------
            # 3. Split prefix / target
            # ---------------------------------------------------------
            split_point = full_input.rfind(last_span)

            if split_point == -1:
                raise ValueError(f"last_span not found in full_input for example {i}")

            full_input_without_last_span = (
                full_input[:split_point] +
                full_input[split_point + len(last_span):]
            )

            # ---------------------------------------------------------
            # 4. Tokenize
            # ---------------------------------------------------------
            prefix_ids = self.tokenizer(
                full_input_without_last_span,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=False,
                padding=False,
            )["input_ids"][0]

            target_ids = self.tokenizer(
                last_span,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=False,
                padding=False,
            )["input_ids"][0]

            # ---------------------------------------------------------
            # 5. Build tensors
            # ---------------------------------------------------------
            input_ids = torch.cat([prefix_ids, target_ids], dim=0)

            labels = torch.cat([
                torch.full_like(prefix_ids, -100),
                target_ids
            ], dim=0)

            attention_mask = torch.ones_like(input_ids)

            self.tokenized_cache.append({
                "input_ids": input_ids,
                "input_ids_without_last_span": prefix_ids,
                "full_input_without_last_span": full_input_without_last_span,
                "labels": labels,
                "attention_mask": attention_mask,
                "target_ids": target_ids,
                # "current_thought": ex["current_thought"],
                # "current_response": ex["current_response"],
            })

    def __len__(self):
        return len(self.examples)

    def format_example(self, idx):
        ex = self.examples[idx]
        tokens = self.boundary_tokens

        user = ex["user"]
        prev = ex["previous_responses"].lstrip()
        thought = ex["current_thought"]
        response = ex["current_response"]
        history_user = ex["conv_history"]["previous_user_turn"]
        history_responder = ex["conv_history"]["previous_responder_turn"]
        history_responder = " ".join(history_responder)
        history_last_responder = ex["conv_history"]["last_previous_responder_phrase"]

        sequence = ""

        # ---- USER ----
        if history_user:
            sequence += f"{tokens['USER_START']}{history_user}{tokens['END']}"
        if history_responder:
            if self.mode == "full":
                sequence += f"{tokens['ASSN_START']}{history_responder}{tokens['END']}"
            else:
                sequence += f"{tokens['ASSN_START']}{history_last_responder}{tokens['END']}"
            
        sequence += f"{tokens['USER_START']}{user}{tokens['END']}"
        if prev:
            # print(f"Previous responses for example {idx}: '{prev}'")
            sequence += f"{tokens['ASSN_START']}{prev}{tokens['END']}"

        # ---- CURRENT STEP ----
        sequence += f"{tokens['KNOWLEDGE_START']}{thought}{tokens['END']}"
        sequence += f"{tokens['ASSN_START']}{response}{tokens['END']}"

        return sequence

    def __getitem__(self, idx):
        return self.tokenized_cache[idx]
    
if __name__ == "__main__":

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    special_tokens = {
        "USER_START": "<U>",
        "KNOWLEDGE_START": "<K>",
        "ASSN_START": "<A>",
        "END": "<|im_end|>\n",
        "SIL": "<|sil|>"
    }

    # ------------------------------------------------------------
    # REALISTIC DATASET (your provided examples)
    # ------------------------------------------------------------
    turns = [
        json.dumps({
            "turn_id": 0,
            "user": "I am a bit nervous about this optical coherence tomography scan. Can you explain what I need to do?",
            "grouped_responses": [
                {
                    "previous_responses": "",
                    "current_thought": "<sil>",
                    "current_response": "Oh, sure, I can definitely explain that."
                },
                {
                    "previous_responses": " Oh, sure, I can definitely explain that.",
                    "current_thought": "Scan captures detailed retina images.",
                    "current_response": "Basically, this scan is designed to capture detailed images of your retina."
                },
                {
                    "previous_responses": " Oh, sure, I can definitely explain that. Basically, this scan is designed to capture detailed images of your retina.",
                    "current_thought": "Sit in front of machine, place chin on chin rest.",
                    "current_response": "First, we will have you sit in front of the machine, and you will need to place your chin on the chin rest."
                }
            ],
            "conv_id": 0,
            "previous_user_turn": "",
            "previous_responder_turn": "",
            "last_previous_responder_phrase": ""
        }),

        json.dumps({
            "turn_id": 1,
            "user": "Alright, that sounds simple enough. Do I need to do anything with my forehead?",
            "grouped_responses": [
                {
                    "previous_responses": "",
                    "current_thought": "<sil>",
                    "current_response": "Yes, good question."
                },
                {
                    "previous_responses": " Yes, good question.",
                    "current_thought": "Forehead against support bar.",
                    "current_response": "You will also need to ensure your forehead is pressed gently against the support bar."
                },
                {
                    "previous_responses": " Yes, good question. You will also need to ensure your forehead is pressed gently against the support bar.",
                    "current_thought": "Keeps head steady during scan.",
                    "current_response": "This helps keep your head steady during the scan."
                }
            ],
            "conv_id": 0,
            "previous_user_turn": "I am a bit nervous about this optical coherence tomography scan. Can you explain what I need to do?",
            "previous_responder_turn": [
                "Oh, sure, I can definitely explain that.",
                "Basically, this scan is designed to capture detailed images of your retina.",
                "First, we will have you sit in front of the machine, and you will need to place your chin on the chin rest."
            ],
            "last_previous_responder_phrase": "Basically, this scan is designed to capture detailed images of your retina."
        }),

        json.dumps({
            "turn_id": 2,
            "user": "Got it. How long will I need to stay in that position?",
            "grouped_responses": [
                {
                    "previous_responses": "",
                    "current_thought": "<sil>",
                    "current_response": "Well, usually not too long."
                },
                {
                    "previous_responses": " Well, usually not too long.",
                    "current_thought": "Scan takes a few minutes.",
                    "current_response": "The scan itself only takes a few minutes."
                },
                {
                    "previous_responses": " Well, usually not too long. The scan itself only takes a few minutes.",
                    "current_thought": "Staying still is important.",
                    "current_response": "Staying still during that time is important for accurate results."
                }
            ],
            "conv_id": 0,
            "previous_user_turn": "Alright, that sounds simple enough. Do I need to do anything with my forehead?",
            "previous_responder_turn": [
                "Yes, good question.",
                "You will also need to ensure your forehead is pressed gently against the support bar.",
                "This helps keep your head steady during the scan."
            ],
            "last_previous_responder_phrase": "You will also need to ensure your forehead is pressed gently against the support bar."
        })
    ]

    # ------------------------------------------------------------
    # BUILD DATASET
    # ------------------------------------------------------------
    dataset = StreamingTurnDataset(turns, special_tokens, tokenizer)

    # ------------------------------------------------------------
    # STRUCTURE TESTS
    # ------------------------------------------------------------
    print("\n================= STRUCTURE VALIDATION =================\n")

    for i in range(len(dataset)):
        ex = dataset[i]

        print(f"\n================= EXAMPLE {i} =================")

        decoded = tokenizer.decode(ex["input_ids"], skip_special_tokens=False)
        print("[DECODED INPUT]\n", decoded)

        # sanity checks
        has_user = "<U>" in decoded
        has_assistant = "<A>" in decoded
        has_end = "<|im_end|>" in decoded

        # verify label alignment
        label_tokens = ex["labels"][ex["labels"] != -100]
        decoded_labels = tokenizer.decode(label_tokens, skip_special_tokens=False)

        print("\n[DECODED LABELS]")
        print(decoded_labels)

        print("\n----------------------------------------------------")