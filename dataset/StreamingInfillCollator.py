import torch
from torch.nn.utils.rnn import pad_sequence


class StreamingInfillCollator:
    def __init__(self, pad_token_id, label_pad_token_id=-100):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, batch):

        input_ids = [b["input_ids"] for b in batch]
        labels = [b["labels"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        full_inputs_without_last_span = [b["full_input_without_last_span"] for b in batch]  

        # ---- logging-only tensors ----
        input_ids_without_last_span = [
            b["input_ids_without_last_span"] for b in batch
        ]
        target_ids = [b["target_ids"] for b in batch]

        # ---- pad main training tensors ----
        input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
        )

        labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.label_pad_token_id,
        )

        attention_mask = pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        )

        return {
            # training
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,

            # logging only
            "input_ids_without_last_span": input_ids_without_last_span,
            "target_ids": target_ids,
            "full_input_without_last_span": full_inputs_without_last_span
        }