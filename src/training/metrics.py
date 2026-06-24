import torch
import torch.nn.functional as F

# Quick note, you can only log SCALAR metrics to pl_module.log_dict. 
# This should be good for any PERFORMANCE metrics you have
# So if you want to log something else (like a histogram, or a text sample), 
# you can do that in "on_validation_epoch_end" or "on_test_epoch_end" in the actual PL module

class Metrics:
    def __init__(self, pad_token_id=-100):
        self.pad_token_id = pad_token_id

    def compute_all(self, loss, logits, labels):
        return {
            "loss": loss,
            "ppl": loss.detach().exp(),
            "top5": self.top_k_accuracy(logits, labels, k=5),
        }

    def log(self, pl_module, metrics, prefix, on_step=False, on_epoch=True):
        log_dict = {}

        for k, v in metrics.items():
            name = f"{prefix}_{k}"

            if torch.is_tensor(v):
                v = v.detach()
                v = v.mean() if v.numel() > 1 else v
                v = v.item()

            log_dict[name] = v

        pl_module.log_dict(
            log_dict,
            prog_bar=True,
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=True,
        )

    def top_k_accuracy(self, logits, labels, k=5):
        topk = logits.topk(k, dim=-1).indices

        mask = labels != self.pad_token_id

        correct = topk.eq(labels.unsqueeze(-1)) & mask.unsqueeze(-1)
        correct = correct.any(dim=-1)

        total = mask.sum().clamp(min=1)

        return correct.sum().float() / total