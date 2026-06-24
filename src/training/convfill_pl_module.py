import torch
import pytorch_lightning as pl
from transformers import get_scheduler
from src.training.metrics import Metrics


import torch
import pytorch_lightning as pl
from transformers import get_scheduler
from src.training.metrics import Metrics
import wandb


class ConvFillModule(pl.LightningModule):
    def __init__(self, model, tokenizer, config, total_steps):
        super().__init__()
        self.model = model

        # Additional code to support new loss function
        print(">> Model config:", self.model.config, flush=True)

        self.tokenizer = tokenizer
        self.cfg = config

        self.boundary_tokens = config["boundary_tokens"]
        self.total_steps = total_steps
        self.metric_manager = Metrics()

        # holds fixed logging examples
        self.example_batch = None

    ##################################################################
    # CORE STEP FUNCTION (TRAIN + VAL)
    ##################################################################
    def _step(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        # print(">>> [LOG] ATTENTION SHAPE Layer 0:", outputs.attentions[0].shape, flush=True)
        # print(">>> [LOG] ATTENTION SHAPE Layer 1:", outputs.attentions[1].shape, flush=True)
        return outputs.loss, outputs.logits

    ##################################################################
    # TRAIN
    ##################################################################
    def training_step(self, batch, batch_idx):
        loss, logits = self._step(batch)

        metrics = self.metric_manager.compute_all(
            loss, logits, batch["labels"]
        )

        self.metric_manager.log(
            self,
            metrics,
            prefix="train",
            on_step=True,
            on_epoch=True,
        )

        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True, on_step=True)

        return loss

    ##################################################################
    # VALIDATION
    ##################################################################
    ##################################################################

    def on_validation_start(self):
        if self.trainer.is_global_zero:
            self.example_batch = None


    ##################################################################
    # VALIDATION
    ##################################################################
    def validation_step(self, batch, batch_idx):
        loss, logits = self._step(batch)

        metrics = self.metric_manager.compute_all(
            loss, logits, batch["labels"]
        )

        self.metric_manager.log(
            self,
            metrics,
            prefix="val",
            on_step=False,
            on_epoch=True,
        )
        if self.trainer.is_global_zero and batch_idx == 0:
            print(f"Validation step examples being logged at step {self.global_step}...")
            # Capture one fresh example batch per validation run
            if self.example_batch is None:
                self.example_batch = {}

                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        self.example_batch[k] = v[:4].detach().cpu().clone()
                    elif isinstance(v, list):
                        self.example_batch[k] = v[:4]
                    else:
                        raise ValueError(
                            f"Unsupported type in batch for key {k}: {type(v)}"
                        )


    ##################################################################
    # LOGGING
    ##################################################################
    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        if self.trainer.is_global_zero:
            print(">>> Logging examples at epoch end...")

        batch = self.example_batch

        if self.trainer.is_global_zero and batch is not None:
            device = self.device
            tokenizer = self.tokenizer

            pad_id = tokenizer.pad_token_id

            inputs = batch["full_input_without_last_span"]
            target_ids = batch["target_ids"]

            preds, labels_out, inputs_out = [], [], []

            for i in range(len(inputs)):
                encoded = tokenizer(
                    inputs[i],
                    return_tensors="pt",
                    padding=False,
                    truncation=False
                )
                input_ids = encoded["input_ids"].to(device)

                lab_seq = target_ids[i]

                with torch.no_grad():
                    output = self.model.generate(
                        input_ids=input_ids,
                        max_new_tokens=128,
                        do_sample=False,
                        pad_token_id=pad_id
                    )

                gen_tokens = output[0][input_ids.shape[1]:]
                gen_tokens = gen_tokens[gen_tokens != pad_id]
                lab_seq = lab_seq[lab_seq != pad_id]

                preds.append(tokenizer.decode(gen_tokens, skip_special_tokens=False))
                labels_out.append(tokenizer.decode(lab_seq, skip_special_tokens=False))
                inputs_out.append(inputs[i])

            self.logger.log_table(
                key="examples",
                columns=["input", "prediction", "ground_truth"],
                data=list(zip(inputs_out, preds, labels_out))
            )

        self.example_batch = None
    ##################################################################
    # OPTIMIZER + SCHEDULER
    ##################################################################
    def configure_optimizers(self):
        training_cfg = self.cfg["training_config"]

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=training_cfg["learning_rate"],
            weight_decay=training_cfg["weight_decay"],
        )

        scheduler = get_scheduler(
            name=training_cfg["scheduler"],
            optimizer=optimizer,
            num_warmup_steps=training_cfg["warmup_steps"],
            num_training_steps=self.total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }