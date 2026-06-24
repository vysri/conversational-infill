import json
import argparse
from pathlib import Path
import torch
import shutil

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger

from dataset.StreamingTurnDatasetContinued import StreamingTurnDataset
from dataset.StreamingInfillCollator import StreamingInfillCollator
from convfill_pl_module import ConvFillModule

import signal
import os

def handle_sigint(signum, frame):
    print("\n>>> Caught SIGINT (Ctrl+C). Shutting down gracefully...")
    raise KeyboardInterrupt

signal.signal(signal.SIGINT, handle_sigint)


# ----------------------------
# ARGS
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file.")

    parser.add_argument("--run_name", type=str, required=True, help="Name of the run.")

    parser.add_argument("--output_dir", type=str, default="./runs", help="Base directory for outputs.")

    parser.add_argument(
        "--resume_from_scratch",
        action="store_true",
        help="Ignore latest.ckpt and start fresh."
    )

    return parser.parse_args()


# ----------------------------
# DEVICES
# ----------------------------
def setup_devices(platform):
    if platform == "cpu":
        return "cpu", 1, "auto"

    if platform == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested but CUDA not available")

        n = torch.cuda.device_count()
        if n > 1:
            return "gpu", n, "ddp_find_unused_parameters_false"
        else:
            return "gpu", 1, "auto"

    raise ValueError(platform)


# ----------------------------
# W&B STEP FIX for resume
# ----------------------------
class WandbStepSync(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.logger is not None:
            trainer.logger.log_metrics({}, step=trainer.global_step)

# ----------------------------
# MAIN
# ----------------------------
def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    run_name = args.run_name

    resume_from_scratch = args.resume_from_scratch

    output_dir = Path(args.output_dir) / run_name

    if resume_from_scratch:
        if output_dir.exists():
            shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # save config
    with open(output_dir / "config.snapshot.json", "w") as f:
        json.dump(config, f, indent=2)

    # ----------------------------
    # W&B (FIXED)
    # ----------------------------
    wandb_logger = WandbLogger(
        project=config.get("wandb_project", "convfill"),
        name=run_name,
        save_dir=output_dir,
        log_model=False,
        resume="never" if resume_from_scratch else "allow",
        save_code=False,
    )

    # ----------------------------
    # MODEL
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    # model = AutoModelForCausalLM.from_pretrained(config["model_name"], attn_implementation="eager")
    model = AutoModelForCausalLM.from_pretrained(config["model_name"])

    print(">> PAD TOKEN ID:", tokenizer.pad_token_id, flush=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))

    print(">> AMMENDED PAD TOKEN ID:", tokenizer.pad_token_id, flush=True)

    if config.get("add_special_tokens", False):
        tokenizer.add_special_tokens(config["special_tokens"])
        model.resize_token_embeddings(len(tokenizer))

    # ----------------------------
    # DATA
    # ----------------------------
    with open(config["dataset_path"], "r") as f:
        turns = f.readlines()

    split_idx = int(config["training_split_percent"] * len(turns))

    train_dataset = StreamingTurnDataset(
        turns[:split_idx],
        config["boundary_tokens"],
        tokenizer,
        mode=config["dataset_mode"]
    )

    val_dataset = StreamingTurnDataset(
        turns[split_idx:],
        config["boundary_tokens"],
        tokenizer,
        mode=config["dataset_mode"]
    )

    collator = StreamingInfillCollator(
        pad_token_id=tokenizer.pad_token_id,
        label_pad_token_id=-100,
    )

    training_cfg = config["training_config"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
    )

    total_steps = len(train_loader) * training_cfg["num_epochs"]

    # ----------------------------
    # MODULE
    # ----------------------------
    print(">>> Initializing model and PL module...", flush=True)
    module = ConvFillModule(model, tokenizer, config, total_steps)

    # ----------------------------
    # CHECKPOINTING
    # ----------------------------
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="latest",
        every_n_train_steps=config["checkpoint_config"]["every_n_train_steps"],
        save_top_k=1,
        enable_version_counter=False,
        save_last=True,
    )

    class DebugCheckpointCallback(Callback):
        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            print(">>> CHECKPOINT SAVING TRIGGERED at step", trainer.global_step)

    # ----------------------------
    # DEVICES SETUP
    # ----------------------------
    accelerator, devices, strategy = setup_devices(config.get("platform", "gpu"))

    # ----------------------------
    # RESUME LOGIC
    # ----------------------------
    ckpt_file = output_dir / "latest.ckpt"

    if resume_from_scratch:
        print(">>> Starting fresh (ignoring checkpoints)")
        ckpt_path = None
    else:
        ckpt_path = str(ckpt_file) if ckpt_file.exists() else None
        print(">>> Resuming from:", ckpt_path if ckpt_path else "scratch")

    # ----------------------------
    # TRAINER
    # ----------------------------

    num_epochs = training_cfg["num_epochs"]
    accumulate_grad_batches = training_cfg["accumulate_grad_batches"]
    log_every_n_steps = config["logging_config"]["log_every_n_steps"]
    val_check_interval = config["training_config"]["val_check_interval"]

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=log_every_n_steps,
        callbacks=[checkpoint_callback, DebugCheckpointCallback(), WandbStepSync()],
        logger=wandb_logger,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        val_check_interval=val_check_interval,
    )

    # ----------------------------
    # TRAIN
    # ----------------------------
    try:
        trainer.fit(
            module,
            train_loader,
            val_loader,
            ckpt_path=ckpt_path,
        )
    except KeyboardInterrupt:
        print(">>> KeyboardInterrupt received...")

    finally:
        print(">>> Cleaning up trainer...")
        trainer.strategy.barrier() if hasattr(trainer.strategy, "barrier") else None

    # ----------------------------
    # FINAL SAVE
    # ----------------------------
    print(">>> Saving final model and tokenizer...", flush=True)
    module.model.save_pretrained(output_dir / "checkpoint-memory")
    tokenizer.save_pretrained(output_dir / "tokenizer-memory")


if __name__ == "__main__":
    main()