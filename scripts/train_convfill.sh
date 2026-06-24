#!/bin/bash
python src/training/finetune_convfill.py \
  # Config for the model and training parameters
  --config configs/convfill_frontend_configs/convfill_gemma3IT_270M_nd.json \
  --run_name my_experiment \
  --output_dir ./runs
