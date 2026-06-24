#!/bin/bash
# data_preprocess.sh


python3 ../dataset_gen/dataset_preprocess.py \
  --user_tag user \
  --thoughts_tag thoughts \
  --responder_tag response \
  --base_data_dir ../your_base_data_dir \
  --output_path ../your_output_path_here \
  --include_history
# To include history, add: --include_history, otherwise it will not include history