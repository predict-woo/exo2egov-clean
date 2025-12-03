#!/bin/bash
# Training script for Exo2Ego with H2O dataset
# This uses a single exocentric view replicated 4 times

accelerate launch --config_file ./accelerate.yaml --main_process_port=8889 ./train_stage1.py \
    --config ./configs/train/stage1_h2o.yaml \
    --output_dir /cluster/work/cvg/students/andrye/exo2egov/outputs

