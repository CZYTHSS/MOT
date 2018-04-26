#!/bin/sh
python evaluate_motchallenge.py \
    --mot_dir=./MOT16/train \
    --output_dir=./ds_origin_output \
    --detection_dir=./resource/detections/MOT16_train \
    --min_confidence=0.3 \
    --nn_budget=100 
