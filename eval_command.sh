#!/bin/sh
python3 evaluate_motchallenge.py \
    --mot_dir=/Users/mark/MOT/MOT16/train \
    --output_dir=/Users/mark/MOT/ds_origin_output \
    --detection_dir=/Users/mark/MOT/resources/detections/MOT16_train \
    --min_confidence=0.3 \
    --nn_budget=100 
