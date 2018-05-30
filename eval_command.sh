#!/bin/sh
python3 evaluate_motchallenge.py \
    --mot_dir=/Users/mark/MOT/MOT16/train \
    --output_dir=/Users/mark/MOT/ds-origin-output-poi \
    --detection_dir=/Users/mark/MOT/resources/detections/MOT16_POI_train \
    --min_confidence=0.3 \
    --nn_budget=150 \
    --max_cosine_distance=0.2
#   --nms_max_overlap=0.9
