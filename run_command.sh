#!/bin/sh
python deep_sort_app.py \
    --sequence_dir=./MOT16/train/MOT16-09 \
    --detection_file=./resources/detections/MOT16_POI_train/MOT16-09.npy \
    --min_confidence=0.3 \
    --nn_budget=150 \
    --output_file=./debug_output/MOT16-09.txt \
    --display=' ' \
    --max_cosine_distance=0.3
    #--nms_max_overlap=0.9 \