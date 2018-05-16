#!/bin/sh
python deep_sort_app.py \
    --sequence_dir=./MOT16/test/MOT16-03 \
    --detection_file=./resources/detections/MOT16_POI_test/MOT16-03.npy \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --output_file=./debug_output/output1.txt \
    --display=True \
    --nms_max_overlap=0.9
