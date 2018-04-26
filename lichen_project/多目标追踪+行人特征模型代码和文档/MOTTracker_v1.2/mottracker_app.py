# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np
import time as time

from MOTTracker import Detection
from MOTTracker import MOTTracker


def gather_sequence_info(sequence_dir, detection_mat):
    #Gather sequence information

    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {int(os.path.splitext(f)[0]): os.path.join(image_dir, f) 
                       for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_mat is not None:
        detections = np.load(detection_mat)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "detection_mat": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    # Create detections from given detection matrix at frame_idx
    # detection_mat
    #   detection_mat[:, :10], MOT standard detections
    #   detection_mat[:, 10:], detections feature

    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        tlwh, confidence, feature = row[2:6], row[6], row[10:]
        if tlwh[3] < min_height:
            continue
        detection_list.append(Detection(tlwh, confidence, feature))
    return detection_list


def run(sequence_dir, detection_mat, output, min_confidence, max_cosine_distance,
        nn_budget, budget_association_threshold, budget_detection_threshold, matching_time_depth):
    # Run MOTTracker on a sequence.

    seq_info = gather_sequence_info(sequence_dir, detection_mat)
    tracker = MOTTracker(nn_budget=nn_budget, matching_time_depth=matching_time_depth, n_init=3, max_cosine_distance=max_cosine_distance)
    results = []

    frame_idx = seq_info["min_frame_idx"]
    last_idx = seq_info["max_frame_idx"]
    for frame_idx in range(seq_info["min_frame_idx"], seq_info["max_frame_idx"] + 1):
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(seq_info["detection_mat"], frame_idx)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Update tracker.
        t1 = time.time()
        tracker.predict()
        tracker.update(detections, budget_association_threshold=budget_association_threshold, budget_detection_threshold=budget_detection_threshold)
        t2 = time.time()
        print("time for track: %f s" % (t2-t1))
        # Store results.
        for track in tracker.tracks:
            if not track.state == 1 or track.time_since_update > 1:
                continue
            bbox = track.kf.state[:4].copy()
            bbox[2] *= bbox[3]
            bbox[:2] -= bbox[2:] / 2
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Store results.
    f = open(output, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="MOTTracker")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_mat", help="Path to detection matrix.", default=None,
        required=True)
    parser.add_argument(
        "--output", help="Path to the tracking output file. This file will",
        default="/tmp.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold.",
        default=0.8, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for appearance association distance metric.",
        type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the track features budget.", 
        type=int, default=None)
    parser.add_argument(
        "--budget_association_threshold", help="New detection feature whose assosiation cost "
        " below this threshold will be added into budget.", type=float, default=None)
    parser.add_argument(
        "--budget_detection_threshold", help="New detection feature whose detection score "
        " over this threshold will be added into budget.", type=float, default=None)
    parser.add_argument(
        "--matching_time_depth", help="Window size of time for taking association.", 
        type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.sequence_dir, args.detection_mat, args.output, args.min_confidence,
        args.max_cosine_distance, args.nn_budget, args.budget_association_threshold,
        args.budget_detection_threshold, args.matching_time_depth)
