import numpy as np
import cv2
import argparse
import os

from generate_features import create_box_encoder

def generate_detections(encoder, sequence_dir, detection_txt, detection_npy):
    assert sequence_dir is not None, 'sequence directory does not exist!'
    if detection_txt is None:
        detection_txt = os.path.join(sequence_dir, "det/det.txt")

    print("generate numpy file of detections for %s" % sequence_dir)
    img_dir = os.path.join(sequence_dir, "img1")
    img_paths = {int(os.path.splitext(img_name)[0]): os.path.join(img_dir, img_name) for img_name in os.listdir(img_dir)}
    detections_wo_f = np.loadtxt(detection_txt, delimiter=',')
    detections_w_f = []
    
    frame_idxs = detections_wo_f[:, 0].astype(np.int)
    min_frame_idx = frame_idxs.min()
    max_frame_idx = frame_idxs.max()
    for frame_idx in xrange(min_frame_idx, max_frame_idx + 1):
        print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
        mask = frame_idxs == frame_idx
        rows = detections_wo_f[mask]

        if frame_idx not in img_paths:
            print("Frame %05d/%05d does not exist!" % (frame_idx, max_frame_idx))
            continue
        img = cv2.imread(img_paths[frame_idx], cv2.IMREAD_COLOR)
        features = encoder(img, rows[:, 2:6].copy())
        #print(features)
        detections_w_f += [np.r_[row, feature] for row, feature in zip(rows, features)]

        print("Frame %05d/%05d completed!" % (frame_idx, max_frame_idx))
    if detection_npy is None:
        detection_npy = detection_txt[:-4] + '.npy'
    np.save(detection_npy, np.asarray(detections_w_f), allow_pickle=False)
    print("Numpy file of detections with feature saved in \"%s\"" % detection_npy)

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_txt", help="Path to the txt file of detections",
        default=None, required=True)
    parser.add_argument(
        "--detection_npy", help="Path to the numpy file of detections with feature",
        default=None, required=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model = 'model/model.ckpt-69000' # checkpoint path
    image_shape = 128, 64, 3 # image patch shape, referencing number of nodes in input layer of encoder network
    encoder = create_box_encoder(model, image_shape)

    generate_detections(encoder, args.sequence_dir, args.detection_txt, args.detection_npy)
