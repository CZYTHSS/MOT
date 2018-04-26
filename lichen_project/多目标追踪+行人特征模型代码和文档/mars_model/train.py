import numpy as np
import cv2
import argparse
import os

import tensorflow as tf

from train_features import create_image_trainer

def generate_training_data(sequence_dir):
    assert sequence_dir is not None, 'sequence directory does not exist!'

    img_dir = sequence_dir
    img_paths = [os.path.join(img_dir, img_name) for img_name in os.listdir(img_dir)]
    
    img_paths = []
    labels = []
    
    dir_names = sorted([int(dir_name) for dir_name in os.listdir(img_dir)])
    for i, dir_name in enumerate(dir_names):
        for img_name in os.listdir(os.path.join(img_dir, "%04d" % dir_name)):
            print("Frame %d/%d" % (dir_name, dir_names[-1]))
            img_path = os.path.join(img_dir, "%04d" % dir_name, img_name)

            label = [0] * len(dir_names)
            label[i] = 1
            labels.append(label)
            img_paths.append(img_path)
    return img_paths, labels

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    return parser.parse_args()


def main(argv=None):
    args = parse_args()
    #ckpt_path = 'model/model.ckpt-69000'
    ckpt_path=None # None for training from scratch, '/path' for training from a checkpoint
    image_shape = 128, 64, 3 # training image shape
    num_classes = 1259 # number of classes of training dataset(Mars)
    learning_rate_base = 0.01 # learning rate
    learning_rate_decay_interval = 7500 # learning rate decays every 7500 steps
    learning_rate_decay = 0.99 # decay rate of learning rate 
    epochs = 2 # number of training epochs
    batch_size = 32 # training batch size
    model_save_path = 'model' # model save path(model will be saved every 1000 steps)
    max_to_keep = 100 # max number of saved models
    log_file_path = 'log/train.log' # log saved file

    # create trainer for encoding image feature
    image_trainer =  create_image_trainer(
        image_shape, num_classes, epochs, batch_size, learning_rate_base, learning_rate_decay_interval, learning_rate_decay, ckpt_path, model_save_path, max_to_keep, log_file_path)

    # load training dataset information
    # 'image_paths' is image saved paths, image will be load when it is in current training batch
    # 'labels' is one-hot labels for images
    image_paths, labels = generate_training_data(args.sequence_dir)

    # excute training stage
    image_trainer(image_paths, labels)


if __name__ == '__main__':
    tf.app.run()
