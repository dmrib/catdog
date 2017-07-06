"""Dataset manipulation."""

import glob
import os
import random


def get_images_names(pattern='../data/*.jpg'):
    """Return image paths that matches glob pattern.

    Args:
        pattern (str): glob pattern.
    Returns:
        paths (list): contains all image paths.

    """
    paths = glob.glob(pattern)
    return paths


def label_images(paths):
    """Create dictionary pairing image path and her label.

    Args:
        paths (list): paths of images to be labeled.
    Returns:
        image_labels (dict): dictionary with image path as key and label as
                             value.

    """
    image_labels = {}
    for path in paths:
        image_labels[path] = os.path.basename(path)[:3]

    return image_labels


def training_and_validation_set(paths):
    """Separates dataset in training and test set.

    Args:
        paths (list): paths of images to be labeled.
    Returns:
        training_set (list): list of paths of the training set images.
        validation_set (list): list of paths of the validation set images.

    """
    paths = paths[:]
    random.shuffle(paths)
    training_set = paths[:round(len(paths)/10)]
    validation_set = paths[round(len(paths)/10):]

    return training_set, validation_set
