"""Dataset manipulation.

This module deals with operation within the dataset concerning the preparation
for experiments.
"""

import glob
import os


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
