"""Dataset manipulation.

This module deals with operation within the dataset concerning the preparation
for experiments.
"""

import glob


def get_images_names(pattern='../data/*.jpg'):
    """Return image paths that matches glob pattern.

    Args:
        pattern (str): glob pattern.
    Returns:
        paths (list): contains all image paths.

    """
    paths = glob.glob(pattern)
    return paths
