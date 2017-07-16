"""Dataset files operations and helpers."""

import glob


def get_images_paths(pattern='../data/*.jpg'):
    """Return path of images that matches glob pattern.

    Args:
        pattern (str): glob pattern.
    Returns:
        paths (list): contains all image paths.

    """
    paths = glob.glob(pattern)
    return paths
