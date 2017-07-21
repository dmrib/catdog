"""Dataset preparation for ML algorithms."""


import random
import glob
import numpy as np
import samples


def get_images_paths(pattern='../data/*.jpg', size='full', test_percentage=0.2,
                     fast_set_size=300):
    """Return path of images that matches glob pattern.

    Args:
        pattern (str): glob pattern.
        size (str): 'fast' for reduced dataset size, 'full' for entire data.
        test_percentage (float): test set size percentage.
        fast_set_size (int): size of the reduced set.
    Returns:
        training_paths (list): contains all training set image paths.
        test_paths (list): contains all test set image paths.

    """
    paths = glob.glob(pattern)
    random.shuffle(paths)

    if size == 'fast':
        paths = paths[:fast_set_size]

    test_set_size = int(len(paths)*test_percentage)
    test_paths = paths[:test_set_size]
    training_paths = paths[test_set_size:]

    return training_paths, test_paths


class Dataset():
    """Abstraction for a image dataset."""

    def __init__(self, paths, config, verbose):
        """Initializer.

        Args:
            paths (list): list containing images paths.
            config (dict): dictionary containing setup parameters.
            verbose (bool): display progress messages.
        Returns:
            None.

        """
        self.data = self.load_images(paths)
        self.dimensions = self.compute_default_size()
        self.load()

    def load(self):
        """Load dataset to memory.

        Args:
            path (str): path to images.
        Returns:
            None.

        """
        if self.verbose:
            print('   Applying filters...')
        self.apply_filters(self.config['filters'])

        if self.config['resize'] == 'True':
            if self.verbose:
                print('   Resizing dataset...')
            self.resize()

        if self.config['expand'] == 'True':
            if self.verbose:
                print('   Expanding dataset...')
            self.generate_sintetic_dataset()

    def load_images(self, paths):
        """Load every image in paths list to memory.

        Args:
            paths (list): list containing images paths.

        Returns:
            images (list): list of sample images.

        """
        images = []
        for image_path in paths:
            sample = samples.Sample(image_path)
            images.append(sample)

        return images

    def compute_default_size(self):
        """Compute the mean image width and height in the dataset.

        Args:
            None.
        Returns:
            width (int): mean image width.
            height (int): mean image height.

        """
        widths = []
        heights = []
        for sample in self.data:
            width, height = sample.image.shape[:2]
            widths.append(width)
            heights.append(height)

        return (int(np.mean(widths)), int(np.mean(heights)))

    def resize(self):
        """Resize entire dataset to given width and height.

        Args:
            None.
        Returns:
            None.

        """
        for image in self.data:
            image.resize(self.dimensions[0], self.dimensions[1])

    def apply_filters(self, filters):
        """Apply filters in the dataset images.

        Args:
            None.
        Returns:
            None.

        """
        if filters == 'grayscale':
            for image in self.data:
                image.grayscale()
        elif filters == 'thresholding':
            for image in self.data:
                image.adaptive_thresholding()
        elif filters == 'median':
            for image in self.data:
                image.median()
        elif filters == 'grayscale and median':
            for image in self.data:
                image.grayscale()
                image.median()
        elif filters == 'median and thresholding':
            for image in self.data:
                image.median()
                image.adaptive_thresholding()

    def generate_sintetic_dataset(self):
        """Artificially expand the dataset mirroing and adding noise to samples.

        Args:
            None.
        Returns:
            None.

        """
        sintetic_dataset = []
        for image in random.sample(self.data, int((len(self.data) * 0.3))):
            altered_image = image.mirrored('h')
            sintetic = samples.SinteticSample(altered_image, 'sin-' +
                                              image.name, image.label)
            sintetic_dataset.append(sintetic)

        for image in random.sample(self.data, int((len(self.data) * 0.3))):
            altered_image = image.mirrored('v')
            sintetic = samples.SinteticSample(altered_image, 'sin-' +
                                              image.name, image.label)
            sintetic_dataset.append(sintetic)

        for image in random.sample(self.sintetic_dataset,
                                   int((len(sintetic_dataset) * 0.3))):
            image.apply_noise()

        self.data.extend(sintetic_dataset)

    def compute_data_matrix(self):
        """Compute data matrix.

        Args:
            None.
        Returns:
            data_matrix (ndarray): data matrix.

        """
        data_matrix = []
        for sample in self.data:
            linear = sample.as_linear_array()
            data_matrix.append(linear)
        return np.array(data_matrix)

    def labels_array(self):
        """Create array with sequencial dataset labels.

        Args:
            None.
        Returns:
            labels (list): image labels in sequencial order.

        """
        labels = []
        for image in self.data:
            labels.append(image.label)
        return labels

    def show_dataset(self):
        """Display the entire dataset to user.

        Args:
            None.
        Returns:
            None.

        """
        for image in self.data:
            image.show()
