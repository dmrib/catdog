"""Dataset preparation for ML algorithms."""


import random
import numpy as np
import preprocessors
import samples


class Dataset():
    """Abstraction for a image dataset."""

    def __init__(self, paths, filters, use_mean):
        """Initializer.

        Args:
            paths (list): list containing images paths.
        Returns:
            None.

        """
        self.data = self.load_images(paths)
        self.dimensions = self.compute_default_size(use_mean)
        self.apply_filters(filters)

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

    def compute_default_size(self, use_mean=False):
        """Compute the maximum and mean image width and height in the dataset.

        Args:
            paths (list): paths of images to be labeled.
        Returns:
            width (int): maximum image width.
            height (int): maximum image height.

        """
        widths = []
        heights = []
        for sample in self.data:
            width, height = sample.image.shape[:2]
            widths.append(width)
            heights.append(height)

        if use_mean:
            return (int(np.mean(widths)), int(np.mean(heights)))
        else:
            return (min(widths), min(heights))

    def resize_dataset(self, dimensions):
        """Resize entire dataset to given width and height.

        Args:
            dimensions (tuple): dimensions of the resized dataset.
        Returns:
            None.

        """
        for image in self.data:
            image.resize(dimensions[0], dimensions[1])

    def apply_filters(self, filters):
        """Apply filters in the dataset images.

        Args:
            filters (list): name of the filters to be applied.
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

        self.resize_dataset(self.dimensions)

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

        for image in random.sample(self.data,
                                   int((len(sintetic_dataset) * 0.3))):
            image.with_noise()

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

    def show_dataset(self):
        """Display the entire dataset to user.

        Args:
            None.
        Returns:
            None.

        """
        for image in self.data:
            image.show()


if __name__ == '__main__':
    s = Dataset(preprocessors.get_images_paths()[:105],
                'thresholding', True)
    s.generate_sintetic_dataset()
    matrix = s.compute_data_matrix()
