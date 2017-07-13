"""Dataset preparation for ML algorithms."""

import os
import random
import cv2
import numpy as np
import preprocessors


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
            sample = Sample(image_path)
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
            sintetic = SinteticSample(altered_image, 'sin-' + image.name,
                                      image.label)
            sintetic_dataset.append(sintetic)

        for image in random.sample(self.data, int((len(self.data) * 0.3))):
            altered_image = image.mirrored('v')
            sintetic = SinteticSample(altered_image, 'sin-' + image.name,
                                      image.label)
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


class Sample():
    """Abstraction for a image sample."""

    def __init__(self, path):
        """Initializer.

        Args:
            path(str): sample image path.
        Returns:
            None.

        """
        self.path = path
        self.name = os.path.basename(path)
        self.label = os.path.basename(path)[:3]
        self.image = cv2.imread(path)

    def show(self):
        """Display image.

        Args:
            None.
        Returns:
            None.

        """
        cv2.imshow('Current Image', self.image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def resize(self, width, height):
        """Resize image for given width and height.

        Args:
            None.
        Returns:
            None.

        """
        self.image = cv2.resize(self.image, (height, width),
                                interpolation=cv2.INTER_AREA)

    def grayscale(self):
        """Convert image to grayscale.

        Args:
            None.
        Returns:
            None.

        """
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def adaptive_thresholding(self):
        """Apply global adaptative thresholding filter.

        Args:
            None.
        Returns:
            None.

        """
        self.grayscale()
        self.image = cv2.adaptiveThreshold(self.image, 255,
                                           cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, 5, 0)

    def median(self):
        """Apply median filter filter.

        Args:
            None.
        Returns:
            None.

        """
        self.image = cv2.medianBlur(self.image, 5)

    def mirrored(self, axis='h'):
        """Return horizontaly or verticaly (h or v) mirrored image.

        Args:
            axis (str): v for verticaly h for horizontaly.
        Returns:
            mirrored (np.ndarray): flipped image.

        """
        if axis == 'v':
            return cv2.flip(self.image, 0)
        else:
            return cv2.flip(self.image, 1)

    def with_noise(self):
        """Return image with gaussian noise.

        Args:
            None.
        Returns:
            None.

        """
        noise = int(0.3 * (self.image.shape[0] * self.image.shape[1]))
        salt = noise * 0.5
        for i in range(noise):
            y = random.randint(0, self.image.shape[1] - 1)
            x = random.randint(0, self.image.shape[0] - 1)
            if i <= salt:
                self.image[x][y] = 255
            else:
                self.image[x][y] = 0

    def as_linear_array(self):
        """Return image matrix as linear array.

        Args:
            None.
        Returns:
            (np.ndarray): image matrix in linear array form.

        """
        row = []
        for line in range(self.image.shape[0]):
            for column in range(self.image.shape[1]):
                row.append(self.image[line][column])
        return np.array(row)


class SinteticSample(Sample):
    """Abstraction for a artificially generated image."""

    def __init__(self, image, name, label):
        """Initializer.

        Args:
            image (np.ndarray): artificially generated image.
            name (str): name for the image.
            label (str): original image label.
        Returns:
            None.

        """
        self.image = image
        self.name = name
        self.label = label


if __name__ == '__main__':
    s = Dataset(preprocessors.get_images_paths()[:105],
                'thresholding', True)
    s.generate_sintetic_dataset()
    matrix = s.compute_data_matrix()
