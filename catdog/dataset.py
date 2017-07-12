"""Dataset preparation for ML algorithms.""" 

import os
import random
import cv2
import numpy as np
import dataset


class Dataset():
    """Abstraction for a image dataset."""

    def __init__(self, paths, filters):
        """Initializer.

        Args:
            paths (list): list containing images paths.
        Returns:
            None.

        """
        self.data = self.load_images(paths)
        self.dimensions = self.compute_default_size(use_mean=True)
        self.apply_filters(filters)

    def load_images(self, paths):
        """Read every image in paths list.

        Args:
            paths (list): list containing images paths.

        Returns:
            images (list): list of loaded sample images.

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
        for image in self.data:
            width, height = image.shape
            widths.append(width)
            heights.append(height)

        if use_mean:
            return (int(np.mean(widths)), int(np.mean(heights)))
        else:
            return (min(widths), min(heights))

    def resize_dataset(self, dimensions):
        """Resize entire dataset to default width and height.

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
        if 'grayscale' in filters:
            for image in self.data:
                image.grayscale()
        elif 'thresholding' in filters:
            for image in self.data:
                image.adaptive_thresholding()
        elif 'median' in filters:
            for image in self.data:
                image.median()
        elif 'grayscale and median' in filters:
            for image in self.data:
                image.grayscale()
                image.median()
        elif 'median and thresholding' in filters:
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
        for image in self.data[0:int((len(self.data) * 0.3))]:
            sintetic = image.mirrored('h')
            name = os.path.basename(image.path)
            cv2.imwrite('../data/sintetic/' + name, sintetic)

        for image in self.data[0:-int((len(self.data) * 0.3))]:
            sintetic = image.mirrored('v')
            name = os.path.basename(image.path)
            cv2.imwrite('../data/sintetic/' + name, sintetic)

        paths = dataset.get_images_paths('../data/sintetic/*jpg')
        random.shuffle(paths)

        for image in self.data[0:int((len(self.data) * 0.3))]:
            sintetic = image.with_noise()
            name = os.path.basename(image.path)
            cv2.imwrite('../data/sintetic/' + name, sintetic)


class Sample():
    """Abstraction for a dataset image sample."""

    def __init__(self, path):
        """Initializer.

        Args:
            path(str): sample image path.
        Returns:
            None.

        """
        self.path = path
        self.image = cv2.imread(path)
        self.shape = self.image.shape[:2]

    def show(self):
        """Display image.

        Args:
            None.
        Returns:
            None.

        """
        cv2.imshow('Image', self.image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def resize(self, width, height):
        """Resize image for given width and height.

        Args:
            None.
        Returns:
            None.

        """
        self.image = cv2.resize(self.image, (width, height),
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
            axis (string): v for verticaly h for horizontaly.
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
            noise (np.ndarray): image with noise.

        """
        with_noise = self.image
        noise = int(0.3 * (with_noise.shape[0] * with_noise.shape[1]))
        salt = noise * 0.5
        for i in range(noise):
            y = random.randint(0, with_noise.shape[1] - 1)
            x = random.randint(0, with_noise.shape[0] - 1)
            if i <= salt:
                with_noise[x][y] = 255
            else:
                with_noise[x][y] = 0
        return with_noise


if __name__ == '__main__':
    s = Dataset(dataset.get_images_paths()[:10], ['median and thresholding'])
    s.generate_sintetic_dataset()
