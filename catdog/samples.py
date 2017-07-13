"""Implementation of images behavior."""


import random
import os
import numpy as np
import cv2


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
