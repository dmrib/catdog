"""Image preprocessing."""


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
                image.show()
        elif 'median and thresholding' in filters:
            for image in self.data:
                image.median()
                image.adaptive_thresholding()

        self.resize_dataset(self.dimensions)


class Sample():
    """Abstraction for a dataset image sample."""

    def __init__(self, path):
        """Initializer.

        Args:
            path(str): sample image path.
        Returns:
            None.

        """
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


if __name__ == '__main__':
    s = Dataset(dataset.get_images_paths()[:6], ['grayscale and median'])
    cv2.waitKey()
    cv2.destroyAllWindows()
