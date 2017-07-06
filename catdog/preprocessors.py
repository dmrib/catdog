"""Image preprocessing."""


import cv2
import dataset


class Dataset():
    """Abstraction for a image dataset."""

    def __init__(self, paths):
        """Initializer.

        Args:
            paths (list): list containing images paths.
        Returns:
            None.

        """
        self.data = self.load_images(paths)
        self.default_width, self.default_height = self.compute_defaults()

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


if __name__ == '__main__':
    s = Dataset(dataset.get_images_paths())
