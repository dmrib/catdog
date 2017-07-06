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
        self.default_width, self.default_height = self.compute_default_size()

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

    def compute_default_size(self):
        """Compute the maximum image width and height in the dataset.

        Args:
            paths (list): paths of images to be labeled.
        Returns:
            width (int): maximum image width.
            height (int): maximum image height.

        """
        min_height = 1000000
        min_width = 1000000
        for image in self.data:
            width, height = image.shape
            if width < min_width:
                min_width = width
            if height < min_height:
                height = min_height

        return min_width, min_height


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


if __name__ == '__main__':
    s = Dataset(dataset.get_images_paths()[:100])
