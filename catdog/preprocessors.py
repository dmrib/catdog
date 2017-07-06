"""Image preprocessing."""


import cv2


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


if __name__ == '__main__':
    s = Sample('../data/cat.0.jpg')
    s.show()
