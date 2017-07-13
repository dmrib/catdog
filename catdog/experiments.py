"""Experiments on image classification."""

import preprocessors as pps
import dataset
from sklearn.decomposition import PCA


def one():
    """First experiment."""
    # Dataset file operations
    paths = pps.get_images_paths()[:20]
    training_set = paths
    
    # Loading dataset
    input_data = dataset.Dataset(training_set, filters='', use_mean=True)

    # Generating sintetic images
    input_data.generate_sintetic_dataset()

    # Input data matrix
    data_matrix = input_data.compute_data_matrix()

    # Compute PCA
    pca = PCA(n_components=25)
    pca.fit(data_matrix)

    print(pca.explained_variance_ratio_)


if __name__ == '__main__':
    one()
