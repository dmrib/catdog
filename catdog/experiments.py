"""Experiments on image classification."""


import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import preprocessors as pps
import dataset


def neural_nets(filters, layers, verbose=False):
    """Neural nets experiment."""
    # Dataset folder operations
    if verbose:
        print('Getting dataset images paths...')
    paths = pps.get_images_paths('../data/training/*.jpg')

    # Loading dataset
    if verbose:
        print('Loading images dataset...')
    input_data = dataset.Dataset(paths, filters=filters, use_mean=True)

    # Generating sintetic images
    if verbose:
        print('Generating sintetic dataset...')
    input_data.generate_sintetic_dataset()

    # Extract labels
    if verbose:
        print('Extracting labels...')
    labels = input_data.labels_array()

    # Input data matrix
    if verbose:
        print('Generating data matrix...')
    data_matrix = input_data.compute_data_matrix()

    # Compute PCA
    if verbose:
        print('Making principal component analysis...')
    pca = PCA(n_components=25)
    pca.fit_transform(data_matrix, y=labels)

    # Instantiate neural net model
    if verbose:
        print('Instantiating model...')
    classifier = MLPClassifier(random_state=3, hidden_layer_sizes=layers)

    # Training neural net model and computing score
    if verbose:
        print('Training and computing score...')
    shuffle_split = ShuffleSplit(n_splits=15, test_size=0.15, train_size=0.85,
                                 random_state=0)
    scores = cross_val_score(estimator=classifier, X=data_matrix,
                             y=labels, cv=shuffle_split,
                             n_jobs=-1)

    # Reporting results
    mean_score = np.mean(scores)
    std_deviation = np.std(scores)
    f = mean_score - std_deviation
    print('\nMean score: ', mean_score)
    print('Standard deviation: ', std_deviation)
    print('F: ', f)


def suport_vector(filters, kernel, verbose=False, degree=None):
    """Suport vector classifier experiment."""
    # Dataset folder operations
    if verbose:
        print('Getting dataset images paths...')
    paths = pps.get_images_paths('../data/training/*.jpg')

    # Loading dataset
    if verbose:
        print('Loading images dataset...')
    input_data = dataset.Dataset(paths, filters=filters, use_mean=True)

    # Generating sintetic images
    if verbose:
        print('Generating sintetic dataset...')
    input_data.generate_sintetic_dataset()

    # Extract labels
    if verbose:
        print('Extracting labels...')
    labels = input_data.labels_array()

    # Input data matrix
    if verbose:
        print('Generating data matrix...')
    data_matrix = input_data.compute_data_matrix()

    # Compute PCA
    if verbose:
        print('Making principal component analysis...')
    pca = PCA(n_components=25)
    pca.fit_transform(data_matrix, y=labels)

    # Instantiate suport vector classifier model
    if verbose:
        print('Instantiating model...')
    if kernel != 'poly':
        classifier = SVC(random_state=3, max_iter=500, kernel=kernel)
    else:
        classifier = SVC(random_state=3, max_iter=500, kernel=kernel,
                         degree=degree)

    # Training suport vector classifier model and computing score
    if verbose:
        print('Training and computing score...')
    shuffle_split = ShuffleSplit(n_splits=15, test_size=0.15, train_size=0.85,
                                 random_state=0)
    scores = cross_val_score(estimator=classifier, X=data_matrix,
                             y=labels, cv=shuffle_split,
                             n_jobs=-1)

    # Reporting results
    mean_score = np.mean(scores)
    std_deviation = np.std(scores)
    f = mean_score - std_deviation
    print('\nMean score: ', mean_score)
    print('Standard deviation: ', std_deviation)
    print('F: ', f)


def random_forest_classifier(filters, n_estimators, criterion, verbose=False):
    """Random forest tree classifier experiment."""
    # Dataset folder operations
    if verbose:
        print('Getting dataset images paths...')
    paths = pps.get_images_paths('../data/training/*.jpg')

    # Loading dataset
    if verbose:
        print('Loading images dataset...')
    input_data = dataset.Dataset(paths, filters=filters, use_mean=True)

    # Generating sintetic images
    if verbose:
        print('Generating sintetic dataset...')
    input_data.generate_sintetic_dataset()

    # Extract labels
    if verbose:
        print('Extracting labels...')
    labels = input_data.labels_array()

    # Input data matrix
    if verbose:
        print('Generating data matrix...')
    data_matrix = input_data.compute_data_matrix()

    # Compute PCA
    if verbose:
        print('Making principal component analysis...')
    pca = PCA(n_components=25)
    pca.fit_transform(data_matrix, y=labels)

    # Instantiate random forest classifier model
    if verbose:
        print('Instantiating model...')
    classifier = RandomForestClassifier(random_state=3, n_jobs=-1,
                                        n_estimators=n_estimators,
                                        criterion=criterion)

    # Training random forest classifier model and computing score
    if verbose:
        print('Training and computing score...')
    shuffle_split = ShuffleSplit(n_splits=15, test_size=0.15, train_size=0.85,
                                 random_state=0)
    scores = cross_val_score(estimator=classifier, X=data_matrix,
                             y=labels, cv=shuffle_split,
                             n_jobs=-1)

    # Reporting results
    mean_score = np.mean(scores)
    std_deviation = np.std(scores)
    f = mean_score - std_deviation
    print('\nMean score: ', mean_score)
    print('Standard deviation: ', std_deviation)
    print('F: ', f)
