import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

FEATURE_NAMES = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols",
    "proanthocyanins", "color_intensity", "hue",
    "od280_od315_of_diluted_wines", "proline",
]

TARGET_NAMES = {0: "class_0", 1: "class_1", 2: "class_2"}


def load_data():
    """
    Load the Wine dataset and return features and target values.
    Returns:
        X (numpy.ndarray): The 13 features of the Wine dataset.
        y (numpy.ndarray): The target class labels (0, 1, 2).
    """
    wine = load_wine()
    X = wine.data
    y = wine.target
    return X, y


def split_data(X, y):
    """
    Split the data into training and testing sets.
    Args:
        X (numpy.ndarray): The features of the dataset.
        y (numpy.ndarray): The target values of the dataset.
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test