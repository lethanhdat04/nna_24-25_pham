import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42

def generate_1d_convex(n_samples=1000, interval=(0, 1), seed=None, ratio=0.2,
                       func=lambda x: np.square(x)):
    """
    Generate datasets for a 1D convex function approximation task.

    :param int n_samples: 
        The total number of samples to generate. Default is 1000.
    :param tuple interval:
        Tuple (a, b) specifying the range of input values.
    :return: 
        A tuple containing the training and validation splits:
        (X_train, X_val, y_train, y_val)
    :rtype: tuple of numpy.ndarray
    """
    a, b = interval
    X = np.linspace(a, b, n_samples).reshape(-1, 1)
    y = func(X)  # Example of a convex function
    seed = seed if seed is not None else SEED
    return train_test_split(X, y, test_size=ratio, random_state=seed)