import numpy as np


def sum_squares(a, b):
    """
    Return the sum of squares of the difference between a and b, aka MSE
    """
    return np.sum([(a[i] - b[i])**2 for i in range(len(a))])


def euclid_d(x, y):
    """
    1D euclidean distance
    """
    return np.sqrt((x-y)**2)


def man_d(x, y):
    """
    1D manhattan distance
    """
    return np.abs((x-y))


def classif_d(x, y):
    """ 
    Function adapted to classification problems
    """
    return np.array(0 if x == y else 1)


def rmse(pred, ground_truth):
    import numpy as np
    return np.sqrt(sum_squares(pred, ground_truth)/len(ground_truth))


def accuracy(pred, ground_truth):
    import numpy as np
    return np.mean([a == b for a, b in zip(pred, ground_truth)])


def rbf_k(D, sigma=1):
    return np.exp(-(D**2)/sigma)
