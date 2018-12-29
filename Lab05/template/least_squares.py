# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

def least_squares(y, tx):
    """
    Least squares regression using normal equations
    Args:
        y: labels
        tx: features
    Returns:
        (w, loss): (optimized weight vector for the model, 
            optimized final loss based on mean squared error)
    """
    coefficient_matrix = tx.T.dot(tx)
    constant_vector = tx.T.dot(y)
    w = np.linalg.solve(coefficient_matrix, constant_vector)
    error_vector = compute_error_vector(y, tx, w)
    loss = compute_mse(error_vector)
    return w, loss

def compute_error_vector(y, tx, w):
    """
    Computes the error vector that is defined as y - tx . w
    Args:
        y: labels 
        tx: features
        w: weight vector
    Returns:
        error_vector: the error vector defined as y - tx.dot(w)
    """
    return y - tx.dot(w)

def compute_mse(error_vector):
    """
    Computes the mean squared error for a given error vector.
    Args:
        error_vector: error vector computed for a specific dataset and model
    Returns:
        mse: numeric value of the mean squared error
    """
    return np.mean(error_vector ** 2) / 2