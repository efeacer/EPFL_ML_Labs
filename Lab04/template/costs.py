# -*- coding: utf-8 -*-
"""A function to compute the cost."""

import numpy as np

def compute_error_vector(y, tx, w):
    return y - tx.dot(w)

def compute_mse(error_vector):
    return np.mean(error_vector ** 2) / 2

def compute_rmse(loss_mse):
    return np.sqrt(2 * loss_mse)
