# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np

def build_polynomial(x, degree):
    num_cols = x.shape[1] if len(x.shape) > 1 else 1
    augmented_x = np.ones((len(x), 1))
    for col in range(num_cols):
        for degree in range(1, degree + 1):
            if num_cols > 1:
                augmented_x = np.c_[augmented_x, np.power(x[ :, col], degree)]
            else:
                augmented_x = np.c_[augmented_x, np.power(x, degree)]
        if num_cols > 1 and col != num_cols - 1:
            augmented_x = np.c_[augmented_x, np.ones((len(x), 1))]
    return augmented_x