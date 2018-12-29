# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np

def expand_row(row_of_x, degree, num_cols):
    expanded_row = np.array([])
    if num_cols > 1:
        for i in range(num_cols):
            expanded_row = np.append(expanded_row, np.fromfunction(lambda j: row_of_x[i] ** j, (degree + 1, )))
    else:
        expanded_row = np.fromfunction(lambda i: row_of_x ** i, (degree + 1, ))
    return expanded_row
        
def build_poly(x, degree):
    num_rows = x.shape[0]
    num_cols = None
    if len(x.shape) > 1:
        num_cols = x.shape[1]
    else:
        num_cols = 1
    augmented_x = np.empty((num_rows, num_cols * (degree + 1)))
    for i in range(num_rows):
        augmented_x[i] = np.fromfunction(lambda _: expand_row(x[i], degree, num_cols), (num_cols * (degree + 1), ))
    return augmented_x