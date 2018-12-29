# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""

import numpy as np

def split_data(x, y, ratio, seed=1):
    np.random.seed(seed)
    permutation = np.random.permutation(len(y))
    shuffled_x = x[permutation]
    shuffled_y = y[permutation]
    split_position = int(len(y) * ratio)
    x_training, x_test = shuffled_x[ : split_position], shuffled_x[split_position : ]
    y_training, y_test = shuffled_y[ : split_position], shuffled_y[split_position : ]
    return x_training, y_training, x_test, y_test