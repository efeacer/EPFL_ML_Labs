# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

#def compute_loss(y, tx, w): #Using MSE
    #error_vector = y - np.matmul(tx, w) 
    #error_vector_processed = error_vector * error_vector 
    #summation_result = np.sum(error_vector_processed)
    #loss = summation_result / (2 * y.size) 
    #return loss

def compute_loss(y, tx, w): #Using MAE
    error_vector = y - np.matmul(tx, w) 
    error_vector_processed = np.absolute(error_vector)
    summation_result = np.sum(error_vector_processed)
    loss = summation_result / y.size
    return loss