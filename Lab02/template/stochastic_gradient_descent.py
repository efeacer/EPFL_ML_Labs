# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

#def compute_stoch_gradient(y, tx, w): #Using MSE
    #error = y - np.matmul(tx, w) 
    #return - np.matmul(tx.T, error) / y.size 

def compute_stoch_subgradient(error): 
        if error < 0: 
            return -1.0
        elif error == 0: 
            return float(np.random.uniform(-1, 1))
        else:
            return 1.0
        
compute_stoch_subgradient = np.vectorize(compute_stoch_subgradient)

def compute_stoch_gradient(y, tx, w): #Using MAE (needs larger step size)
    error_vector = y - np.matmul(tx, w) 
    error_vector_processed = compute_stoch_subgradient(error_vector)
    return - np.matmul(tx.T, error_vector_processed) / y.size

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        stochastic_gradient = np.array([0, 0])
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            stochastic_gradient = stochastic_gradient + compute_stoch_gradient(minibatch_y, minibatch_tx, w)
        loss = compute_loss(y, tx, w)
        w =  w - gamma * stochastic_gradient
        ws.append(w)
        losses.append(loss)
    return losses, ws