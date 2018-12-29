# -*- coding: utf-8 -*-
"""Gradient Descent"""

#def compute_gradient(y, tx, w): #Using MSE
    #error_vector = y - np.matmul(tx, w) 
    #return - np.matmul(tx.T, error_vector) / y.size 
    
def compute_subgradient(error): 
        if error < 0: 
            return -1.0
        elif error == 0: 
            return float(np.random.uniform(-1, 1))
        else:
            return 1.0
        
compute_subgradient = np.vectorize(compute_subgradient)

def compute_gradient(y, tx, w): #Using MAE (needs larger step size)
    error_vector = y - np.matmul(tx, w) 
    error_vector_processed = compute_subgradient(error_vector)
    return - np.matmul(tx.T, error_vector_processed) / y.size

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * gradient
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws