import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.array(X)
    y = np.array(y)
    b = 0.0
    N,d = X.shape
    W = np.zeros(d)
    for _ in range(steps):
        z = X @ W + b
        p = _sigmoid(z)
        w_denta = X.T @(p-y) / N
        b_denta = np.mean(p-y)
        W -= lr * w_denta
        b -= lr * b_denta
    return (W,b)