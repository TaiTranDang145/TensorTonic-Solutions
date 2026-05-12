import numpy as np

def conv_block(x, W1, W2, Ws):
    """
    Returns: np.ndarray with sum of main path output and projected shortcut
    """
    # YOUR CODE HERE
    W1 = np.array(W1)
    W2 = np.array(W2)
    Ws = np.array(Ws)
    h = np.maximum(0, x @ W1)
    z = h @ W2
    s = x @ Ws
    y = np.maximum(0, z + s)
    return y
