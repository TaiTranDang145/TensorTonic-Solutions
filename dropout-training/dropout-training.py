import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.array(x)
    values = rng.random(x.shape) if rng is not None else np.random.random(x.shape)
    dropout_pattern = np.where(values < 1 - p, 1 / (1-p), 0)
    ans = x * dropout_pattern
    return ans, dropout_pattern