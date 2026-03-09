import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    if len(y) == 0:
        return 0.0
    y = np.array(y)
    _, counts = np.unique(y, return_counts = True)
    p = counts / len(y)
    return -sum(p*np.log2(p))