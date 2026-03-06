import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    ans = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1/np.power(base, (2*(i//2))/d_model)
    angles = pos * angle_rates
    ans[:,0::2] = np.sin(angles[:,0::2])
    ans[:,1::2] = np.cos(angles[:,1::2])
    return ans