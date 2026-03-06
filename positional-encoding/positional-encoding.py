import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    ans = []
    x = d_model // 2
    for pos in range(seq_len):
        tmp = []
        for i in range(x):
            tmp.append(np.sin(pos / np.power(base, 2*i/d_model)))
            tmp.append(np.cos(pos / np.power(base, 2*i/d_model)))
        if len(tmp) < d_model:
            tmp.append(np.sin(pos / np.power(base, 2*x/d_model)))
        ans.append(tmp)
    return np.array(ans)