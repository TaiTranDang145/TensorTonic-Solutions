import numpy as np

def unet_decoder_block(x: np.ndarray, skip: np.ndarray, out_channels: int) -> np.ndarray:
    """
    Returns zero array with correct shape.
    """
    # Your implementation here
    B,H,W,C = x.shape
    H1 = H*2
    W1 = W*2
    C = out_channels
    H2 = H1 - 2
    W2 = W1 -2
    H3 = H2 -2
    W3 = W2 -2
    out = np.zeros((B,H3,W3,C))
    return out
    
