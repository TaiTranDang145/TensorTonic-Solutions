import numpy as np

def skip_connection(skip, h, w):
    B,H,W,C = skip.shape
    h_start = (H - h)//2
    w_start = (W - w) //2
    return skip[:,h_start:h_start+h, w_start:w_start+w,:]


def unet_decoder_block(x: np.ndarray, skip: np.ndarray, out_channels: int) -> np.ndarray:
    """
    Returns zero array with correct shape.
    """
    # Your implementation here
    B,H,W,C = x.shape
    H1 = H*2
    W1 = W*2
    C1 = C//2
    x1 = np.zeros((B,H1,W1,C1))
    x2 = skip_connection(skip, H1,W1)
    x3 = np.zeros((B,H1,W1,C1 + x2.shape[-1]))
    
    H_out = x3.shape[1] - 4
    W_out = x3.shape[2] - 4
    out = np.zeros((B, H_out, W_out, out_channels))

    return out