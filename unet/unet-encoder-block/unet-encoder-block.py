import numpy as np

def unet_encoder_block(x: np.ndarray, out_channels: int) -> tuple:
    """
    Returns (pool_out, skip_out) as zero arrays with correct shapes.
    """
    # Your implementation here
    B,H,W,C = x.shape
    H1 = H-2
    W1 = W-2
    C = out_channels
    H2 = H1 - 2
    W2 = W1 - 2
    skip_out = np.zeros((B,H2,W2,C))
    H3 = H2 // 2
    W3 = W2 // 2
    poll_out = np.zeros((B,H3,W3,C))
    return((poll_out, skip_out))
    
    