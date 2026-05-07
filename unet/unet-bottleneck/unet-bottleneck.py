import numpy as np

def unet_bottleneck(x: np.ndarray, out_channels: int) -> np.ndarray:
    """
    U-Net bottleneck: double convolution at lowest resolution.
    Two 3x3 unpadded convolutions, no pooling.
    Returns zero array with correct shape.
    """
    # Your implementation here
    B,H,W,C = x.shape
    h1,w1 = H - 2, W - 2
    h2,w2 = h1 - 2, w1 - 2
    out = np.zeros((B,h2,w2,out_channels))
    return out
    