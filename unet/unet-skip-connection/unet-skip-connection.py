import numpy as np

def crop_and_concat(encoder_features: np.ndarray, decoder_features: np.ndarray) -> np.ndarray:
    """
    Crop encoder features to match decoder spatial dims, then concatenate along channels.
    """
    # Your implementation here
    b1,h1,w1,c1 = encoder_features.shape
    b2,h2,w2,c2 = decoder_features.shape
    h_center,w_center = (h1 - h2)//2, (w1 - w2)//2
    center = encoder_features[:,h_center:h_center + h2, w_center:w_center + w2,:]
    out = np.concatenate((center, decoder_features), axis = -1)
    return out