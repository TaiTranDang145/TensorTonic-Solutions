import numpy as np

def skip_connection(x, target):
    b1,h1,w1,c1 = x.shape
    b2,h2,w2,c2 = target.shape
    k_h = (h1 - h2) // 2
    k_w = (w1 -w2) // 2
    out = x[:,k_h:k_h+h2,k_w:k_w+w2,:]
    return out

def encoder(x):
    b,h,w,c = x.shape
    h1 = h - 4
    w1 = w - 4
    skip = np.zeros((b,h1,w1,c*2))
    h2 = h1 // 2
    w2 = w1 // 2
    out = np.zeros((b,h2,w2,c*2))
    return out,skip

def decoder(x, skip):
    b,h,w,c = x.shape
    h1, w1 = h * 2, w * 2
    c1 = c // 2
    x1 = np.zeros((b,h1,w1,c1))
    sc = skip_connection(skip,x1)
    b2,h2,w2,c2 = sc.shape
    out = np.zeros((b2,h2-4,w2-4,c2))
    return out

def bottleneck(x):
    b,h,w,c = x.shape
    h1, w1 = h - 4, w - 4
    c1 = c * 2
    out = np.zeros((b,h1,w1,c1))
    return out

def output(x, num_classes):
    b,h,w,c = x.shape
    out = np.zeros((b,h,w,num_classes))
    return out
    
def unet(x: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """
    Complete U-Net: trace shape through 4 encoder blocks, bottleneck, 4 decoder blocks, output.
    Each block: two 3x3 unpadded convs (reduce by 4), encoder pools (halve), decoder upsamples (double).
    Returns zero array with correct output shape.
    """
    # Your implementation here
    b,h,w,c = x.shape
    x0 = np.zeros((b,h,w,64))
    x1,s1 = encoder(x0)
    x2,s2 = encoder(x1)
    x3,s3 = encoder(x2)
    x4,s4 = encoder(x3)
    x5 = bottleneck(x4)
    x6 = decoder(x5,s4)
    x7 = decoder(x6,s3)
    x8 = decoder(x7,s2)
    x9 = decoder(x8,s1)
    x10 = output(x9, num_classes)
    return x10
    
