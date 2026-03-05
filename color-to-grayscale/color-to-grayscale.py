import numpy as np
def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    # Write code here
    image = np.array(image)
    r = image[:,:,0]
    g = image[:,:,1]
    b = image[:,:,2]
    gray =  r*0.299 + g*0.587 + b *0.114
    return gray.tolist()