import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here
    T = np.array(T)
    points = np.array(points)
    points_h = np.c_[points.reshape(-1,3),np.ones(points.reshape(-1,3).shape[0])]
    ans = (T@points_h.T).T[:,:-1]
    return ans[0] if points.ndim == 1 else ans