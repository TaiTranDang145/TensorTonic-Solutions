import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    # Write code here
    ans = 0.0
    for i in range(len(fpr) - 1):
        ans += (tpr[i] + tpr[i+1]) * (fpr[i+1] - fpr[i])
    return ans * 1/2