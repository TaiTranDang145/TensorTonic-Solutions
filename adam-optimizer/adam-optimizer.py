import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    m = np.array(m)
    grad = np.array(grad)
    v = np.array(v)
    param = np.array(param)
    m_t = beta1 * m + (1-beta1) * grad
    v_t = beta2 * v + (1-beta2) * (grad ** 2)
    m_predict = m_t / (1-beta1 ** t)
    v_predict = v_t / (1-beta2 ** t)
    p_new = param - lr*m_predict/((v_predict) ** 0.5 + eps)
    return (p_new, m_t, v_t)