import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if len(seqs) == 0:
        return np.array([], dtype=np.int64)
    if max_len is None:
        L = max(len(seq) for seq in seqs)
    else:
        L = max_len
    padded = []
    for seq in seqs:
        seq = np.array(seq)
        if len(seq) > L:
            seq = seq[:L]
        pad_with = L - len(seq)
        if pad_with > 0:
            seq = np.concatenate([seq, np.full(pad_with, pad_value)])
        padded.append(seq)
    return np.array(padded)
        