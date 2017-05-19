import numpy as np


def equal(A, B):
    return A.shape == B.shape and not np.any(A != B)
